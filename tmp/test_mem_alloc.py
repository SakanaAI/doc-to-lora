import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def bytes_to_mib(x: int) -> float:
    return x / (1024**2)


def bytes_to_gib(x: int) -> float:
    return x / (1024**3)


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        # Prefer bfloat16 if supported, else float16, else float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def main():
    model_name = os.environ.get("MODEL_NAME", "google/gemma-2-2b-it")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype()

    if device.type != "cuda":
        print(
            "CUDA is not available; this script is intended for GPU memory measurements."
        )
        return

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Load tokenizer on CPU (doesn't affect CUDA mem)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Measure peak memory during model load
    torch.cuda.reset_peak_memory_stats(device)
    load_start = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,  # load on CPU first, then move to desired single device
    )
    model.to(device)
    model.eval()

    torch.cuda.synchronize()
    model_load_peak = torch.cuda.max_memory_allocated(device)
    load_elapsed = time.time() - load_start

    # Baseline after load
    baseline_alloc = torch.cuda.memory_allocated(device)

    # Prepare a chat-style prompt if available; fallback to plain text
    doc = """
Matryoshka dolls (Russian: матрёшка, romanized: matryoshka/ˌmætriˈɒʃkə/), also known as stacking dolls, nesting dolls, Russian tea dolls, or Russian dolls,[1] are a set of wooden dolls of decreasing size placed one inside another. The name Matryoshka is a diminutive form of Matryosha (Матрёша), in turn a hypocorism of the Russian female first name Matryona (Матрёна).[2]

A set of matryoshkas consists of a wooden figure, which separates at the middle, top from bottom, to reveal a smaller figure of the same sort inside, which has, in turn, another figure inside of it, and so on.

The first Russian nested doll set was made in 1890 by woodturning craftsman and wood carver Vasily Zvyozdochkin from a design by Sergey Malyutin, who was a folk crafts painter at Abramtsevo. Traditionally the outer layer is a woman, dressed in a Russian sarafan dress. The figures inside may be of any gender; the smallest, innermost doll is typically a baby turned from a single piece of wood. Much of the artistry is in the painting of each doll, which can be very elaborate. The dolls often follow a theme; the themes may vary, from fairy tale characters to Soviet leaders. In some countries, matryoshka dolls are often referred to as babushka dolls, though they are not known by this name in Russian; babushka (бабушка) means 'grandmother; old woman'.[3]
History
The original matryoshka set by Zvyozdochkin and Malyutin, 1892

The first Russian nested doll set was carved in 1890 at the Children's Education Workshop by Vasily Zvyozdochkin and designed by Sergey Malyutin, who was a folk crafts painter in the Abramtsevo estate of Savva Mamontov, a Russian industrialist and patron of arts.[4][5] Mamontov's brother, Anatoly Ivanovich Mamontov (1839–1905), created the Children's Education Workshop to make and sell children's toys. The doll set was painted by Malyutin. Malyutin's doll set consisted of eight dolls—the outermost was a mother in a traditional dress holding a red-combed rooster. The inner dolls were her children, girls and a boy, and the innermost a baby. The Children's Education Workshop was closed in the late 1890s, but the tradition of the matryoshka simply relocated to Sergiyev Posad, the Russian city known as a toy-making center since the fourteenth century.[6][4]

The inspiration for matryoshka dolls is not clear. Matryoshka dolls may have been inspired by a nesting doll imported from Japan.[5][7] The Children's Education workshop where Zvyozdochkin was a lathe operator received a five piece, cylinder-shaped nesting doll featuring Fukuruma (Fukurokuju) in the late 1890s,[8] which is now part of the collection at the Sergiev Posad Museum of Toys.[8] Other east Asian dolls share similarities with matryoshka dolls such as the Kokeshi dolls,[4][9] originating in Northern Honshū, the main island of Japan, although they cannot be placed one inside another, and the round hollow daruma doll depicting a Buddhist monk.[9][10] Another possible source of inspiration is the nesting Easter eggs produced on a lathe by Russian woodworkers during the late 19th Century.[3][11]

Savva Mamontov's wife presented a set of matryoshka dolls at the Exposition Universelle in Paris in 1900, and the toy earned a bronze medal. Soon after, matryoshka dolls were being made in several places in Russia and shipped around the world. 
    """
    messages = [
        {
            "role": "user",
            "content": doc + "\n\nSummarize the article.",
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # Optional quick warmup to stabilize kernels (doesn't affect peak calc since we reset after)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1)
        torch.cuda.synchronize()

    # Measure generation peak memory for 512 new tokens
    torch.cuda.reset_peak_memory_stats(device)
    gen_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", 512)),
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    gen_elapsed = time.time() - gen_start

    gen_peak_total = torch.cuda.max_memory_allocated(
        device
    )  # includes baseline present at reset
    gen_additional = max(0, gen_peak_total - baseline_alloc)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print summary
    print("=== Memory allocation summary (CUDA) ===")
    print(f"Device: {device} | {torch.cuda.get_device_name(device)}")
    print(f"Model: {model_name}")
    print(f"DType: {dtype}")
    print()
    print("Model load:")
    print(
        f"  Peak allocated during load: {bytes_to_mib(model_load_peak):.2f} MiB ({bytes_to_gib(model_load_peak):.2f} GiB)"
    )
    print(f"  Time to load + move to device: {load_elapsed:.2f}s")
    print(
        f"  Baseline after load (current allocated): {bytes_to_mib(baseline_alloc):.2f} MiB ({bytes_to_gib(baseline_alloc):.2f} GiB)"
    )
    print()
    print("Generation (max_new_tokens=512):")
    print(
        f"  Peak allocated total during generation: {bytes_to_mib(gen_peak_total):.2f} MiB ({bytes_to_gib(gen_peak_total):.2f} GiB)"
    )
    print(
        f"  Additional peak over baseline during generation: {bytes_to_mib(gen_additional):.2f} MiB ({bytes_to_gib(gen_additional):.2f} GiB)"
    )
    print(f"  Time to generate: {gen_elapsed:.2f}s")
    print()
    # Response preview
    preview = decoded
    if len(preview) > 800:
        preview = preview[:800] + "\n... [truncated]"
    print("=== Response preview ===")
    print(preview)


if __name__ == "__main__":
    main()
