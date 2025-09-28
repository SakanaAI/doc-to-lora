if __name__ == "__main__":
    # New imports for analysis
    from collections import defaultdict
    from functools import partial

    import torch
    from peft import get_peft_model
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        TSNE = None
    try:
        import umap
    except ImportError:
        umap = None
    # Added plotting imports
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    from ctx_to_lora.data.collator import generation_collator
    from ctx_to_lora.data.definitions import CTX_AFFIXES
    from ctx_to_lora.data.processing import get_tokenized_dataset
    from ctx_to_lora.data.self_gen_template import SELF_QA_INTX
    from ctx_to_lora.model_loading import get_lora_config, get_model_and_tokenizer

    # from ctx_to_lora.modeling.context_distillation import CtxDistillModel
    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

    model_name_or_path = "google/gemma-2-2b-it"
    base_model, tokenizer = get_model_and_tokenizer(
        model_name_or_path, train=False, requires_grad=False
    )
    ds = get_tokenized_dataset(
        ds_name="squad",
        split="test",
        max_qas_len=-1,
        max_qas_per_sample=1,
        base_model_max_len=8192,
        tokenizer=tokenizer,
        ctx_model_max_len=8192,
        ctx_tokenizer=tokenizer,
        max_ctx_chunk_len=-1,
        min_ctx_chunk_len=-1,
        num_chunk_probs=None,
        max_ctx_chunk_num=None,
        add_ctx_to_chat=False,
        use_kl_loss=True,
        max_new_tokens=0,
        add_self_distill_template=False,
        set_format="pt",
    )
    ds = ds.take(100)

    peft_config = get_lora_config(
        model_name_or_path,
        lora_r=8,
        lora_dropout=0,
        target_modules=["down_proj"],
    )
    peft_config.lora_alpha = 16
    peft_model = get_peft_model(base_model, peft_config)
    sep_seq = (
        tokenizer(
            SELF_QA_INTX.strip("\n"),
            add_special_tokens=False,
            return_tensors="pt",
        )
        .input_ids[0]
        .to(base_model.device)
    )
    ctx_distill_kwargs = dict(
        prefix_tokens=torch.tensor(
            CTX_AFFIXES[model_name_or_path]["prefix"], device=base_model.device
        ),
        ctx_inp_sep_seq=sep_seq,
        pad_token_id=tokenizer.pad_token_id,
        update_iterations=100,
        tokenizer=tokenizer,
        reprompt_ctx=False,
    )

    # if args.cd_use_gen_q:
    #     q_model, q_tokenizer = get_model_and_tokenizer(
    #         "google/gemma-3-4b-it",
    #         train=False,
    #         requires_grad=False,
    #     )
    #     ctx_distill_kwargs["q_model"] = q_model
    #     ctx_distill_kwargs["q_tokenizer"] = q_tokenizer
    # model = CtxDistillModel(peft_model, **ctx_distill_kwargs)
    checkpoint_path = "train_outputs/runs/Sep06_12-27-02_slurm0-a3nodeset-11_84523_bbcb67ca/checkpoint-40000/pytorch_model.bin"
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict=torch.load(checkpoint_path, weights_only=False),
        train=False,
        base_model_kwargs=dict(attn_implementation="flash_attention_2"),
        use_flash_attn=True,
        use_sequence_packing=False,  # for generation
    )
    dataloader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=partial(generation_collator, tokenizer=tokenizer),
        pin_memory=False,
        drop_last=False,
        timeout=0,
        persistent_workers=False,
    )

    # Control / config variables for analysis
    collect_max_samples = None  # set int to limit samples processed
    modules_filter_substr = None  # e.g. "down_proj" to restrict; or None for all
    perform_analysis = True  # set False to only collect
    save_dir = "./lora_weight_analysis"  # will be created if not exists
    use_fp32_storage = False  # store ΔW in float32 (else float16 if original)
    apply_pca_before_tsne = True
    pca_dim = 64
    random_seed = 42
    perform_plotting = True  # NEW: set False to skip figure generation

    # Data structures:
    # delta_w_flat[module] -> list[Tensor(num_params,)]  (one per sample)
    delta_w_flat = defaultdict(list)
    sample_ids = []  # overall sample index order
    # Keep raw shapes (out_features, in_features)
    delta_w_shapes = {}

    # Precompute scaling factor (α / r)
    lora_alpha = peft_config.lora_alpha
    lora_r = peft_config.r
    lora_scaling = lora_alpha / lora_r

    for sample_idx, sample in tqdm(enumerate(dataloader)):
        if collect_max_samples is not None and sample_idx >= collect_max_samples:
            break
        for k, v in sample.items():
            sample[k] = v.to(model.device)
        with torch.no_grad():
            lora_state, _ = model.generate_weights(
                sample["ctx_ids"], sample["ctx_attn_mask"]
            )

        # # NEW: lora_state is now a nested dict:
        # # { module_name: { "A": [1, n_layers, r, d_in], "B": [1, n_layers, r, d_out] } }
        for module_name, parts in lora_state.items():
            if "A" not in parts or "B" not in parts:
                continue
            A_all = parts["A"].squeeze(0).float()  # [n_layers, r, d_in]
            B_all = parts["B"].squeeze(0).float()  # [n_layers, r, d_out]
            n_layers_local = A_all.size(0)
            for layer_idx in range(n_layers_local):
                A = A_all[layer_idx]  # [r, d_in]
                B = B_all[layer_idx]  # [r, d_out]
                # Original expectation was B: [d_out, r]; now transpose
                delta_w = B.T @ A  # [d_out, d_in]
                mod_key = f"{module_name}.layer{layer_idx}"
                if modules_filter_substr and modules_filter_substr not in mod_key:
                    continue
                delta_w_shapes.setdefault(mod_key, delta_w.shape)
                if use_fp32_storage:
                    delta_w = delta_w.float()
                delta_w_flat[mod_key].append(delta_w.reshape(-1).cpu())
        sample_ids.append(sample_idx)

    if perform_analysis and len(delta_w_flat):
        import os

        os.makedirs(save_dir, exist_ok=True)
        torch.manual_seed(random_seed)

        # Optional PCA helper
        def run_pca(x, k):
            # x: (n, d)
            if k >= x.shape[1]:
                return x
            # Center
            mean = x.mean(0, keepdim=True)
            x0 = x - mean
            # SVD economy
            try:
                U, S, Vt = torch.linalg.svd(x0, full_matrices=False)
            except RuntimeError:
                # fallback CPU
                U, S, Vt = torch.linalg.svd(x0.cpu(), full_matrices=False)
                U, S, Vt = U.to(x0.device), S.to(x0.device), Vt.to(x0.device)
            comp = Vt[:k]
            return torch.matmul(x0, comp.T)

        analysis_summary = {}

        for module_name, vec_list in delta_w_flat.items():
            X = torch.stack(vec_list)  # (num_samples, num_params)
            # Cosine similarity matrix
            X_norm = torch.nn.functional.normalize(X, dim=1)
            cosine_sim = X_norm @ X_norm.T  # (N,N)
            print(cosine_sim)

            # Dimensionality reduction inputs
            X_dr = X
            if apply_pca_before_tsne and TSNE is not None:
                # Keep at most pca_dim or (num_samples - 1, num_params)
                k = min(pca_dim, X.shape[1])
                X_dr = run_pca(X.to(torch.float32), k)

            embeddings = {}
            if TSNE is not None and X.shape[0] > 2:
                try:
                    perplexity = max(2, min(30, X.shape[0] - 1))
                    tsne_2d = TSNE(
                        n_components=2,
                        perplexity=perplexity,
                        random_state=random_seed,
                        init="pca",
                        learning_rate="auto",
                    ).fit_transform(X_dr.numpy())
                    embeddings["tsne"] = torch.from_numpy(tsne_2d)
                except Exception as e:
                    print(f"[WARN] TSNE failed for {module_name}: {e}")

            if umap is not None and X.shape[0] > 2:
                try:
                    reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=min(15, max(2, X.shape[0] - 1)),
                        random_state=random_seed,
                        metric="cosine",
                    )
                    umap_2d = reducer.fit_transform(X_dr.numpy())
                    embeddings["umap"] = torch.from_numpy(umap_2d)
                except Exception as e:
                    print(f"[WARN] UMAP failed for {module_name}: {e}")

            # Save tensors
            base_fname = module_name.replace(".", "_")
            torch.save(
                {
                    "module": module_name,
                    "sample_ids": sample_ids,
                    "delta_W_flat": X,  # shape (N, P)
                    "shape_matrix": delta_w_shapes[module_name],
                    "cosine_sim": cosine_sim,  # (N, N)
                    "embeddings": embeddings,  # dict
                    "config": {
                        "lora_alpha": lora_alpha,
                        "lora_r": lora_r,
                        "scaling": lora_scaling,
                        "pca_applied": apply_pca_before_tsne and TSNE is not None,
                    },
                },
                os.path.join(save_dir, f"{base_fname}.pt"),
            )

            analysis_summary[module_name] = {
                "num_samples": X.shape[0],
                "num_params": X.shape[1],
                "saved_file": os.path.join(save_dir, f"{base_fname}.pt"),
                "have_tsne": "tsne" in embeddings,
                "have_umap": "umap" in embeddings,
            }

            # -------- NEW PLOTTING SECTION --------
            if perform_plotting and plt is not None:
                # Cosine similarity heatmap
                try:
                    fig, ax = plt.subplots(figsize=(4, 4))

                    im = ax.imshow(cosine_sim.numpy(), cmap="viridis")
                    ax.set_xticks(range(len(sample_ids)))
                    ax.set_yticks(range(len(sample_ids)))
                    ax.set_xticklabels(sample_ids, fontsize=6)
                    ax.set_yticklabels(sample_ids, fontsize=6)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(
                        f"Cosine Sim: {module_name.split('.')[-3:]}".replace(",", "")
                    )
                    ax.set_xlabel("Sample")
                    ax.set_ylabel("Sample")
                    fig.tight_layout()
                    fig.savefig(
                        os.path.join(save_dir, f"{base_fname}_cosine.png"), dpi=300
                    )
                    plt.close(fig)
                except Exception as e:
                    print(f"[WARN] Cosine plot failed for {module_name}: {e}")

                # TSNE / UMAP scatter
                def _scatter(emb_key):
                    emb = embeddings.get(emb_key)
                    if emb is None:
                        return
                    try:
                        emb_np = emb.numpy()
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.scatter(
                            emb_np[:, 0],
                            emb_np[:, 1],
                            s=30,
                            c=range(len(emb_np)),
                            cmap="tab10",
                        )
                        if len(emb_np) <= 20:
                            for i, (xv, yv) in enumerate(emb_np):
                                ax.text(xv, yv, str(sample_ids[i]), fontsize=6)
                        ax.set_title(f"{emb_key.upper()} {module_name.split('.')[-3:]}")
                        ax.set_xticks([])
                        ax.set_yticks([])
                        fig.tight_layout()
                        fig.savefig(
                            os.path.join(save_dir, f"{base_fname}_{emb_key}.png"),
                            dpi=150,
                        )
                        plt.close(fig)
                    except Exception as e:
                        print(f"[WARN] {emb_key} plot failed for {module_name}: {e}")

                _scatter("tsne")
                _scatter("umap")
            # -------- END NEW PLOTTING SECTION --------

        # Print a concise summary
        print("LoRA ΔW analysis summary:")
        for m, info in analysis_summary.items():
            print(
                f"{m}: samples={info['num_samples']} params={info['num_params']} "
                f"tsne={info['have_tsne']} umap={info['have_umap']} -> {info['saved_file']}"
            )
