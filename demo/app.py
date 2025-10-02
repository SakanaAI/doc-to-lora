import os
import sys
from pathlib import Path

import gradio as gr
import torch

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ctx_to_lora.data.processing import tokenize_ctx_text
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling import hypernet

sys.modules["ctx_to_lora.modeling_utils"] = hypernet

# Global state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modulated_model = None
chat_history = []
ctx_tokenizer = None
base_tokenizer = None

try:
    DEFAULT_CONTEXT = Path("data/sakana_wiki.txt").read_text(encoding="utf-8").strip()
except FileNotFoundError:
    DEFAULT_CONTEXT = ""

WARNING_MESSAGE = (
    "⚠️ **Caution**: This is an academic proof-of-concept application.\n"
    "The model may generate inaccurate information or hallucinate facts."
)

FOOTER = """
⚠️ This model is an experimental prototype and is only available for educational and research and development purposes. It is not suitable for commercial use or in environments where failure can have significant effects (mission-critical environments).
The use of this model is at the user's own risk and its performance and results is not guaranteed in any way.
Sakana AI is not responsible for any direct or indirect loss resulting from using this model, regardless of the outcome.
"""


def load_custom_chat_template(tokenizer, model_name):
    if "gemma" in model_name.lower():
        template_path = "chat_templates/google/gemma-2-2b-it.jinja"
        if os.path.exists(template_path):
            with open(template_path) as f:
                template_content = f.read()
                tokenizer.chat_template = template_content
                print(f"Loaded custom chat template from {template_path}")
                return True
    return False


def get_available_checkpoints():
    checkpoints = []
    trained_dir = Path("trained_t2l")

    if trained_dir.exists():
        for model_dir in trained_dir.iterdir():
            if model_dir.is_dir():
                for ckpt_file in model_dir.glob("**/*.bin"):
                    if "pytorch_model.bin" in ckpt_file.name:
                        checkpoints.append(str(ckpt_file))

    train_outputs = Path("train_outputs/runs")
    if train_outputs.exists():
        for run_dir in train_outputs.iterdir():
            if run_dir.is_dir():
                for ckpt_dir in run_dir.glob("checkpoint-*"):
                    ckpt_file = ckpt_dir / "pytorch_model.bin"
                    if ckpt_file.exists():
                        checkpoints.append(str(ckpt_file))

    return sorted(checkpoints) if checkpoints else ["No checkpoints found"]


def load_checkpoint(checkpoint_path: str) -> tuple[str, gr.update]:
    global modulated_model, ctx_tokenizer, base_tokenizer, chat_history

    if not checkpoint_path or checkpoint_path == "No checkpoints found":
        return "⚠️ Please select a valid checkpoint", gr.update()

    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

        state_dict = torch.load(checkpoint_path, weights_only=False)
        modulated_model = ModulatedPretrainedModel.from_state_dict(
            state_dict,
            train=False,
            use_flash_attn=True,
            use_sequence_packing=False,
        )
        modulated_model = modulated_model.to(device).to(torch.bfloat16)
        modulated_model.eval()

        ctx_encoder_model_name_or_path = (
            modulated_model.ctx_encoder_args.ctx_encoder_model_name_or_path
            or modulated_model.base_model.config.name_or_path
        )
        ctx_tokenizer = get_tokenizer(ctx_encoder_model_name_or_path, train=False)
        base_tokenizer = get_tokenizer(
            modulated_model.base_model.config.name_or_path, train=False
        )

        load_custom_chat_template(
            base_tokenizer, modulated_model.base_model.config.name_or_path
        )

        chat_history = [{"role": "system", "content": ""}]

        model_name = modulated_model.base_model.config.name_or_path
        success_msg = (
            f"✅ Successfully loaded checkpoint!\n\nBase Model: {model_name}\n\n"
            "You can now add context and start chatting."
        )
        return success_msg, gr.update(interactive=True)

    except Exception as e:
        import traceback

        error_msg = (
            f"❌ Error loading checkpoint:\n{str(e)}\n\n{traceback.format_exc()}"
        )
        print(error_msg)
        return error_msg, gr.update(interactive=False)


def process_multiple_contexts(contexts: list[str]) -> dict:
    contexts = [ctx for ctx in contexts if ctx.strip()]
    if not contexts:
        contexts = [""]
    print(f"Processing {len(contexts)} non-empty contexts")
    tokenized_contexts = tokenize_ctx_text({"context": contexts}, ctx_tokenizer)
    ctx_ids = tokenized_contexts["ctx_ids"]
    ctx_ids = [
        torch.tensor(ctx_id, dtype=torch.long, device=device) for ctx_id in ctx_ids
    ]
    ctx_attn_mask = [torch.ones_like(ids) for ids in ctx_ids]
    ctx_attn_mask = [
        torch.tensor(mask, dtype=torch.long, device=device) for mask in ctx_attn_mask
    ]
    ctx_ids = torch.nn.utils.rnn.pad_sequence(
        ctx_ids,
        batch_first=True,
        padding_value=0,
    )
    ctx_attn_mask = torch.nn.utils.rnn.pad_sequence(
        ctx_attn_mask,
        batch_first=True,
        padding_value=0,
    )
    return {"ctx_ids": ctx_ids, "ctx_attn_mask": ctx_attn_mask}


def add_user_message(message: str, history):
    if not message.strip():
        return history, ""
    return history + [[message, None]], ""


def generate_response(
    history: list[list[str]],
    system_msg: str,
    context_1: str,
    scaler_1: float,
    context_2: str,
    scaler_2: float,
    context_3: str,
    scaler_3: float,
    bias_scaler: float,
):
    global modulated_model, chat_history, ctx_tokenizer, base_tokenizer

    if modulated_model is None:
        history[-1][1] = "Please load a checkpoint first."
        yield history
        return

    if not history or history[-1][0] is None:
        yield history
        return

    try:
        user_message = history[-1][0]

        if system_msg.strip() and chat_history[0]["role"] == "system":
            chat_history[0]["content"] = system_msg.strip()

        chat_history.append({"role": "user", "content": user_message})

        contexts = []
        scalers = []

        if context_1 and context_1.strip():
            contexts.append(context_1)
            scalers.append(scaler_1)
        if context_2 and context_2.strip():
            contexts.append(context_2)
            scalers.append(scaler_2)
        if context_3 and context_3.strip():
            contexts.append(context_3)
            scalers.append(scaler_3)

        if not contexts:
            contexts = [""]
            scalers = [1.0]

        print(f"Processing {len(contexts)} contexts with scalers: {scalers}")
        print(f"Bias scaler: {bias_scaler}")

        with torch.inference_mode(), torch.amp.autocast(str(device)):
            ctx_inputs = process_multiple_contexts(contexts)
            ctx_ids = ctx_inputs["ctx_ids"].to(device)
            ctx_attn_mask = ctx_inputs["ctx_attn_mask"].to(device)

            scalers_tensor = torch.tensor(scalers, dtype=torch.float32, device=device)

            model_inputs = base_tokenizer.apply_chat_template(
                chat_history, return_tensors="pt", add_generation_prompt=True
            ).to(device)

            print(f"Contexts: {contexts}")
            print(f"Chat history: {chat_history}")

            outputs = modulated_model.generate(
                ctx_ids=ctx_ids,
                ctx_attn_mask=ctx_attn_mask,
                n_ctx_chunks=torch.tensor([len(ctx_ids)], device=ctx_ids.device),
                scalers=scalers_tensor,
                bias_scaler=bias_scaler,
                input_ids=model_inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0,
            )

            response = base_tokenizer.decode(
                outputs[0][model_inputs.shape[1] :], skip_special_tokens=True
            )

            chat_history.append({"role": "assistant", "content": response})

            words = response.split()
            partial_response = ""
            for word in words:
                partial_response += word + " "
                history[-1][1] = partial_response.strip()
                yield history

            history[-1][1] = response
            yield history

    except Exception as e:
        import traceback

        error_msg = f"❌ Error: {str(e)}"
        print(f"Error generating response: {str(e)}\n\n{traceback.format_exc()}")
        history[-1][1] = error_msg
        yield history


def reset_chat(system_msg: str):
    global chat_history
    chat_history = [
        {"role": "system", "content": system_msg.strip() if system_msg else ""}
    ]
    return [[None, WARNING_MESSAGE]], "Chat history reset successfully!"


def update_context_display(num_contexts: int):
    updates = []
    for i in range(3):
        updates.append(gr.update(visible=(i < num_contexts)))
    return updates


custom_css = """
:root {
    color-scheme: light;
}

.gradio-container {
    font-family: 'Inter', sans-serif;
}

.chat-container {
    border-radius: 10px;
    border: 1px solid #e0e0e0;
}

.context-field {
    background-color: #f9f9f9;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    border: 1px solid #e0e0e0;
}

.status-box {
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}

.primary-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 6px;
    color: white;
    font-weight: 600;
}

.secondary-button {
    background-color: #607d8b;
    border: none;
    border-radius: 6px;
    color: white;
}

#chatbot {
    height: 500px;
}

.instruction-text {
    font-style: italic;
    color: #666;
    font-size: 0.9em;
    margin-bottom: 10px;
}

.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 12px;
    margin: 10px 0;
    color: #856404;
    font-size: 0.95em;
}

.warning-box strong {
    color: #d97706;
}
"""


def create_demo():
    with gr.Blocks(
        title="Context-to-LoRA Chat Interface",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as demo:
        gr.Markdown(
            """
            # 🎯 Context-to-LoRA Chat Interface
            
            Load a hypernetwork checkpoint and chat with a context-modulated language model.
            Add multiple contexts with individual scaling parameters to influence the model's responses.
            """
        )

        gr.Markdown(
            """
            ### 📖 Usage Instructions
            
            1. **Load a Checkpoint**: Select a hypernetwork checkpoint from the dropdown and click "Load Checkpoint"
            2. **Configure Contexts**: 
               - Use the slider to choose how many contexts you want (0-3)
               - Enter your context information in each text field
               - Adjust the scaling sliders to control the influence of each context
            3. **Set Bias Scaler**: Adjust the bias scaler to control overall model behavior
            4. **Start Chatting**: Once the model is loaded, type your message and press Shift+Enter or click Send
            5. **Reset**: Use the "Reset Chat" button to start a new conversation
            
            💡 **Tip**: You can use context to provide background information or specific knowledge
            that should influence the model's responses.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📦 Load Checkpoint")
                gr.Markdown(
                    "*Select a trained hypernetwork checkpoint to begin.*",
                    elem_classes="instruction-text",
                )

                checkpoint_dropdown = gr.Dropdown(
                    choices=get_available_checkpoints(),
                    label="Select Checkpoint",
                    value=None,
                    interactive=True,
                )

                load_btn = gr.Button("Load Checkpoint", variant="primary", size="lg")

                status_box = gr.Textbox(
                    label="Status",
                    lines=8,
                    interactive=False,
                    elem_classes="status-box",
                )

                gr.Markdown("---")

                gr.Markdown("### 📝 Context Configuration")
                gr.Markdown(
                    "*Add contexts to influence the model's responses. Each context can be scaled independently.*",
                    elem_classes="instruction-text",
                )

                num_contexts = gr.Slider(
                    minimum=0,
                    maximum=3,
                    step=1,
                    value=1,
                    label="Number of Contexts",
                    interactive=True,
                )

                with gr.Group(visible=True) as context_1_group:
                    context_1 = gr.Textbox(
                        label="Context 1",
                        placeholder="Enter context information here...",
                        lines=4,
                        elem_classes="context-field",
                        value=DEFAULT_CONTEXT,
                    )
                    scaler_1 = gr.Slider(
                        minimum=-2.0,
                        maximum=2.0,
                        step=0.01,
                        value=1.0,
                        label="Context 1 Scaling",
                    )

                with gr.Group(visible=False) as context_2_group:
                    context_2 = gr.Textbox(
                        label="Context 2",
                        placeholder="Enter additional context information...",
                        lines=4,
                        elem_classes="context-field",
                    )
                    scaler_2 = gr.Slider(
                        minimum=-2.0,
                        maximum=2.0,
                        step=0.01,
                        value=1.0,
                        label="Context 2 Scaling",
                    )

                with gr.Group(visible=False) as context_3_group:
                    context_3 = gr.Textbox(
                        label="Context 3",
                        placeholder="Enter additional context information...",
                        lines=4,
                        elem_classes="context-field",
                    )
                    scaler_3 = gr.Slider(
                        minimum=-2.0,
                        maximum=2.0,
                        step=0.01,
                        value=1.0,
                        label="Context 3 Scaling",
                    )

                gr.Markdown("---")

                bias_scaler = gr.Slider(
                    minimum=-2.0,
                    maximum=2.0,
                    step=0.01,
                    value=1.0,
                    label="Bias Scaler",
                    info="A single scalar applied to bias parameters (independent of contexts)",
                )

            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat Interface")

                system_msg = gr.Textbox(
                    label="System Message",
                    placeholder="Enter system message (optional)...",
                    lines=2,
                )

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    elem_id="chatbot",
                    elem_classes="chat-container",
                    value=[[None, WARNING_MESSAGE]],
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here... (Shift+Enter for new line)",
                        lines=2,
                        scale=4,
                        interactive=False,
                    )
                    send_btn = gr.Button(
                        "Send", variant="primary", scale=1, interactive=False
                    )

                with gr.Row():
                    clear_btn = gr.Button("🔄 Reset Chat", variant="secondary")
                    reset_status = gr.Textbox(label="Reset Status", visible=False)

        load_btn.click(
            fn=load_checkpoint,
            inputs=[checkpoint_dropdown],
            outputs=[status_box, msg],
        ).then(fn=lambda: gr.update(interactive=True), outputs=[send_btn])

        num_contexts.change(
            fn=update_context_display,
            inputs=[num_contexts],
            outputs=[context_1_group, context_2_group, context_3_group],
        )

        msg.submit(
            fn=add_user_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
        ).then(
            fn=generate_response,
            inputs=[
                chatbot,
                system_msg,
                context_1,
                scaler_1,
                context_2,
                scaler_2,
                context_3,
                scaler_3,
                bias_scaler,
            ],
            outputs=[chatbot],
        )

        send_btn.click(
            fn=add_user_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
        ).then(
            fn=generate_response,
            inputs=[
                chatbot,
                system_msg,
                context_1,
                scaler_1,
                context_2,
                scaler_2,
                context_3,
                scaler_3,
                bias_scaler,
            ],
            outputs=[chatbot],
        )

        clear_btn.click(
            fn=reset_chat,
            inputs=[system_msg],
            outputs=[chatbot, reset_status],
        )

        gr.Markdown(f"---\n{FOOTER.strip()}")
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
    )
