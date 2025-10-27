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
    "⚠️ **Caution**: This is an academic proof-of-concept demonstration.\n"
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


def load_checkpoint(
    checkpoint_path: str,
) -> tuple[str, gr.update, gr.update, gr.update, gr.update]:
    global modulated_model, ctx_tokenizer, base_tokenizer, chat_history

    if not checkpoint_path or checkpoint_path == "No checkpoints found":
        return (
            "⚠️ Please select a valid checkpoint",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

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
        return (
            success_msg,
            gr.update(interactive=True),  # msg
            gr.update(interactive=True),  # send_btn
            gr.update(interactive=True),  # system_msg
            gr.update(interactive=True),  # clear_btn
        )

    except Exception as e:
        import traceback

        error_msg = (
            f"❌ Error loading checkpoint:\n{str(e)}\n\n{traceback.format_exc()}"
        )
        print(error_msg)
        return (
            error_msg,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )


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
    context_4: str,
    scaler_4: float,
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
        if context_4 and context_4.strip():
            contexts.append(context_4)
            scalers.append(scaler_4)

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
    for i in range(4):
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
    border: 2px solid #d1d5db;
}

.context-field {
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    border: 2px solid #d1d5db;
}

.status-box {
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    border: 2px solid #e5e7eb;
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
    border: 2px solid #ffc107;
    border-radius: 8px;
    padding: 12px;
    margin: 10px 0;
    color: #856404;
    font-size: 0.95em;
}

.warning-box strong {
    color: #d97706;
}

.disabled-overlay {
    background-color: #f5f5f5;
    border: 3px dashed #999;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    color: #999;
}

.chat-disabled-notice {
    border: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    color: #92400e;
    font-weight: 600;
    text-align: center;
    background-color: #fef3c7;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.internalization-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 15px;
    text-align: center;
    font-weight: 600;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    border: 3px solid #5568d3;
}

.internalization-banner h3 {
    margin: 0 0 10px 0;
    font-size: 1.2em;
    color: white;
}

.internalization-banner p {
    margin: 5px 0;
    font-size: 0.95em;
    font-weight: 400;
    color: white;
}

.context-section-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: 3px solid #5568d3;
    border-left: 6px solid #4c51bf;
    padding: 15px;
    margin-bottom: 15px;
    border-radius: 6px;
    color: white;
    box-shadow: 0 2px 6px rgba(102, 126, 234, 0.3);
}

.context-section-header strong,
.context-section-header small {
    color: white;
}

.chat-section-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: 3px solid #2563eb;
    border-left: 6px solid #1d4ed8;
    padding: 15px;
    margin-bottom: 15px;
    border-radius: 6px;
    color: white;
    box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);
}

.chat-section-header strong,
.chat-section-header small {
    color: white;
}

.panel-box {
    background-color: rgba(249, 250, 251, 0.5);
    border: 2px solid rgba(209, 213, 219, 0.5);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(10px);
}

.chat-panel-box {
    background-color: rgba(249, 250, 251, 0.5);
    border: 2px solid rgba(209, 213, 219, 0.5);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(10px);
}

/* Dark mode support */
.dark .panel-box,
.dark .chat-panel-box {
    background-color: rgba(31, 41, 55, 0.5);
    border: 2px solid rgba(75, 85, 99, 0.5);
}

.dark .context-field,
.dark .chat-container {
    border-color: rgba(75, 85, 99, 0.5);
}

.dark .status-box {
    border-color: rgba(55, 65, 81, 0.5);
}
"""


def create_demo():
    with gr.Blocks(
        title="Doc-to-LoRA Chat Interface",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as demo:
        gr.Markdown(
            """
            # 📜 Doc-to-LoRA Chat Interface
            
            Load a hypernetwork checkpoint and chat with a context-modulated language model.
            Add multiple contexts with individual scaling parameters to influence the model's responses.
            """
        )

        gr.HTML(
            """
            <div class="internalization-banner">
                <h3>🧠 How Context Internalization Works</h3>
                <p>📥 Contexts are processed by the hypernetworkto dynamically modulate the base model's parameters</p>
                <p>🚫 Contexts are NOT passed as text to the base model — they influence behavior internally</p>
                <p>💬 Only your chat messages (below) are sent to the language model</p>
            </div>
            """
        )

        gr.Markdown(
            """
            ### 📖 Usage Instructions
            
            1. **Load a Checkpoint**: Select a hypernetwork checkpoint from the dropdown and click "Load Checkpoint"
            2. **Configure Contexts**: 
               - Use the slider to choose how many contexts you want (1-4)
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
            with gr.Column(scale=1, elem_classes="panel-box"):
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

                gr.HTML(
                    """
                    <div class="context-section-header">
                        <strong>🧠 Context Internalization (Hypernetwork Input)</strong><br>
                        <small>These contexts modulate the model internally — they are NOT shown to the base model</small>
                    </div>
                    """
                )

                num_contexts = gr.Slider(
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=1,
                    label="Number of Contexts to Internalize",
                    interactive=True,
                )

                with gr.Group(visible=True) as context_1_group:
                    context_1 = gr.Textbox(
                        label="🧠 Context 1 (Internalized via Hypernetwork)",
                        placeholder="Enter context to be internalized by the hypernetwork...",
                        lines=4,
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
                        label="🧠 Context 2 (Internalized via Hypernetwork)",
                        placeholder="Enter additional context to be internalized...",
                        lines=4,
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
                        label="🧠 Context 3 (Internalized via Hypernetwork)",
                        placeholder="Enter additional context to be internalized...",
                        lines=4,
                    )
                    scaler_3 = gr.Slider(
                        minimum=-2.0,
                        maximum=2.0,
                        step=0.01,
                        value=1.0,
                        label="Context 3 Scaling",
                    )

                with gr.Group(visible=False) as context_4_group:
                    context_4 = gr.Textbox(
                        label="🧠 Context 4 (Internalized via Hypernetwork)",
                        placeholder="Enter additional context to be internalized...",
                        lines=4,
                    )
                    scaler_4 = gr.Slider(
                        minimum=-2.0,
                        maximum=2.0,
                        step=0.01,
                        value=1.0,
                        label="Context 4 Scaling",
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

            with gr.Column(scale=2, elem_classes="chat-panel-box"):
                gr.HTML(
                    """
                    <div class="chat-section-header">
                        <strong>💬 Chat Interface (Direct Input to Base Model)</strong><br>
                        <small>Your messages here are the ONLY text the base model sees — contexts above influence it internally</small>
                    </div>
                    """
                )

                chat_status_notice = gr.HTML(
                    """
                    <div class="chat-disabled-notice">
                        🔒 <strong>Chat Disabled:</strong> Please load a checkpoint first to enable chat functionality.
                    </div>
                    """,
                    visible=True,
                )

                system_msg = gr.Textbox(
                    label="System Message (Optional - Sent to Base Model)",
                    placeholder="Load a checkpoint to enable chat...",
                    lines=2,
                    interactive=False,
                )

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    elem_id="chatbot",
                    elem_classes="chat-container",
                    value=[
                        [
                            None,
                            "🔒 Chat is currently disabled. Please load a checkpoint from the left panel to begin chatting.",
                        ]
                    ],
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Message (Sent Directly to Base Model)",
                        placeholder="⚠️ Load a checkpoint first to start chatting...",
                        lines=2,
                        scale=4,
                        interactive=False,
                    )
                    send_btn = gr.Button(
                        "🔒 Send (Disabled)",
                        variant="primary",
                        scale=1,
                        interactive=False,
                    )

                with gr.Row():
                    clear_btn = gr.Button(
                        "🔒 Reset Chat (Disabled)",
                        variant="secondary",
                        interactive=False,
                    )
                    reset_status = gr.Textbox(label="Reset Status", visible=False)

        load_btn.click(
            fn=load_checkpoint,
            inputs=[checkpoint_dropdown],
            outputs=[status_box, msg, send_btn, system_msg, clear_btn],
        ).then(
            fn=lambda: (
                gr.update(visible=False),
                gr.update(
                    placeholder="Type your message here... (Shift+Enter for new line)"
                ),
                gr.update(value="Send"),
                gr.update(value="🔄 Reset Chat"),
                gr.update(value=[[None, WARNING_MESSAGE]]),
            ),
            outputs=[chat_status_notice, msg, send_btn, clear_btn, chatbot],
        )

        num_contexts.change(
            fn=update_context_display,
            inputs=[num_contexts],
            outputs=[
                context_1_group,
                context_2_group,
                context_3_group,
                context_4_group,
            ],
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
                context_4,
                scaler_4,
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
                context_4,
                scaler_4,
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
