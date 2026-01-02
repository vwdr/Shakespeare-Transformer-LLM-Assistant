# chat_app.py
import torch
from tokenizers import Tokenizer
import gradio as gr

from model import MiniTransformerLM, ModelConfig
from safety import is_unsafe, REFUSAL_MESSAGE

def load_instruct_model(seq_len=128):
    tokenizer = Tokenizer.from_file("tokenizer.json")
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=seq_len
    )
    model = MiniTransformerLM(config)
    model.load_state_dict(torch.load("mini_lm_instruct.pt", map_location="cpu"))
    model.eval()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_instruct_model()

@torch.no_grad()
def generate_reply(prompt, history, temperature=0.8, top_k=20, max_new_tokens=80, seq_len=128):
    # history is list of [user, assistant] pairs (gradio format)
    # Build a chat-style prompt
    convo = ""
    for user_msg, assistant_msg in history:
        convo += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
    convo += f"User: {prompt}\nAssistant: "

    if is_unsafe(prompt):
        return REFUSAL_MESSAGE

    encoded = tokenizer.encode(convo)
    ids = encoded.ids
    # cut to last seq_len-1 tokens
    ids = ids[-(seq_len-1):]

    input_ids = torch.tensor([ids], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        if input_ids.shape[1] > seq_len:
            input_ids = input_ids[:, -seq_len:]

        logits = model(input_ids)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)

        if top_k > 0:
            values, indices = torch.topk(probs, top_k)
            probs_filtered = torch.zeros_like(probs).scatter_(1, indices, values)
            probs = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        token_id = next_token.item()
        # Stop if EOS
        eos_id = tokenizer.token_to_id("[EOS]")
        if token_id == eos_id:
            break

    output_ids = input_ids[0].cpu().tolist()
    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Extract only the last assistant message by splitting
    # We know convo ended with "User: <prompt>\nAssistant: "
    if "User:" in text:
        # crude but works for toy model
        parts = text.split("User:")
        last = "User:".join(parts[-1:])
        if "Assistant:" in last:
            last = last.split("Assistant:")[-1]
        reply = last.strip()
    else:
        reply = text

    return reply

def chat_fn(message, history):
    reply = generate_reply(message, history or [])
    history = history or []
    history.append((message, reply))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("# Mini LLM Assistant (Your Model)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message")
    clear = gr.Button("Clear")

    msg.submit(chat_fn, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: ([], []), None, [chatbot, chatbot])

if __name__ == "__main__":
    demo.launch()
