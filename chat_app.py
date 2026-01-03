# chat_app.py
import re

import torch
from tokenizers import Tokenizer
import gradio as gr

from model import MiniTransformerLM, ModelConfig
from safety import is_unsafe, REFUSAL_MESSAGE

SYSTEM_PROMPT = (
    "You are a helpful assistant. Reply in a light Shakespearean tone while staying clear and relevant.\n"
)

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

def clean_text(text: str) -> str:
    """Lightly post-process model output to tidy spacing and contractions."""
    text = text.replace(" \n", "\n").replace("\n ", "\n")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    for pat, repl in [
        (" n't", "n't"),
        (" 's", "'s"),
        (" 'm", "'m"),
        (" 're", "'re"),
        (" 've", "'ve"),
        (" 'd", "'d"),
        (" 'll", "'ll"),
    ]:
        text = text.replace(pat, repl)
    return text.strip()

@torch.no_grad()
def generate_reply(prompt, history, temperature=0.6, top_k=30, max_new_tokens=80, seq_len=128):
    # history is list of [user, assistant] pairs (gradio format)
    # Build a chat-style prompt
    convo = SYSTEM_PROMPT
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
    start_len = input_ids.shape[1]
    eos_id = tokenizer.token_to_id("[EOS]")

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
        if eos_id is not None and token_id == eos_id:
            break

    output_ids = input_ids[0].cpu().tolist()
    new_token_ids = output_ids[start_len:]
    # If nothing new was generated, fall back to decoding everything
    decode_ids = new_token_ids if new_token_ids else output_ids
    text = tokenizer.decode(decode_ids, skip_special_tokens=True)

    reply = clean_text(text)
    # Strip any spurious role labels if the model emits them
    if "Assistant:" in reply:
        reply = clean_text(reply.split("Assistant:", 1)[-1])
    if "User:" in reply:
        reply = clean_text(reply.split("User:", 1)[-1])
    # Keep only the first sentence-ish fragment to avoid rambles
    parts = re.split(r"(?<=[.!?])\s+", reply)
    if parts and parts[0].strip():
        reply = parts[0].strip()
    # Hard cap length in words
    words = reply.split()
    if len(words) > 40:
        reply = " ".join(words[:40]).rstrip(",.;:") + "."

    return reply

def chat_fn(message, history):
    """
    Gradio ChatInterface callback. History comes in as list of (user, assistant) tuples.
    Return only the assistant reply; Gradio will append it to the history it manages.
    """
    reply = generate_reply(message, history or [])
    return reply

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Mini LLM Assistant (Your Model)",
    chatbot=gr.Chatbot(type="tuples", height=420),
    textbox=gr.Textbox(label="Your message", placeholder="Ask anything..."),
    clear_btn="Clear",
)

if __name__ == "__main__":
    demo.launch()
