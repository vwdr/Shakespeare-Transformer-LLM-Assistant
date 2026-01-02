# chat_cli.py
import torch
from tokenizers import Tokenizer

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


@torch.no_grad()
def generate_reply(
    model,
    tokenizer,
    device,
    prompt,
    history,
    temperature=0.8,
    top_k=20,
    max_new_tokens=80,
    seq_len=128,
):
    """
    Generate a reply from the assistant given the new user prompt and chat history.

    history: list of (user, assistant) tuples.
    """
    # Build the chat-style context
    convo = ""
    for user_msg, assistant_msg in history:
        convo += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
    convo += f"User: {prompt}\nAssistant: "

    if is_unsafe(prompt):
        return REFUSAL_MESSAGE

    encoded = tokenizer.encode(convo)
    ids = encoded.ids
    # keep only last seq_len-1 tokens if too long
    ids = ids[-(seq_len - 1):]

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
        eos_id = tokenizer.token_to_id("[EOS]")
        if eos_id is not None and token_id == eos_id:
            break

    output_ids = input_ids[0].cpu().tolist()
    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Try to extract only the last assistant reply
    if "User:" in text:
        parts = text.split("User:")
        last = "User:".join(parts[-1:])
        if "Assistant:" in last:
            last = last.split("Assistant:")[-1]
        reply = last.strip()
    else:
        reply = text

    return reply


def main():
    print("Loading instruction-tuned model...")
    model, tokenizer, device = load_instruct_model()
    print("Model loaded. Type 'exit' to quit.\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        reply = generate_reply(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=user_input,
            history=history,
            temperature=0.8,
            top_k=20,
            max_new_tokens=80,
            seq_len=128,
        )

        history.append((user_input, reply))
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
