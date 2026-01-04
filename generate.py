import torch
from tokenizers import Tokenizer

from model import MiniTransformerLM, ModelConfig

def load_model(seq_len=128):
    tokenizer = Tokenizer.from_file("tokenizer.json")
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=seq_len
    )
    model = MiniTransformerLM(config)
    model.load_state_dict(torch.load("mini_lm.pt", map_location="cpu"))
    model.eval()
    return model, tokenizer

@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=50,
    temperature=1.0,
    top_k=0,
    top_p=0.0,
    seq_len=128
):
    model, tokenizer = load_model(seq_len=seq_len)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    encoded = tokenizer.encode(prompt)
    input_ids = encoded.ids

    # Start with prompt
    input_ids = input_ids[-(seq_len-1):]  # keep last part if too long
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        # If too long, keep last seq_len tokens
        if input_ids_tensor.shape[1] > seq_len:
            input_ids_tensor = input_ids_tensor[:, -seq_len:]

        logits = model(input_ids_tensor)
        # Take last time step
        logits = logits[:, -1, :]  # (1, vocab_size)

        # Apply temperature
        logits = logits / max(temperature, 1e-6)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Top-k sampling
        if top_k > 0:
            values, indices = torch.topk(probs, top_k)
            probs_filtered = torch.zeros_like(probs).scatter_(1, indices, values)
            probs = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)

        # Top-p (nucleus) sampling
        if top_p > 0.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > top_p
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = 0
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_token_id = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token_id)
        else:
            # Sample directly
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids_tensor = torch.cat([input_ids_tensor, next_token], dim=1)

    output_ids = input_ids_tensor[0].cpu().tolist()
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return text

if __name__ == "__main__":
    out = generate("Once upon a time", max_new_tokens=50, temperature=0.8, top_k=20)
    print(out)
