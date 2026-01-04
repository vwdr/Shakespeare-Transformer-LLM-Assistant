import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from model import MiniTransformerLM, ModelConfig
from prepare_instructions import build_instruction_tensors

def finetune_instructions(
    seq_len=128,
    batch_size=8,
    lr=5e-5,
    num_epochs=3
):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    examples, tokenizer = build_instruction_tensors(seq_len=seq_len)
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=seq_len
    )
    model = MiniTransformerLM(config)
    model.load_state_dict(torch.load("mini_lm.pt", map_location="cpu"))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))

    # Simple batching
    def batch_iter():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]
            inputs = torch.stack([ex[0] for ex in batch])
            targets = torch.stack([ex[1] for ex in batch])
            yield inputs.to(device), targets.to(device)

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(list(batch_iter()), desc=f"Finetune epoch {epoch+1}/{num_epochs}")
        for x, y in pbar:
            optimizer.zero_grad()
            logits = model(x)
            B, T, V = logits.shape
            loss = criterion(logits.view(B*T, V), y.view(B*T))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "mini_lm_instruct.pt")
    print("Saved instruction-tuned model to mini_lm_instruct.pt")

if __name__ == "__main__":
    finetune_instructions()
