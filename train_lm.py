import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from model import MiniTransformerLM, ModelConfig
from data_loader import create_dataloaders

def train_language_model(
    num_epochs=1,
    lr=3e-4,
    seq_len=128,
    batch_size=64,
    d_model=256,
    n_heads=4,
    n_layers=4,
    d_ff=1024,
    max_steps=1500,
):
    # Device: MPS (Apple Silicon) -> CUDA -> CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    train_loader, val_loader, tokenizer = create_dataloaders(
        seq_len=seq_len,
        batch_size=batch_size
    )

    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=seq_len
    )
    model = MiniTransformerLM(config).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    def evaluate():
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)  # (B, T, vocab)
                # reshape for cross entropy: (B*T, vocab), (B*T)
                B, T, V = logits.shape
                loss = criterion(logits.view(B*T, V), y.view(B*T))
                total_loss += loss.item()
                count += 1
        model.train()
        return total_loss / max(count, 1)

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        step = 0
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)  # (B, T, vocab)
            B, T, V = logits.shape
            loss = criterion(logits.view(B*T, V), y.view(B*T))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            step += 1
            if max_steps and step >= max_steps:
                break

        val_loss = evaluate()
        print(f"Validation loss after epoch {epoch+1}: {val_loss:.4f}")

    torch.save(model.state_dict(), "mini_lm.pt")
    print("Saved language model to mini_lm.pt")

if __name__ == "__main__":
    train_language_model()
