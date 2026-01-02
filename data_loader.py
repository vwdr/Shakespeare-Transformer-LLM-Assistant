# data_loader.py
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader

class TextSequenceDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        # Each example is a window of length seq_len
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        x = self.token_ids[idx:idx + self.seq_len]
        y = self.token_ids[idx + 1:idx + 1 + self.seq_len]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def create_dataloaders(tokenizer_path="tokenizer.json", seq_len=128, batch_size=32, split_ratio=0.9):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    with open("data_raw.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Encode entire corpus into token ids
    encoded = tokenizer.encode(text)
    ids = encoded.ids  # list[int]

    # Train/val split
    split_idx = int(len(ids) * split_ratio)
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]

    train_dataset = TextSequenceDataset(train_ids, seq_len)
    val_dataset = TextSequenceDataset(val_ids, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, tokenizer
