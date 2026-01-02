# model.py
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256       # hidden size
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024         # feedforward size
    max_seq_len: int = 128   # context window

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        B, T, C = x.shape

        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split heads
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, heads, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # attn_scores: (B, heads, T, T)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, heads, T, d_head)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(attn_output)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention + residual
        attn_out = self.attn(self.ln1(x), mask=mask)
        x = x + attn_out
        # Feedforward + residual
        ff_out = self.ff(self.ln2(x))
        x = x + ff_out
        return x

class MiniTransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.n_layers)
        ])
        self.ln_final = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).unsqueeze(0).unsqueeze(0)
        )
        # shape: (1, 1, T, T)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        B, T = input_ids.shape
        if T > self.config.max_seq_len:
            raise ValueError("Sequence length > max_seq_len")

        x = self.token_embed(input_ids)  # (B, T, d_model)
        x = self.pos_encoding(x)

        # Build causal mask: tokens can only see previous tokens
        mask = self.causal_mask[:, :, :T, :T]  # (1, 1, T, T)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_final(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits
