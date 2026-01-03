# prepare_instructions.py
import json

import torch
from tokenizers import Tokenizer

from prompts import SYSTEM_PROMPT

def load_instruction_pairs(path="instructions.jsonl"):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pairs.append((obj["instruction"], obj["response"]))
    return pairs

def build_instruction_tensors(tokenizer_path="tokenizer.json", seq_len=128):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    pairs = load_instruction_pairs()
    examples = []

    for instr, resp in pairs:
        prefix = f"{SYSTEM_PROMPT}User: {instr}\nAssistant: "
        text = f"{prefix}{resp}"
        encoded = tokenizer.encode(text)
        prefix_ids = tokenizer.encode(prefix).ids
        ids = encoded.ids
        if bos_id is not None:
            ids = [bos_id] + ids
        if eos_id is not None:
            ids = ids + [eos_id]
        # truncate if too long
        if len(ids) > seq_len:
            ids = ids[:seq_len]
        # pad if short
        if pad_id is not None and len(ids) < seq_len:
            ids = ids + [pad_id] * (seq_len - len(ids))

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)  # all but last
        target_ids = torch.tensor(ids[1:], dtype=torch.long)  # shifted
        if pad_id is not None:
            if bos_id is not None:
                prefix_len = len(prefix_ids)
            else:
                prefix_len = max(len(prefix_ids) - 1, 0)
            prefix_len = min(prefix_len, target_ids.numel())
            target_ids[:prefix_len] = pad_id
        examples.append((input_ids, target_ids))

    return examples, tokenizer

if __name__ == "__main__":
    examples, _ = build_instruction_tensors()
    print("Number of instruction examples:", len(examples))
