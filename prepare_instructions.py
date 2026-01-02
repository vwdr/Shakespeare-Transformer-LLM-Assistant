# prepare_instructions.py
import json
from tokenizers import Tokenizer
import torch

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

    pairs = load_instruction_pairs()
    examples = []

    for instr, resp in pairs:
        text = f"User: {instr}\nAssistant: {resp}"
        encoded = tokenizer.encode(text)
        ids = [bos_id] + encoded.ids + [eos_id]
        # truncate if too long
        if len(ids) > seq_len:
            ids = ids[:seq_len]
        # pad if short
        pad_id = tokenizer.token_to_id("[PAD]")
        if len(ids) < seq_len:
            ids = ids + [pad_id] * (seq_len - len(ids))

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)  # all but last
        target_ids = torch.tensor(ids[1:], dtype=torch.long)  # shifted
        examples.append((input_ids, target_ids))

    return examples, tokenizer

if __name__ == "__main__":
    examples, _ = build_instruction_tensors()
    print("Number of instruction examples:", len(examples))
