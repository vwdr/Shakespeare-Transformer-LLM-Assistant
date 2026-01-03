# prepare_hf_instructions.py
import argparse
import json
import re

from datasets import load_dataset


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def shorten_response(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    if parts and parts[0]:
        text = parts[0]
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build instructions.jsonl from a Hugging Face style-pair dataset."
    )
    parser.add_argument(
        "--dataset",
        default="Roudranil/shakespearean-and-modern-english-conversational-dataset",
        help="Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (e.g. train, test, or train+test)",
    )
    parser.add_argument(
        "--output",
        default="instructions.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of examples (0 = no limit)",
    )
    parser.add_argument(
        "--max-response-chars",
        type=int,
        default=240,
        help="Trim Shakespeare responses to this many characters (0 = no limit)",
    )
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)

    count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for row in ds:
            instr = clean_text(row.get("translated_dialog", ""))
            resp = shorten_response(row.get("og_response", ""), args.max_response_chars)
            if not instr or not resp:
                continue
            f.write(
                json.dumps(
                    {"instruction": instr, "response": resp},
                    ensure_ascii=True,
                )
                + "\n"
            )
            count += 1
            if args.limit and count >= args.limit:
                break

    print(f"Wrote {count} examples to {args.output}")


if __name__ == "__main__":
    main()
