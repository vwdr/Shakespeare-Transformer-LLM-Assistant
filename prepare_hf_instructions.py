# prepare_hf_instructions.py
import argparse
import json
import re

from datasets import load_dataset

_REPLACEMENTS = {
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2014": "-",
    "\u2013": "-",
    "\u2026": "...",
}


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"(\w)\s+'([a-z])", r"\1'\2", text)
    return text.encode("ascii", "ignore").decode("ascii")


def shorten_response(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if not text:
        return ""
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def extract_pair(row, input_field: str | None, output_field: str | None) -> tuple[str, str]:
    if input_field and output_field:
        return row.get(input_field, ""), row.get(output_field, "")
    if "input" in row and "output" in row:
        return row.get("input", ""), row.get("output", "")
    if "translated_dialog" in row and "og_response" in row:
        return row.get("translated_dialog", ""), row.get("og_response", "")
    if "instruction" in row and "response" in row:
        return row.get("instruction", ""), row.get("response", "")
    return "", ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build instructions.jsonl from a Hugging Face style-pair dataset."
    )
    parser.add_argument(
        "--dataset",
        default="ayaan04/english-to-shakespeare",
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
    parser.add_argument(
        "--input-field",
        default=None,
        help="Optional dataset field name for the modern input text",
    )
    parser.add_argument(
        "--output-field",
        default=None,
        help="Optional dataset field name for the Shakespearean output text",
    )
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)

    count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for row in ds:
            instr_raw, resp_raw = extract_pair(row, args.input_field, args.output_field)
            instr = clean_text(instr_raw)
            resp = shorten_response(resp_raw, args.max_response_chars)
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
