# Shakespeare Transformer LLM Assistant

This is my end-to-end, from-scratch language model project. It is entirely text-based: you type text, it returns text. Out of the box the demo is trained on tiny Shakespeare, so completions and replies come out in a Shakespearean flavor (“thou”, “hath”, and dramatic tone). I trained a tokenizer, built a small decoder-only Transformer in PyTorch, teached it to predict the next token, and (optionally) fine-tuned it on instruction/answer pairs so it behaves more like a chat assistant. You can interact with it through CLI. 

## Table of Contents
- [About the Project](#about-the-project)
- [Built With](#built-with)
- [How It Works](#how-it-works)
- [Key Features](#key-features)
- [Repository Map](#repository-map)
- [Getting Started](#getting-started)
- [Training and Tuning Workflow](#training-and-tuning-workflow)
- [Usage](#usage)
- [Customization and Experiments](#customization-and-experiments)
- [Safety](#safety)

## About the Project
I wanted a clear, reproducible example of how a modern generative text model is assembled. The project walks from raw text to a working model and a chat experience:
- It learns a tokenizer on my corpus.
- It trains a decoder-only Transformer language model.
- It can be fine-tuned on short instruction/response pairs to act more like a helpful assistant.
- It exposes both CLI and web chat frontends to try the model quickly.

## Built With
- PyTorch (model definition, training, device handling)
- `tokenizers` (Byte Pair Encoding tokenizer)
- Gradio (lightweight web UI for chat)
- tqdm (training progress bars)

## How It Works
- **Tokenization:** `build_tokenizer.py` trains a Byte Pair Encoding tokenizer on `data_raw.txt`, producing `tokenizer.json`. This converts raw text into token IDs.
- **Model architecture:** `model.py` implements a decoder-only Transformer with multi-head self-attention, residual connections, and sinusoidal positional encodings. It predicts the next token given previous tokens.
- **Language model training:** `train_lm.py` trains the model on plain text using AdamW, gradient clipping, and cross-entropy loss. The result is `mini_lm.pt`.
- **Instruction tuning:** `finetune_instructions.py` starts from `mini_lm.pt` and fine-tunes on instruction/response pairs from `instructions.jsonl`, producing `mini_lm_instruct.pt` for chat-style replies.
- **Sampling:** `generate.py` adds temperature, top-k, and top-p controls so you can choose between safer or more creative generations.
- **Safety filter:** `safety.py` blocks prompts containing a small set of unsafe keywords so the demo does not answer harmful requests.

## Key Features
- Text-only generation: no images or audio.
- From-scratch Transformer LM with configurable depth/width.
- BPE tokenizer you can retrain on any corpus.
- Instruction tuning for assistant-like behavior.
- Terminal and web chat frontends.
- Temperature, top-k, and top-p sampling controls.
- Simple keyword-based safety refusal.

## Repository Map
- `prepare_data.py` — download a sample (tiny Shakespeare) to `data_raw.txt` or swap in your own text.
- `build_tokenizer.py` — train and save `tokenizer.json` (BPE vocab ~2k by default).
- `train_lm.py` — train the base LM and save `mini_lm.pt`.
- `prepare_instructions.py` + `instructions.jsonl` — convert instruction/answer pairs into tensors.
- `finetune_instructions.py` — fine-tune the base LM for chat and save `mini_lm_instruct.pt`.
- `generate.py` — load the base LM and continue prompts with sampling controls.
- `chat_cli.py` / `chat_app.py` — terminal and Gradio chat frontends that use the instruction-tuned model and safety filter.
- `huggingface_hub.py` — shim to satisfy Gradio’s legacy imports without HF auth.

## Getting Started
1) Use Python 3.10+ if possible.  
2) Install dependencies: `pip install torch tokenizers tqdm gradio`.  
3) Hardware is auto-detected: on Apple Silicon, MPS is used when available; otherwise CUDA or CPU.
4) Checkpoints `mini_lm.pt` and `mini_lm_instruct.pt` are included for quick trials; regenerate them if you change data or configs.

## Training and Tuning Workflow
1) Get text: `python prepare_data.py` writes a sample to `data_raw.txt`. Replace this file with your own corpus if you want.  
2) Train tokenizer: `python build_tokenizer.py` creates `tokenizer.json`.  
3) Train base LM: `python train_lm.py` creates `mini_lm.pt`.  
4) (Optional) Add instruction/answer pairs to `instructions.jsonl` to shape behavior.  
5) Instruction-tune: `python finetune_instructions.py` creates `mini_lm_instruct.pt`.

## Usage
- **Prompt continuation (base LM):** edit the prompt and sampling knobs in `generate.py`, then run `python generate.py`. The model will continue your text.  
- **Chat in the terminal:** `python chat_cli.py`, then type messages; replies build on the conversation history.  
- **Chat in the browser:** `python chat_app.py` to launch Gradio and chat in a web UI.  
- Both chat paths use the instruction-tuned model and apply the safety filter.

## Customization and Experiments
- Replace `data_raw.txt` with your own corpus; retrain the tokenizer and base model.
- Increase `VOCAB_SIZE` in `build_tokenizer.py` or adjust model depth/width in `train_lm.py` / `finetune_instructions.py` to scale capacity (trains slower but learns more).
- Tweak `temperature`, `top_k`, and `top_p` in `generate.py` to shift between predictable and creative outputs.
- Grow `instructions.jsonl` with more pairs to steer tone and helpfulness.

## Safety
- `safety.py` holds `DISALLOWED_KEYWORDS` and `REFUSAL_MESSAGE`. Expand the list to tighten guardrails. The safety check runs before generation in both chat frontends.
