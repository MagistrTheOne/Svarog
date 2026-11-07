# NULLXES SVAROG

An independent RU/EN decoder-only language model trained from scratch. **Current published artifact is a base model, not yet instruction-tuned.** SVAROG is not KLAREN 4E.

**Model weights:** https://huggingface.co/MagistrTheOne/NULLXES-SVAROGV2

## Current local target (2026)

- Architecture: LLaMA-inspired decoder, 12 layers, 768 hidden units, 12 heads, RoPE, RMSNorm, SwiGLU.
- Parameters: **131,697,408** (131.7M).
- Existing tokenizer: **24,000** SentencePiece vocabulary, unchanged.
- Context capacity: 2,048 tokens; tested pretraining sequence length 256.
- Training system: **Windows, RTX 2080 SUPER 8 GB, Python 3.11, global CUDA PyTorch**.
- Released HF base weights have been loaded successfully on the local RTX 2080 SUPER and a full FP16 optimizer step completed.

## Stage 3: 50M pretraining + separate personality SFT

**[Read the Windows Stage 3 runbook](STAGE3_WINDOWS.md)**.

- **Step 11–12:** stage3_build_corpus.py — stream, filter, deduplicate, tokenize and split **50M** RU/EN tokens (70% / 30%), with persistent SQLite progress for interrupted downloads.
- **Step 13:** continued pretraining from **existing HF weights**, all model parameters, new optimizer, separate checkpoints. This is the next training implementation; not claimed complete.
- **Step 14:** stage3_sft_persona.py — curated persona examples; no generic GPT impersonation. Full multilingual SFT dataset still required.
- **Step 15:** assistant-only SFT, with correct causal alignment and held-out evaluation.
- **Step 16:** regression and generation QA, new release.

Start on the existing Windows Python:

    git pull --ff-only origin main
    python stage3_build_corpus.py --tokenizer .\hf_model\tokenizer.model --output-dir .\data_stage3_50m

The builder **does not overwrite** the existing dataset or HF bundle. The generated 50M corpus is local and ignored by Git. Full training does not start merely by running the builder.

For the persona seed, run:

    python stage3_sft_persona.py

The existing chat formatting in chat_template_v3.py uses textual markers. No special-token ID or embedding resize is introduced.

## Important model behavior

The published base can generate incoherent or irrelevant completions. This is a model-quality limitation, not a CUDA error. Continued pretraining aims to improve language modeling; instruction-following is addressed by later SFT.

The model forward pass **already shifts labels** for causal LM. Correct call:

    output = model(input_ids=x, labels=x)
    loss = output["loss"]

Do not double-shift labels.

## Legacy code

The repository contains previous experiments (V2, V2.1, V3 Colab/T4, old identity SFT). They remain for history and reproducibility; **do not assume old checkpoints/configs are interchangeable with the current HF bundle**.

## Data and licensing

Before distributing the newly collected corpus, review each source's terms/attribution. FineWeb2 is ODC-By 1.0 subject to Common Crawl terms; FineWeb and Wikimedia have their own terms. Initial filters and hashed deduplication are not a complete privacy or quality audit.

---

**NULLXES — SVAROG** · Built from scratch. Personality without pretending to be human.
