# SVAROG STAGE 3 — WINDOWS RTX 2080 SUPER

## Working baseline
- Local folder: D:\NULLXES\NULLXES UNOS\Svarog
- GPU: NVIDIA RTX 2080 SUPER, 8 GB
- Python 3.11, globally installed PyTorch CUDA (no venv)
- Base weights: MagistrTheOne/NULLXES-SVAROGV2
- Existing 24,000-token SentencePiece tokenizer; DO NOT retrain.
- Track A: 50,000,000 continued-pretraining tokens (70% RU / 30% EN).
- Track B: independent supervised fine-tuning with a distinct SVAROG persona.

## Step 11–12: acquire and clean corpus

PowerShell commands (from the existing local Svarog repo):

    cd "D:\NULLXES\NULLXES UNOS\Svarog"
    git pull origin main
    python stage3_build_corpus.py --tokenizer .\hf_model\tokenizer.model --output-dir .\data_stage3_50m

Required packages: datasets, huggingface_hub, sentencepiece. If datasets is missing, install only it (python -m pip install datasets); do not reinstall PyTorch.

Source targets, using **this model's tokenizer** and counting BOS/EOS per document:

| Source | Lang | Tokens |
|---|---|---:|
| FineWeb2 rus_Cyrl | RU | 25,000,000 |
| Wikimedia Wikipedia 20231101.ru | RU | 10,000,000 |
| FineWeb sample-10BT | EN | 10,000,000 |
| Wikimedia Wikipedia 20231101.en | EN | 5,000,000 |

Streaming, normalization, basic language/script checks, spam and PII heuristics, exact+prefix dedup across sources, and stable 2% document-level held-out validation. HF source revisions are pinned on first use. A source finishes within 512 tokens of its requested quota, without fabricating/padding data.

The builder stores progress in data_stage3_50m/state.sqlite3 and commits every 100 scanned rows. Ctrl+C or connection loss: rerun exactly the same command. Streaming replay/skip may be slow but already committed rows will not be inserted twice. Do not delete the SQLite state. Raw remote datasets are streamed, not fully downloaded.

Output once complete:
- data_stage3_50m/corpus.txt (train; one document per line)
- data_stage3_50m/validation.txt (held-out validation)
- data_stage3_50m/provenance.jsonl (source and split for each document)
- data_stage3_50m/manifest.json (counts, pinned revisions and policies)

No pretraining checkpoints or released model files are modified by this script. The old tiny data/corpus.txt is NOT the training corpus for Stage 3.

## Stage 14: independent SFT personality seed

    python stage3_sft_persona.py

This writes data_stage3_sft/seed_identity.jsonl with system/user/assistant messages. The persona is direct, sarcastic and independent, but still accurate, and acknowledges fictional Asterra characters as fictional. The Марчело/trousers line is an *occasional contextual joke*, not a default boilerplate response.

**Do NOT train on this seed alone**. Combine a small curated percentage with a much larger, verified multilingual instruction corpus and evaluate general task quality. Chat delimiters remain ordinary text tokens rather than new token IDs.

## Quality and licensing

FineWeb2 is published under ODC-By 1.0 and subject to Common Crawl terms. FineWeb and Wikipedia have their own applicable dataset/content licensing and attribution requirements. Review source cards and terms before redistributing any derived data. The script implements cheap deduplication and filters, not full near-duplicate MinHash, benchmark decontamination or an audited PII scrub. Manual sample review remains mandatory before long training.

## Next Stage 13 / 15 / 16

Stage 13 will load HF safetensors weights, keep tokenizer unchanged, create new optimizer, train all 131,697,408 parameters, checkpoint and resume locally on RTX 2080 SUPER. Correct causal loss: model(input_ids=x, labels=x)["loss"], because model.forward shifts labels itself. SFT must instead mask system/user tokens and train loss on assistant tokens only. Final QA and HF release happen only after training results exist.

There is no claim of 50M tokens being downloaded until the local pipeline finishes and the manifest confirms counts.

## Stage 13: START FULL-PARAMETER TRAINING

After the 50M-token corpus is complete and its manifest is inspected:

    git pull --ff-only origin main
    python stage3_train.py --model-dir .\hf_model --data-dir .\data_stage3_50m --output .\stage3\checkpoints --steps 6000 --seq 256 --batch 1 --accum 32 --lr 2e-5 --eval-every 200 --save-every 200

This builds local compact uint32 memory-mapped token caches (about 200 MB for 50M tokens), then trains the full 131.7M model with FP16 and AdamW. Checkpoints include model, optimizer, scaler, RNG and dataset fingerprints. The first run starts from the released HF safetensors. The 6000 steps correspond to approximately 49.15M token presentations at 256 x 1 x 32, with random document sampling (not a guaranteed unique-token epoch).

After a restart, rerun the same command with `--resume`. If `latest.pt` already exists, the script refuses to start from scratch. Training logs are printed to the terminal. The model is not published automatically.

Important: this is full-parameter training and may take many hours or days on an RTX 2080 SUPER. Monitor temperature, disk space and validation loss. Do not run the old train_v2.py with the HF model bundle.

