# SVAROG STAGE 14–15 — INSTRUCTION SFT (Windows)

**Stage 13 baseline is immutable:** `stage3/checkpoints/best.pt` and `latest.pt`. Do not move, rename or overwrite them.

## Stage 14: build a separate RU/EN SFT corpus

PowerShell from `D:\NULLXES\NULLXES UNOS\Svarog`:

```powershell
git pull --ff-only origin main
python stage3_sft_persona.py
python stage4_build_sft.py --output-dir .\data_stage3_sft --ru-target 35000 --en-target 35000
```

If `seed_identity.jsonl` already exists, **do not regenerate it**; the builder reads it directly. The persona generator intentionally refuses to overwrite existing files.

The builder streams these public dataset candidates:
- `IlyaGusev/ru_turbo_alpaca` (RU instruction/response)
- `CohereForAI/aya_dataset` (multilingual)
- `HuggingFaceH4/ultrachat_200k` (English conversations, first user/assistant pair)

**Source availability, split names, dataset licenses, and actual RU/EN quotas have not yet been validated on your machine.** If a dataset has moved or lacks sufficient quality examples, the builder will stop; do not bypass quotas by fabricating or duplicating conversations. Some sources contain machine-generated answers; sample audit is required before training. The builder currently streams without a durable mid-stream resume. Re-running after failure starts acquisition from the beginning, but never overwrites completed train/validation outputs. A complete run produces `train.jsonl`, `validation.jsonl`, `manifest.json`.

Persona examples are a tiny separate component, not a replacement for general instructions. A 3% deterministic holdout is reserved for evaluation. The builder excludes simple spam and exact duplicates but **is not a full quality, privacy or benchmark-contamination audit**.

## Stage 15: full-parameter assistant-only SFT

Only after the Stage 14 manifest exists and representative examples have been reviewed:

```powershell
python stage4_train_sft.py --model-dir .\hf_model --stage13-best .\stage3\checkpoints\best.pt --data-dir .\data_stage3_sft --output .\stage4\checkpoints --steps 1500 --seq 512 --batch 1 --accum 16 --lr 1e-5 --eval-every 100 --save-every 100
```

To resume after interruption, repeat with `--resume`. Stage 15 checkpoints are written to `stage4/checkpoints` only.

Training serializes `system/user/assistant` turns using the existing textual `[im_start]` / `[im_end]` format. It masks **all system/user and assistant header tokens** with `-100`, computing CE only for assistant answer tokens and end markers. The model internally shifts labels; **do not shift them manually**. Overlong records are discarded rather than truncating the assistant answer or silently dropping the user's instruction. The training script reports how many records survive the 512-token context limit. If too many are discarded, stop and reassess dataset quality/sequence length.

Important: this is an initial full-parameter SFT configuration, not a proven optimal schedule. Watch held-out loss for overfitting. Do not claim the model is instruction-capable until post-SFT generation QA.

## Stage 16
Compare HF base, Stage 13 best and Stage 15 best on separate held-out instruction questions (RU/EN technical, factual, code, uncertainty and persona), and compare Stage 13 and Stage 15 with the same chat prompt serialization. Do not evaluate on persona seed examples used in training.
