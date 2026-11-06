"""SVAROG V2 — bounded RU/EN corpus builder from Hugging Face datasets.

Uses streaming, avoids full dataset downloads, records provenance and SHA256.
No tokens, model weights, or remote Python code are executed.
Run:
    python build_corpus_v2.py --dry-run
    python build_corpus_v2.py --ru-docs 7000 --en-docs 3000
    python build_corpus_v2.py --ru-docs 700 --en-docs 300 --max-chars 1200

Dependencies: datasets (HF), huggingface_hub; compatible installed versions.
Data source: Wikimedia Wikipedia (CC BY-SA 4.0 / GFDL; attribution/share-alike
requirements apply to downstream use; verify the dataset card before publishing).
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

SOURCE = "wikimedia/wikipedia"
REVISION = None
CONFIGS = {"ru": "20231101.ru", "en": "20231101.en"}
SPACE = re.compile(r"[ \t\u00a0]+")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean(value: str) -> str:
    value = CONTROL.sub("", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(SPACE.sub(" ", line).strip() for line in value.split("\n")).strip()


def docs_for_language(language, target, min_chars, max_chars, seed, scan_factor):
    from datasets import load_dataset
    config = CONFIGS[language]
    stream = load_dataset(SOURCE, config, split="train", streaming=True,
                          trust_remote_code=False, revision=REVISION)
    # Streaming shuffle uses bounded memory; this is not uniform random
    # sampling of the entire Wikipedia corpus.
    stream = stream.shuffle(seed=seed, buffer_size=1000)
    inspected = 0
    for item in stream:
        inspected += 1
        if inspected > max(target * scan_factor, 1000):
            break
        raw = item.get("text", "")
        if not isinstance(raw, str):
            continue
        text = clean(raw)
        if len(text) < min_chars:
            continue
        if max_chars:
            text = text[:max_chars]
        if len(text) < min_chars:
            continue
        yield {
            "language": language,
            "source": SOURCE,
            "source_config": config,
            "title": str(item.get("title", "")),
            "url": str(item.get("url", "")),
            "text": text,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ru-docs", type=int, default=7000)
    parser.add_argument("--en-docs", type=int, default=3000)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--scan-factor", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.ru_docs, args.en_docs, args.min_chars, args.scan_factor) < 1:
        parser.error("Document targets, min-chars, scan-factor must be positive")
    if args.max_chars and args.max_chars < args.min_chars:
        parser.error("max-chars must be >= min-chars (or zero to disable)")
    if not 0 < args.val_fraction < 0.5:
        parser.error("val-fraction must be between 0 and 0.5")
    if args.dry_run:
        print(json.dumps({
            "source": SOURCE, "configs": CONFIGS,
            "requested_docs": {"ru": args.ru_docs, "en": args.en_docs},
            "max_chars": args.max_chars, "val_fraction": args.val_fraction,
            "output_dir": str(args.output_dir),
            "note": "No download started. Verify dataset availability and license.",
        }, ensure_ascii=False, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "corpus.txt"
    val_path = args.output_dir / "validation.txt"
    meta_path = args.output_dir / "corpus_manifest.json"
    provenance_path = args.output_dir / "provenance.jsonl"
    rng = random.Random(args.seed)
    seen = set()
    counts = Counter()
    sources = Counter()
    with train_path.open("w", encoding="utf-8", newline="\n") as train, \
         val_path.open("w", encoding="utf-8", newline="\n") as val, \
         provenance_path.open("w", encoding="utf-8", newline="\n") as provenance:
        for language, target in (("ru", args.ru_docs), ("en", args.en_docs)):
            accepted = 0
            for item in docs_for_language(language, target, args.min_chars,
                                          args.max_chars, args.seed, args.scan_factor):
                digest = hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
                if digest in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(digest)
                # One document per line for SentencePiece training.
                flattened = " ".join(item["text"].split())
                is_val = rng.random() < args.val_fraction
                (val if is_val else train).write(flattened + "\n")
                provenance.write(json.dumps({
                    "sha256": digest, "language": language, "source": item["source"],
                    "source_config": item["source_config"],
                    "title": item["title"], "url": item["url"],
                    "split": "validation" if is_val else "train",
                    "characters": len(item["text"]),
                }, ensure_ascii=False) + "\n")
                accepted += 1
                counts[f"{language}_{'validation' if is_val else 'train'}"] += 1
                sources[language] += 1
                if accepted % 500 == 0:
                    print(f"{language}: {accepted}/{target} documents accepted", flush=True)
                if accepted >= target:
                    break
            if accepted < target:
                print(f"WARNING: {language} only {accepted}/{target} documents accepted")
    manifest = {
        "version": "svarog_v2_corpus_1",
        "dataset": SOURCE, "revision": REVISION, "configs": CONFIGS,
        "license_note": "Wikipedia CC BY-SA/GFDL: review attribution/share-alike obligations.",
        "seed": args.seed, "min_chars": args.min_chars, "max_chars": args.max_chars,
        "requested": {"ru": args.ru_docs, "en": args.en_docs},
        "actual": dict(counts), "accepted_languages": dict(sources),
        "split": {"train": str(train_path), "validation": str(val_path)},
        "deduplication": "exact SHA256 of cleaned document text",
        "streaming_shuffle_buffer": 1000,
    }
    meta_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Corpus complete:", json.dumps(manifest["actual"], ensure_ascii=False))
    print("Training:", train_path, "Validation:", val_path)
    print("Manifest:", meta_path, "Provenance:", provenance_path)


if __name__ == "__main__":
    main()
