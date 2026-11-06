"""Stream multilingual V3 continuation corpus without changing the V3 tokenizer.

Writes data_v3_next/corpus.txt and validation.txt with a unique split;
keeps the old corpus/validation entirely separate. Every record is one line.

Examples:
  python build_corpus_v3_next.py --fineweb-ru 30000 --fineweb-en 5000 --wikipedia-ru 10000\n\nThe legacy --fineweb-en flag transparently uses English Wikipedia; FineWeb2\ndoes not publish an eng_Latn config.
  python build_corpus_v3_next.py --dry-run
"""
import argparse
import hashlib
import json
import random
import re
import time
import traceback
import sys
from collections import Counter
from pathlib import Path

SPACE = re.compile(r"\s+")
SOURCE_PLANS = {
    "fineweb_ru": ("HuggingFaceFW/fineweb-2", "rus_Cyrl", "train", "text"),
    "fineweb_en": ("wikimedia/wikipedia", "20231101.en", "train", "text"),
    "wikipedia_ru": ("wikimedia/wikipedia", "20231101.ru", "train", "text"),
    "wikipedia_en": ("wikimedia/wikipedia", "20231101.en", "train", "text"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fineweb-ru", type=int, default=30000)
    p.add_argument("--fineweb-en", type=int, default=5000,
                   help="Legacy flag: English Wikipedia fallback, not FineWeb2")
    p.add_argument("--wikipedia-ru", type=int, default=10000)
    p.add_argument("--wikipedia-en", type=int, default=0)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--max-chars", type=int, default=6000)
    p.add_argument("--scan-factor", type=int, default=20)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--val-fraction", type=float, default=0.03)
    p.add_argument("--output-dir", type=Path, default=Path("data_v3_next"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-retries", type=int, default=3)
    a = p.parse_args()
    targets = {"fineweb_ru": a.fineweb_ru, "fineweb_en": a.fineweb_en,
               "wikipedia_ru": a.wikipedia_ru, "wikipedia_en": a.wikipedia_en}
    if any(n < 0 for n in targets.values()) or not any(targets.values()):
        p.error("Specify positive document targets")
    if a.min_chars < 50 or a.max_chars < a.min_chars or a.scan_factor < 1 or a.max_retries < 1:
        p.error("Invalid text length or scan factor")
    if not (0 < a.val_fraction < .5):
        p.error("Invalid val fraction")
    if a.dry_run:
        print(json.dumps({"targets": targets, "sources": SOURCE_PLANS,
                          "output": str(a.output_dir)}, indent=2))
        return
    from datasets import load_dataset
    a.output_dir.mkdir(parents=True, exist_ok=True)
    tr = a.output_dir / "corpus.txt"
    va = a.output_dir / "validation.txt"
    prov = a.output_dir / "provenance.jsonl"
    manifest_path = a.output_dir / "corpus_manifest.json"
    if any(path.exists() for path in (tr, va, prov, manifest_path)):
        p.error("Output files already exist. Choose a NEW --output-dir; no overwrites.")
    tr_tmp = tr.with_suffix(".txt.partial")
    va_tmp = va.with_suffix(".txt.partial")
    prov_tmp = prov.with_suffix(".jsonl.partial")
    for partial in (tr_tmp, va_tmp, prov_tmp):
        if partial.exists():
            p.error(f"Previous incomplete build found: {partial}. Choose a new output directory to avoid accidental overwrite.")
    seen = set()
    counts = Counter()
    rng = random.Random(a.seed)
    started = time.monotonic()
    with tr_tmp.open("w", encoding="utf-8") as train, va_tmp.open("w", encoding="utf-8") as valid, prov_tmp.open("w", encoding="utf-8") as provenance:
        for key, target in targets.items():
            if target == 0:
                continue
            repo, conf, split, field = SOURCE_PLANS[key]
            print(f"OPEN {key}: {repo} config={conf}", flush=True)
            accepted = 0
            scanned = 0
            # Reopen the stream after transient network errors. Duplicate hashes
            # prevent repeated documents; the scan limit remains bounded.
            for attempt in range(1, a.max_retries + 1):
                try:
                    ds = load_dataset(repo, conf, split=split, streaming=True)
                    ds = ds.shuffle(seed=a.seed, buffer_size=1000)
                    for item in ds:
                        scanned += 1
                        if scanned > target * a.scan_factor:
                            break
                        value = item.get(field)
                        if not isinstance(value, str):
                            continue
                        value = SPACE.sub(" ", value).strip()
                        if len(value) < a.min_chars:
                            continue
                        value = value[:a.max_chars].strip()
                        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
                        if digest in seen:
                            continue
                        seen.add(digest)
                        split_name = "validation" if rng.random() < a.val_fraction else "train"
                        (valid if split_name == "validation" else train).write(value + "\n")
                        provenance.write(json.dumps({"sha256": digest, "source": repo,
                            "config": conf, "split": split_name,
                            "url": item.get("url", ""), "chars": len(value)}, ensure_ascii=False) + "\n")
                        counts[f"{key}_{split_name}"] += 1
                        accepted += 1
                        if accepted % 1000 == 0:
                            print(f"{key}: {accepted}/{target} elapsed={time.monotonic()-started:.0f}s", flush=True)
                        if accepted == target:
                            break
                    if accepted >= target or scanned > target * a.scan_factor:
                        break
                    print(f"WARNING {key}: stream exhausted at {accepted}/{target}", flush=True)
                    break
                except (OSError, ConnectionError, TimeoutError) as exc:
                    print(f"RETRY {key} attempt={attempt}/{a.max_retries}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                    if attempt == a.max_retries:
                        raise
                    time.sleep(min(2 ** attempt, 15))
            print(f"DONE {key}: accepted={accepted}/{target} scanned={scanned}", flush=True)
            if accepted < target:
                raise RuntimeError(f"{key} incomplete: {accepted}/{target}; partial files retained")
            continue
    if counts["fineweb_ru_train"] + counts["fineweb_ru_validation"] == 0:
        raise RuntimeError("No Russian FineWeb2 documents accepted")
    if sum(counts.values()) < 100:
        raise RuntimeError("Insufficient documents; partial files preserved")
    manifest = {"version": "svarog_v3_next_2", "targets": targets,
                "counts": dict(counts), "source_configs": SOURCE_PLANS,
                "split_seed": a.seed, "val_fraction": a.val_fraction,
                "min_chars": a.min_chars, "max_chars": a.max_chars,
                "note": "Old tokenizer is preserved; no new special token IDs. Review dataset licenses before redistribution."}
    manifest["source_note"] = "fineweb_en is a legacy flag mapped to English Wikipedia because FineWeb2 has no eng_Latn config"
    manifest_path.with_suffix(".json.partial").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tr_tmp.replace(tr)
    va_tmp.replace(va)
    prov_tmp.replace(prov)
    manifest_path.with_suffix(".json.partial").replace(manifest_path)
    # Complete manifest is the final commit marker.
    # No partial corpus is considered complete without it.
    # Manifest content already written above.
    #
    # Legacy redundant write removed below.

    print("DONE", json.dumps(counts, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
