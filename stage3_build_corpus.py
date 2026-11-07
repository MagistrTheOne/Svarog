#!/usr/bin/env python3
"""SVAROG Stage 3: resumable 50M-token RU/EN pretraining corpus, Windows-compatible.

Uses the existing 24k SentencePiece tokenizer. Downloads are streamed, not
materialized as whole HF datasets. State and deduplication live in SQLite.
Output is finalized only after all four source quotas are reached.
"""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

SOURCES = (
    ("fineweb_ru", "ru", "HuggingFaceFW/fineweb-2", "rus_Cyrl", 25_000_000),
    ("wikipedia_ru", "ru", "wikimedia/wikipedia", "20231101.ru", 10_000_000),
    ("fineweb_en", "en", "HuggingFaceFW/fineweb", "sample-10BT", 10_000_000),
    ("wikipedia_en", "en", "wikimedia/wikipedia", "20231101.en", 5_000_000),
)
SPACE = re.compile(r"\s+")
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
URL = re.compile(r"https?://\S+|www\.\S+", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s()\-]*){10,}(?!\d)")
BLOCK = ("casino", "porn", "xxx", "slot machine", "viagra", "ставки на спорт",
         "казино онлайн", "знакомства для секса", "купить диплом", "займ без отказа")
SEED = 73019
VALIDATION_PCT = 2
MIN_CHARS, MAX_CHARS = 350, 6000
STOP_MARGIN = 512


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def clean(raw, lang):
    if not isinstance(raw, str):
        return None
    text = unicodedata.normalize("NFKC", raw)
    text = SPACE.sub(" ", text).strip()
    if len(text) < MIN_CHARS:
        return None
    text = text[:MAX_CHARS].strip()
    low = text.lower()
    if any(word in low for word in BLOCK):
        return None
    if EMAIL.search(text) or PHONE.search(text):
        return None
    url_count = len(URL.findall(text))
    if url_count > 1:
        return None
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 180:
        return None
    script = sum(("\u0400" <= c <= "\u052f") if lang == "ru"
                 else ("a" <= c.lower() <= "z") for c in letters)
    if script / len(letters) < (0.52 if lang == "ru" else 0.74):
        return None
    words = text.lower().split()
    if len(words) < 65 or len(set(words)) / len(words) < 0.24:
        return None
    if max((len(w) for w in words), default=0) > 150:
        return None
    return text


def get_db(path, tokenizer_digest):
    db = sqlite3.connect(str(path))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("""CREATE TABLE IF NOT EXISTS documents (
      digest TEXT PRIMARY KEY, prefix TEXT UNIQUE, source TEXT NOT NULL,
      lang TEXT NOT NULL, split TEXT NOT NULL, tokens INTEGER NOT NULL,
      text TEXT NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS progress (
      source TEXT PRIMARY KEY, scanned INTEGER NOT NULL DEFAULT 0,
      accepted INTEGER NOT NULL DEFAULT 0, tokens INTEGER NOT NULL DEFAULT 0,
      revision TEXT NOT NULL)""")
    db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    contract = json.dumps({
        "sources": SOURCES, "seed": SEED, "validation_percent": VALIDATION_PCT,
        "min_chars": MIN_CHARS, "max_chars": MAX_CHARS, "stop_margin": STOP_MARGIN,
        "tokenizer_sha256": tokenizer_digest
    }, sort_keys=True)
    old = db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
    if old and old[0] != contract:
        raise RuntimeError("Existing corpus uses a different contract/tokenizer; choose new --output-dir")
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('contract',?)", (contract,))
    db.commit()
    return db


def source_revision(db, source, repo):
    saved = db.execute("SELECT revision FROM progress WHERE source=?", (source,)).fetchone()
    if saved:
        return saved[0]
    from huggingface_hub import HfApi
    revision = HfApi().dataset_info(repo).sha
    if not revision:
        raise RuntimeError("Could not pin dataset revision: " + repo)
    db.execute("INSERT INTO progress(source,revision) VALUES (?,?)", (source, revision))
    db.commit()
    return revision


def totals(db, source):
    row = db.execute("SELECT scanned,accepted,tokens FROM progress WHERE source=?",
                     (source,)).fetchone()
    return tuple(int(v) for v in row)


def collect(db, sp, source, lang, repo, config, budget, max_scan_factor):
    from datasets import load_dataset
    revision = source_revision(db, source, repo)
    scanned, accepted, tokens = totals(db, source)
    print(f"OPEN {source} {tokens:,}/{budget:,} tokens (resume row {scanned:,})", flush=True)
    if tokens >= budget - STOP_MARGIN:
        return
    stream = load_dataset(repo, name=config, split="train", streaming=True, revision=revision)
    stream = stream.shuffle(seed=SEED, buffer_size=2048)
    if scanned:
        stream = stream.skip(scanned)
    max_scanned = max_scan_factor * max(1, budget // 200)
    pending = 0
    t0 = time.monotonic()
    for row in stream:
        scanned += 1
        if scanned > max_scanned:
            raise RuntimeError(f"Too many rejected rows from {source}: {scanned}")
        value = clean(row.get("text"), lang)
        if value:
            key = digest(value)
            prefix = digest(value[:280])
            # Exact and prefix deduplication across all four sources/splits.
            ids = sp.encode(value, out_type=int)
            count = len(ids) + 2  # BOS + EOS, same as legacy training.
            if count <= budget - tokens and count >= 80:
                split = "validation" if int(key[:8], 16) % 100 < VALIDATION_PCT else "train"
                cur = db.execute(
                    """INSERT OR IGNORE INTO documents
                       (digest,prefix,source,lang,split,tokens,text)
                       VALUES (?,?,?,?,?,?,?)""",
                    (key, prefix, source, lang, split, count, value))
                if cur.rowcount:
                    tokens += count
                    accepted += 1
                    pending += 1
        if scanned % 100 == 0 or tokens >= budget - STOP_MARGIN:
            db.execute(
                "UPDATE progress SET scanned=?,accepted=?,tokens=? WHERE source=?",
                (scanned, accepted, tokens, source))
            db.commit()
            pending = 0
        if accepted and accepted % 1000 == 0 and scanned % 100 == 0:
            elapsed = time.monotonic() - t0
            print(f"{source}: {tokens:,}/{budget:,} tokens; docs={accepted:,}; "
                  f"scanned={scanned:,}; elapsed={elapsed:.0f}s", flush=True)
        if tokens >= budget - STOP_MARGIN:
            break
    else:
        raise RuntimeError(f"Source exhausted before quota for {source}: {tokens:,}/{budget:,}")
    db.execute("UPDATE progress SET scanned=?,accepted=?,tokens=? WHERE source=?",
               (scanned, accepted, tokens, source))
    db.commit()
    print(f"DONE {source}: {tokens:,}/{budget:,} tokens; docs={accepted:,}", flush=True)


def export(db, output, tokenizer_sha):
    for source, lang, repo, config, budget in SOURCES:
        _, _, n = totals(db, source)
        if n < budget - STOP_MARGIN:
            raise RuntimeError(f"{source} incomplete: {n}/{budget}")
    counts = Counter()
    outputs = {
        "train": output / "corpus.txt",
        "validation": output / "validation.txt",
        "provenance": output / "provenance.jsonl"
    }
    for file in outputs.values():
        if file.exists():
            raise RuntimeError(f"Refusing to overwrite existing finalized file: {file}")
    tmp = {k: Path(str(v) + ".partial") for k, v in outputs.items()}
    for file in tmp.values():
        file.unlink(missing_ok=True)
    with tmp["train"].open("w", encoding="utf-8") as train, \
         tmp["validation"].open("w", encoding="utf-8") as valid, \
         tmp["provenance"].open("w", encoding="utf-8") as prov:
        for key, source, lang, split, n, value in db.execute(
            "SELECT digest,source,lang,split,tokens,text FROM documents ORDER BY rowid"):
            (train if split == "train" else valid).write(value + "\n")
            prov.write(json.dumps({"sha256": key, "source": source, "language": lang,
                                   "split": split, "tokens_with_bos_eos": n},
                                  ensure_ascii=False) + "\n")
            counts[f"{lang}_{split}_tokens"] += n
            counts[f"{source}_{split}_docs"] += 1
    if not counts["ru_validation_tokens"] or not counts["en_validation_tokens"]:
        raise RuntimeError("Validation split is empty for a language")
    manifest = {
        "schema": "nullxes_svarog_stage3_50m_v1",
        "target_tokens_total": sum(s[4] for s in SOURCES),
        "counts": dict(counts),
        "tokenizer_sha256": tokenizer_sha,
        "sources": [{"name": name, "language": lang, "repo": repo, "config": cfg,
                     "target_tokens": target,
                     "revision": db.execute("SELECT revision FROM progress WHERE source=?",
                                            (name,)).fetchone()[0]}
                    for name, lang, repo, cfg, target in SOURCES],
        "split": "sha256(document) mod 100 < 2 => validation",
        "deduplication": "sha256 canonical text + sha256 first 280 chars",
        "licensing": "Review source licenses, attribution and Common Crawl terms before redistribution.",
        "warning": "No automatic semantic or benchmark contamination audit; review samples before training.",
    }
    mt = output / "manifest.json.partial"
    mt.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for kind in outputs:
        os.replace(tmp[kind], outputs[kind])
    os.replace(mt, output / "manifest.json")
    print("READY: corpus.txt / validation.txt / provenance.jsonl / manifest.json")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data_stage3_50m"))
    parser.add_argument("--tokenizer", type=Path, default=Path("hf_model/tokenizer.model"))
    parser.add_argument("--max-scan-factor", type=int, default=40)
    args = parser.parse_args()
    if args.max_scan_factor < 2:
        parser.error("--max-scan-factor must be >= 2")
    if not args.tokenizer.is_file():
        parser.error(f"Tokenizer not found: {args.tokenizer}")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if sp.vocab_size() != 24000:
        parser.error("Wrong tokenizer: expected SVAROG 24k")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_sha = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
    db = get_db(args.output_dir / "state.sqlite3", tokenizer_sha)
    for source, lang, repo, config, budget in SOURCES:
        collect(db, sp, source, lang, repo, config, budget, args.max_scan_factor)
    export(db, args.output_dir, tokenizer_sha)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. SQLite committed progress is retained. Re-run same command.")
        sys.exit(130)
