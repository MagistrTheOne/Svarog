#!/usr/bin/env python3
"""Stage 14: stream and curate independent RU/EN instruction SFT dataset.

Sources are configurable and must be reviewed for licensing and quality.
No synthetic duplication, no pretraining text, no checkpoint writes.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from stage3_sft_persona import SVAROG_SYSTEM

SOURCES = {
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "train_sft"),
    "aya": ("CohereForAI/aya_dataset", "train"),
    "ru_alpaca": ("IlyaGusev/ru_turbo_alpaca", "train"),
}
SPACE = re.compile(r"\s+")
SPAM = ("download free", "click here", "buy now", "casino", "скачать бесплатно",
        "без регистрации", "ставки на спорт", "seo optimization")


def norm(text):
    return SPACE.sub(" ", text).strip()


def valid(text):
    if not isinstance(text, str) or not 12 <= len(text) <= 7000:
        return False
    if any(x in text.lower() for x in SPAM):
        return False
    if len(set(text.split())) < 5:
        return False
    return True


def language(text):
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 12:
        return None
    ru = sum("\u0400" <= c <= "\u052f" for c in letters) / len(letters)
    en = sum("a" <= c.lower() <= "z" for c in letters) / len(letters)
    if ru > 0.60:
        return "ru"
    if en > 0.75:
        return "en"
    return None


def extract(source, row):
    if source == "ultrachat":
        messages = row.get("messages", [])
        if not isinstance(messages, list):
            return None
        # Keep one-turn pairs only; avoids leaking later turns and role confusion.
        turns = [m for m in messages if m.get("role") in ("user", "assistant")]
        if len(turns) < 2 or turns[0]["role"] != "user" or turns[1]["role"] != "assistant":
            return None
        return turns[0].get("content"), turns[1].get("content")
    if source == "ru_alpaca":
        question = row.get("instruction", "")
        context = row.get("input", "")
        if context:
            question += "\n" + context
        return question, row.get("output")
    if source == "aya":
        # Aya's instruction/response dataset.
        return row.get("inputs"), row.get("targets")
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("data_stage3_sft"))
    p.add_argument("--ru-target", type=int, default=35000)
    p.add_argument("--en-target", type=int, default=35000)
    p.add_argument("--persona", type=Path, default=Path("data_stage3_sft/seed_identity.jsonl"))
    p.add_argument("--seed", type=int, default=1307)
    a = p.parse_args()
    if min(a.ru_target, a.en_target) < 1:
        p.error("Targets must be positive")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    train_file = a.output_dir / "train.jsonl"
    val_file = a.output_dir / "validation.jsonl"
    manifest_file = a.output_dir / "manifest.json"
    if any(x.exists() for x in (train_file, val_file, manifest_file)):
        p.error("SFT outputs already exist. Refusing to overwrite.")
    if not a.persona.is_file():
        p.error(f"Missing persona seed: {a.persona}. Run python stage3_sft_persona.py first")
    seen = set()
    counts = {"ru": 0, "en": 0, "persona": 0, "train": 0, "validation": 0}
    provenance = {}
    train_tmp = train_file.with_suffix(".partial")
    val_tmp = val_file.with_suffix(".partial")
    try:
        with train_tmp.open("w", encoding="utf-8") as train, val_tmp.open("w", encoding="utf-8") as val:
            def accept(question, answer, origin, persona=False):
                if not isinstance(question, str) or not isinstance(answer, str):
                    return False
                if persona:
                    if not question.strip() or not answer.strip():
                        return False
                elif not valid(question) or not valid(answer):
                    return False
                question, answer = norm(question), norm(answer)
                lang = language(question)
                if not persona and (lang is None or language(answer) != lang):
                    return False
                if not persona and counts[lang] >= (a.ru_target if lang == "ru" else a.en_target):
                    return False
                key = hashlib.sha256((question.lower() + "\n" + answer.lower()).encode("utf-8")).hexdigest()
                if key in seen:
                    return False
                seen.add(key)
                if persona:
                    counts["persona"] += 1
                else:
                    counts[lang] += 1
                messages = [
                    {"role": "system", "content": SVAROG_SYSTEM},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
                record = {"messages": messages, "origin": origin, "language": lang or "mixed",
                          "persona": persona}
                # Holdout determined by stable hash, not a random global shuffle.
                split = "validation" if int(key[:8], 16) % 100 < 3 else "train"
                (val if split == "validation" else train).write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
                provenance[origin] = provenance.get(origin, 0) + 1
                return True

            # Round-robin sources, one stream per dataset; stop if exhausted.
            for source in ("ru_alpaca", "aya", "ultrachat"):
                repo, split = SOURCES[source]
                print("STREAM", repo, split, flush=True)
                stream = load_dataset(repo, split=split, streaming=True)
                stream = stream.shuffle(seed=a.seed, buffer_size=2000)
                scanned = 0
                for row in stream:
                    scanned += 1
                    pair = extract(source, row)
                    if pair:
                        accept(pair[0], pair[1], source)
                    if scanned % 10000 == 0:
                        print(source, "scanned", scanned, "accepted", counts["ru"], counts["en"], flush=True)
                    if counts["ru"] >= a.ru_target and counts["en"] >= a.en_target:
                        break
                if counts["ru"] >= a.ru_target and counts["en"] >= a.en_target:
                    break
            if counts["ru"] < a.ru_target or counts["en"] < a.en_target:
                raise RuntimeError("Insufficient source examples for quotas. Review source availability and language mix; no fake padding.")
            # Persona remains a tiny, distinct slice, not multiplied to inflate counts.
            with a.persona.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    turns = row["messages"]
                    accept(turns[-2]["content"], turns[-1]["content"], "NULLXES_persona", persona=True)
        train_tmp.replace(train_file)
        val_tmp.replace(val_file)
        manifest_file.write_text(json.dumps({
            "counts": counts, "origins": provenance, "seed": a.seed,
            "sources": SOURCES, "split": "3% stable hash validation",
            "warning": "Source license, quality, privacy, contamination and held-out benchmark audits required before release."
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("STAGE14 READY", counts)
    except BaseException:
        # Incomplete outputs are not advertised as ready; partials kept for diagnosis.
        raise


if __name__ == "__main__":
    main()
