"""Svarog V2 SentencePiece tokenizer trainer and audit.

Usage:
  python tokenizer_v2.py train --input data/corpus.txt --vocab-size 16000
  python tokenizer_v2.py audit --model tokenizer/svarog_v2.model
Input is UTF-8 plain text, one document or paragraph per line.
This command does not download datasets or install dependencies.
"""
import argparse
import json
from pathlib import Path

import sentencepiece as spm

SPECIAL = ["<pad>", "<unk>", "<s>", "</s>"]


def audit(model_path: Path):
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    print("Tokenizer:", model_path, "vocab:", processor.vocab_size())
    for text in (
        "Привет, Сварог! Как дела?",
        "NULLXES разрабатывает физический искусственный интеллект.",
        "The quick brown fox jumps over the lazy dog.",
        "Код: def hello(): return 42",
        "Ёжик съел 5 яблок. 2026 год. 🚀",
    ):
        ids = processor.encode(text, out_type=int)
        recovered = processor.decode(ids)
        print(json.dumps({"text": text, "ids": ids[:32], "tokens": len(ids),
                          "roundtrip": recovered == text, "decoded": recovered}, ensure_ascii=False))
    print("Special IDs:", {x: processor.piece_to_id(x) for x in SPECIAL})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    train = sub.add_parser("train")
    train.add_argument("--input", required=True, type=Path)
    train.add_argument("--vocab-size", type=int, default=16000)
    train.add_argument("--output-prefix", default="tokenizer/svarog_v2")
    sub.add_parser("audit").add_argument("--model", type=Path, default=Path("tokenizer/svarog_v2.model"))
    args = parser.parse_args()
    if args.action == "audit":
        audit(args.model)
        return
    if not args.input.is_file():
        parser.error(f"Missing UTF-8 corpus: {args.input}")
    if args.vocab_size < 512:
        parser.error("vocab-size must be at least 512")
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(args.input), model_prefix=str(prefix),
        vocab_size=args.vocab_size, model_type="bpe",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        byte_fallback=True, hard_vocab_limit=False,
        character_coverage=0.9995,
        normalization_rule_name="identity",
        split_digits=True,
        max_sentence_length=16384,
        num_threads=8,
    )
    audit(prefix.with_suffix(".model"))


if __name__ == "__main__":
    main()
