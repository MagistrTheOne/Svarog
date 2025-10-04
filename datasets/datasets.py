"""
Svarog Dataset Preparation v1.1
Target: 50GB clean ru/en corpus for tokenizer + training
Optimized for RTX 2080 Super + CPU i9 10900F
"""
import os
import gc
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import ftfy
from langid import classify
from datasets import load_dataset

# ============================================
# CONFIG
# ============================================
OUTPUT_DIR = "svarog/data"
TOTAL_SAMPLES_TARGET = 20_000_000  # ~50GB text
RU_EN_RATIO = 0.6  # 60% Russian, 40% English
MIN_TEXT_LENGTH = 100
MAX_TEXT_LENGTH = 8192
FLUSH_INTERVAL = 100_000  # Write every N samples

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# TEXT CLEANING
# ============================================
def clean_text(text: str) -> str:
    """Aggressive cleaning for training corpus (multi-lang safe)"""
    text = ftfy.fix_text(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)         # URLs
    text = re.sub(r'<[^>]+>', '', text)                    # HTML tags
    text = re.sub(r'\s+', ' ', text)                      # Extra spaces
    text = re.sub(r'[^\w\s.,!?()\-:;\"\'«»—–A-Za-zА-Яа-яЁё]', '', text)
    return text.strip()

def detect_lang(text: str) -> str:
    try:
        lang, _ = classify(text)
        return lang
    except Exception:
        return "unknown"

def is_valid_text(text: str, target_lang: str) -> bool:
    if len(text) < MIN_TEXT_LENGTH or len(text) > MAX_TEXT_LENGTH:
        return False
    lang = detect_lang(text)
    if lang != target_lang:
        return False
    if len(set(text.split())) < 10:
        return False
    return True

# ============================================
# PARALLEL PROCESSING HELPERS
# ============================================
def process_sample(item, target_lang):
    text = clean_text(item.get('text', ''))
    if is_valid_text(text, target_lang):
        return {"text": text, "lang": target_lang}
    return None

def process_stream(dataset, target_lang, num_samples, desc):
    samples = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, result in enumerate(tqdm(ex.map(lambda x: process_sample(x, target_lang), dataset), total=num_samples, desc=desc)):
            if i >= num_samples:
                break
            if result:
                samples.append(result)
                if len(samples) % FLUSH_INTERVAL == 0:
                    flush_partial(samples, target_lang)
                    samples.clear()
    flush_partial(samples, target_lang)
    return samples

def flush_partial(samples, name):
    if not samples:
        return
    path = os.path.join(OUTPUT_DIR, f"partial_{name}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

# ============================================
# DATASET LOADING
# ============================================
def load_oscar_russian(num_samples):
    print("[1/3] Loading OSCAR Russian...")
    dataset = load_dataset("oscar-corpus/OSCAR-2301", "ru", split="train", streaming=True)
    return process_stream(dataset, "ru", num_samples, desc="OSCAR-RU (clean)")

def load_pile_english(num_samples):
    print("[2/3] Loading The Pile English...")
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    return process_stream(dataset, "en", num_samples, desc="Pile-EN (clean)")

def load_cc_news(num_samples):
    print("[3/3] Loading CC-News...")
    dataset = load_dataset("cc_news", split="train", streaming=True)
    samples = []
    for i, item in enumerate(tqdm(dataset, total=num_samples, desc="CC-News (clean)")):
        if i >= num_samples:
            break
        text = clean_text(item['text'])
        lang = detect_lang(text)
        if lang in ["ru", "en"] and is_valid_text(text, lang):
            samples.append({"text": text, "lang": lang})
            if len(samples) % FLUSH_INTERVAL == 0:
                flush_partial(samples, f"cc_{lang}")
                samples.clear()
    flush_partial(samples, "cc_final")
    return samples

# ============================================
# MAIN PIPELINE
# ============================================
if __name__ == "__main__":
    print("=== SVAROG DATASET PREPARATION v1.1 ===\n")

    ru_target = int(TOTAL_SAMPLES_TARGET * RU_EN_RATIO)
    en_target = TOTAL_SAMPLES_TARGET - ru_target

    ru_samples = load_oscar_russian(ru_target)
    en_samples = load_pile_english(int(en_target * 0.7))
    news_samples = load_cc_news(int(en_target * 0.3))

    all_samples = ru_samples + en_samples + news_samples

    gc.collect()
    random.seed(42)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.9)
    train_data, val_data = all_samples[:split_idx], all_samples[split_idx:]

    with open(f"{OUTPUT_DIR}/train.jsonl", "w", encoding="utf-8") as f:
        for item in tqdm(train_data, desc="Writing train"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(f"{OUTPUT_DIR}/val.jsonl", "w", encoding="utf-8") as f:
        for item in tqdm(val_data, desc="Writing val"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    ru_count = sum(1 for x in all_samples if x['lang'] == 'ru')
    en_count = sum(1 for x in all_samples if x['lang'] == 'en')

    print(f"""\n=== DATASET READY ===
Train: {len(train_data)} samples
Val:   {len(val_data)} samples

Russian: {ru_count} ({ru_count / len(all_samples) * 100:.1f}%)
English: {en_count} ({en_count / len(all_samples) * 100:.1f}%)

Files: {OUTPUT_DIR}/train.jsonl, {OUTPUT_DIR}/val.jsonl
""")