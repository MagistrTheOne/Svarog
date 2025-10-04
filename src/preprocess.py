#!/usr/bin/env python3
"""
Svarog Data Preprocessing Pipeline
Handles data downloading, cleaning, and tokenization preparation
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets
import pandas as pd
from tqdm import tqdm
import sentencepiece as spm
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SvarogDataPreprocessor:
    """Handles data preprocessing for Svarog training"""

    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.data_dir = Path(self.config['paths']['data_dir'])
        self.processed_dir = Path(self.config['paths']['processed_dir'])
        self.tokenizer_dir = Path(self.config['paths']['tokenizer_dir'])

        # Create directories
        for dir_path in [self.data_dir, self.processed_dir, self.tokenizer_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def download_datasets(self) -> Dict[str, Any]:
        """Download and cache datasets"""
        logger.info("Downloading datasets...")

        datasets = {}

        # OSCAR dataset (multilingual)
        try:
            logger.info("Loading OSCAR dataset...")
            oscar_ru = load_dataset("oscar", "unshuffled_deduplicated_ru", split="train[:10%]")
            oscar_en = load_dataset("oscar", "unshuffled_deduplicated_en", split="train[:5%]")
            datasets['oscar'] = concatenate_datasets([oscar_ru, oscar_en])
            logger.info(f"OSCAR loaded: {len(datasets['oscar'])} samples")
        except Exception as e:
            logger.warning(f"OSCAR download failed: {e}")

        # The Pile (subset)
        try:
            logger.info("Loading The Pile dataset...")
            pile = load_dataset("the_pile", split="train[:1%]", trust_remote_code=True)
            datasets['the_pile'] = pile
            logger.info(f"The Pile loaded: {len(datasets['the_pile'])} samples")
        except Exception as e:
            logger.warning(f"The Pile download failed: {e}")

        # CC-News
        try:
            logger.info("Loading CC-News dataset...")
            cc_news = load_dataset("cc_news", split="train[:5%]")
            datasets['cc_news'] = cc_news
            logger.info(f"CC-News loaded: {len(datasets['cc_news'])} samples")
        except Exception as e:
            logger.warning(f"CC-News download failed: {e}")

        return datasets

    def clean_text(self, text: str) -> str:
        """Clean and normalize text data"""
        import re

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '.', text)

        return text.strip()

    def preprocess_datasets(self, datasets: Dict[str, Any]) -> DatasetDict:
        """Clean and preprocess all datasets"""
        logger.info("Preprocessing datasets...")

        processed_datasets = []

        for name, dataset in datasets.items():
            logger.info(f"Processing {name}...")

            def clean_batch(batch):
                batch['text'] = [self.clean_text(text) for text in batch['text']]
                # Filter out very short texts
                valid_indices = [i for i, text in enumerate(batch['text']) if len(text) > 50]
                batch = {k: [v[i] for i in valid_indices] for k, v in batch.items()}
                return batch

            # Apply cleaning
            cleaned_dataset = dataset.map(
                clean_batch,
                batched=True,
                batch_size=1000,
                desc=f"Cleaning {name}"
            )

            processed_datasets.append(cleaned_dataset)

        # Combine all datasets
        combined_dataset = concatenate_datasets(processed_datasets)

        # Shuffle and split
        combined_dataset = combined_dataset.shuffle(seed=42)
        train_size = int(self.config['data']['train_split'] * len(combined_dataset))
        val_size = len(combined_dataset) - train_size

        split_dataset = DatasetDict({
            'train': combined_dataset.select(range(train_size)),
            'validation': combined_dataset.select(range(train_size, train_size + val_size))
        })

        logger.info(f"Final dataset: {len(split_dataset['train'])} train, {len(split_dataset['validation'])} validation")

        return split_dataset

    def train_tokenizer(self, dataset: DatasetDict) -> None:
        """Train SentencePiece tokenizer"""
        logger.info("Training SentencePiece tokenizer...")

        # Extract sample text for tokenizer training
        sample_size = min(1000000, len(dataset['train']))  # 1M samples max
        sample_texts = dataset['train'].select(range(sample_size))['text']

        # Save sample text to file
        sample_file = self.tokenizer_dir / "tokenizer_sample.txt"
        with open(sample_file, 'w', encoding='utf-8') as f:
            for text in sample_texts:
                f.write(text + '\n')

        # Configure SentencePiece training
        tokenizer_config = self.config['tokenizer']
        model_prefix = str(self.tokenizer_dir / "svarog_tokenizer")

        spm.SentencePieceTrainer.train(
            input=str(sample_file),
            model_prefix=model_prefix,
            vocab_size=tokenizer_config['vocab_size'],
            model_type=tokenizer_config['model_type'],
            character_coverage=tokenizer_config['character_coverage'],
            input_sentence_size=tokenizer_config['input_sentence_size'],
            shuffle_input_sentence=tokenizer_config['shuffle_input_sentence'],
            num_threads=8
        )

        logger.info(f"Tokenizer trained and saved to {model_prefix}.model")

    def save_processed_data(self, dataset: DatasetDict) -> None:
        """Save processed dataset to disk"""
        logger.info("Saving processed data...")

        # Save as Arrow format for efficient loading
        dataset.save_to_disk(str(self.processed_dir / "svarog_dataset"))

        # Save dataset info
        info = {
            'train_samples': len(dataset['train']),
            'val_samples': len(dataset['validation']),
            'total_samples': len(dataset['train']) + len(dataset['validation'])
        }

        with open(self.processed_dir / "dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)

        logger.info(f"Dataset saved to {self.processed_dir}")

    def run_pipeline(self) -> None:
        """Run complete preprocessing pipeline"""
        logger.info("Starting Svarog data preprocessing pipeline...")

        # Download datasets
        datasets = self.download_datasets()

        if not datasets:
            raise ValueError("No datasets were successfully downloaded")

        # Preprocess and combine
        processed_dataset = self.preprocess_datasets(datasets)

        # Train tokenizer
        self.train_tokenizer(processed_dataset)

        # Save processed data
        self.save_processed_data(processed_dataset)

        logger.info("Preprocessing pipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Svarog Data Preprocessing")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    args = parser.parse_args()

    preprocessor = SvarogDataPreprocessor(args.config)
    preprocessor.run_pipeline()


if __name__ == "__main__":
    main()
