#!/usr/bin/env python3
"""
Svarog Training Script
Main training pipeline with DeepSpeed optimization
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
import deepspeed
from deepspeed import DeepSpeedConfig
import wandb
from tqdm import tqdm
import sentencepiece as spm

from model import create_svarog_model, SvarogConfig


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SvarogTrainer:
    """Handles Svarog model training"""

    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.data_config = self.config['data']
        self.paths = self.config['paths']

        # Create directories
        for dir_name, dir_path in self.paths.items():
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.dataloader = None

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')

    def load_tokenizer(self) -> spm.SentencePieceProcessor:
        """Load trained SentencePiece tokenizer"""
        tokenizer_path = Path(self.paths['tokenizer_dir']) / "svarog_tokenizer.model"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(str(tokenizer_path))
        logger.info(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size()}")
        return tokenizer

    def load_dataset(self) -> Dict[str, Any]:
        """Load preprocessed dataset"""
        dataset_path = Path(self.paths['processed_dir']) / "svarog_dataset"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Processed dataset not found at {dataset_path}")

        dataset = load_from_disk(str(dataset_path))
        logger.info(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} validation")
        return dataset

    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize text examples"""
        texts = examples['text']

        # Tokenize
        tokenized = self.tokenizer.encode(texts, out_type=int, add_bos=True, add_eos=True)

        # Pad/truncate to max_length
        max_length = self.data_config['max_length']
        attention_masks = []

        for tokens in tokenized:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                # Pad with pad_token_id
                tokens.extend([self.model_config['pad_token_id']] * (max_length - len(tokens)))

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1 if token != self.model_config['pad_token_id'] else 0 for token in tokens]
            attention_masks.append(attention_mask)

        return {
            'input_ids': tokenized,
            'attention_mask': attention_masks
        }

    def prepare_data(self) -> None:
        """Prepare tokenized datasets and dataloaders"""
        logger.info("Preparing data...")

        # Load tokenizer and dataset
        self.tokenizer = self.load_tokenizer()
        dataset = self.load_dataset()

        # Tokenize datasets
        tokenized_train = dataset['train'].map(
            self.tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=4,
            remove_columns=['text']
        )

        tokenized_val = dataset['validation'].map(
            self.tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=4,
            remove_columns=['text']
        )

        # Convert to PyTorch tensors
        tokenized_train.set_format(type='torch')
        tokenized_val.set_format(type='torch')

        # Create dataloaders
        train_dataloader = DataLoader(
            tokenized_train,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        val_dataloader = DataLoader(
            tokenized_val,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        self.dataloader = {
            'train': train_dataloader,
            'validation': val_dataloader
        }

        logger.info("Data preparation completed")

    def initialize_model(self) -> None:
        """Initialize model and DeepSpeed"""
        logger.info("Initializing model...")

        # Create model
        self.model = create_svarog_model()

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model initialized: {total_params:,} total parameters, {trainable_params:,} trainable")

        # Check if we can fit in memory (rough estimate)
        param_memory_gb = total_params * 4 / (1024**3)  # 4 bytes per float32 param
        logger.info(f"Estimated model memory: {param_memory_gb:.2f} GB")

        # Initialize DeepSpeed
        ds_config_path = "ds_config.json"
        if not Path(ds_config_path).exists():
            raise FileNotFoundError(f"DeepSpeed config not found: {ds_config_path}")

        with open(ds_config_path, 'r') as f:
            ds_config = json.load(f)

        # Initialize DeepSpeed engine
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config=ds_config
        )

        logger.info("Model and DeepSpeed initialized successfully")

    def save_checkpoint(self, step: int, loss: float) -> None:
        """Save model checkpoint"""
        checkpoint_dir = Path(self.paths['checkpoint_dir'])
        checkpoint_path = checkpoint_dir / f"svarog_step_{step}"

        # Save model state
        self.model.save_checkpoint(str(checkpoint_path))

        # Save training state
        training_state = {
            'step': step,
            'loss': loss,
            'best_loss': self.best_loss,
            'config': self.config
        }

        with open(checkpoint_path / "training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)

        logger.info(f"Checkpoint saved at step {step}")

    def validate(self) -> float:
        """Run validation and return average loss"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.dataloader['validation'], desc="Validating"):
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)

                # Create labels for causal LM (shift input_ids)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100  # Ignore padding tokens

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs['loss']
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)

        logger.info(".4f")
        return avg_loss

    def train(self) -> None:
        """Main training loop"""
        logger.info("Starting training...")

        # Initialize wandb
        wandb.init(
            project="svarog-training",
            config=self.config,
            name=f"svarog-{self.model_config['hidden_size']}h-{self.model_config['num_hidden_layers']}l"
        )

        num_epochs = self.training_config['num_epochs']
        save_steps = self.training_config['save_steps']
        eval_steps = self.training_config['eval_steps']
        logging_steps = self.training_config['logging_steps']

        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(tqdm(self.dataloader['train'], desc=f"Epoch {epoch + 1}")):
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)

                # Create labels for causal LM
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs['loss']

                # Backward pass (DeepSpeed handles this)
                self.model.backward(loss)
                self.model.step()

                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1

                # Logging
                if self.global_step % logging_steps == 0:
                    current_loss = epoch_loss / num_batches
                    wandb.log({
                        'train/loss': current_loss,
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/global_step': self.global_step
                    })

                # Evaluation
                if self.global_step % eval_steps == 0:
                    val_loss = self.validate()
                    wandb.log({
                        'validation/loss': val_loss,
                        'validation/perplexity': math.exp(val_loss),
                        'validation/global_step': self.global_step
                    })

                    # Save best model
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint(self.global_step, val_loss)
                        logger.info(".4f")

                # Checkpointing
                if self.global_step % save_steps == 0:
                    self.save_checkpoint(self.global_step, loss.item())

            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Final validation and save
        final_val_loss = self.validate()
        self.save_checkpoint(self.global_step, final_val_loss)

        wandb.finish()
        logger.info("Training completed!")

    def run_training_pipeline(self) -> None:
        """Run complete training pipeline"""
        try:
            self.prepare_data()
            self.initialize_model()
            self.train()
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Train Svarog model")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint path")
    args = parser.parse_args()

    trainer = SvarogTrainer(args.config)

    if args.resume:
        logger.info(f"Resuming training from {args.resume}")
        # Load checkpoint logic would go here

    trainer.run_training_pipeline()


if __name__ == "__main__":
    main()
