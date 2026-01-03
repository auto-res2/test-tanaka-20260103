"""src/preprocess.py
Dataset loading & tokenisation utilities.
"""
from __future__ import annotations

from functools import partial
from typing import Dict, Tuple

import torch
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

CACHE_DIR = ".cache/"

# --------------------------------------------------------------------------------------------------
# Tokeniser helpers
# --------------------------------------------------------------------------------------------------

def get_tokenizer(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=CACHE_DIR, use_fast=True)
    # Defensive: ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer

# --------------------------------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------------------------------

def _format_example(example: Dict, tokenizer, max_len: int) -> Dict:
    instruction = example.get("instruction") or example.get("question") or ""
    inp = example.get("input") or example.get("context") or ""
    out = example.get("output") or example.get("answer") or ""
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    tok = tokenizer(prompt, truncation=True, max_length=max_len, padding="max_length")
    labels = [t if t != tokenizer.pad_token_id else -100 for t in tok["input_ids"]]
    return {
        "input_ids": tok["input_ids"],  # list[int]
        "attention_mask": tok["attention_mask"],
        "labels": labels,
    }


def _torch_collate(batch):
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        collated[k] = torch.tensor([item[k] for item in batch], dtype=torch.long)
    return collated


def build_dataloaders(cfg: DictConfig, tokenizer):
    max_len = cfg.dataset.preprocessing.max_length

    if cfg.dataset.name == "alpaca-cleaned":
        raw = load_dataset("yahma/alpaca-cleaned", cache_dir=CACHE_DIR)
        split_ratio = cfg.dataset.split_ratio
        train_val = raw["train"].train_test_split(test_size=1 - split_ratio.train, seed=42)
        train_ds = train_val["train"]
        val_test = train_val["test"].train_test_split(
            test_size=split_ratio.test / (split_ratio.validation + split_ratio.test), seed=42
        )
        val_ds = val_test["train"]
    else:
        raise ValueError(f"Unsupported dataset {cfg.dataset.name}")

    format_fn = partial(_format_example, tokenizer=tokenizer, max_len=max_len)
    train_ds = train_ds.map(format_fn, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(format_fn, remove_columns=val_ds.column_names)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True, collate_fn=_torch_collate
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=_torch_collate)
    return train_loader, val_loader
