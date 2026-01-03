"""src/model.py
Model loading and toy scheduler implementations (baseline & CASS-OC).
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

CACHE_DIR = ".cache/"

# --------------------------------------------------------------------------------------------------
# Model loader
# --------------------------------------------------------------------------------------------------

def load_model(cfg: DictConfig, tokenizer):
    name = cfg.model.name
    try:
        model = AutoModelForCausalLM.from_pretrained(name, cache_dir=CACHE_DIR, torch_dtype=torch.float16, device_map="auto")
    except Exception as e:
        print(f"[WARN] Could not load {name}: {e}. Falling back to facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", cache_dir=CACHE_DIR, torch_dtype=torch.float16, device_map="auto")
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    return model

# --------------------------------------------------------------------------------------------------
# Toy shape schedulers for demonstration
# --------------------------------------------------------------------------------------------------

class BaseScheduler:
    """Baseline scheduler that uses only error predictor E_P(C) (no cost awareness)."""
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.shapes = {
            "S": {"a": 0.45, "d": 5.0, "b": 0.55, "c": 0.05},
            "M": {"a": 0.30, "d": 6.0, "b": 0.60, "c": 0.08},
            "L": {"a": 0.22, "d": 7.0, "b": 0.70, "c": 0.12},
        }

    def _E(self, name: str, C: float):
        p = self.shapes[name]
        return p["a"] * (C + p["d"]) ** (-p["b"]) + p["c"]

    def compute_scores(self):
        C_rem = 100.0
        return {k: self._E(k, C_rem) for k in self.shapes}

    def update_after_epoch(self, *_):
        pass

class CassOCScheduler(BaseScheduler):
    """Compute-Aware Shape Scheduler with Online Cost Adaptation (toy version)."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        hp = cfg.get("hyper_opt", {})
        self.lambda_t = hp.get("lambda_init", 0.5)
        self.eta = hp.get("eta", 0.05)
        self.time_weight = hp.get("time_feature_weight", 1.0)
        self.w = torch.zeros(2)  # simple linear predictor parameters
        self.rng = torch.Generator().manual_seed(42)

    def _feature(self, name: str):
        idx = {k: i + 1 for i, k in enumerate(sorted(self.shapes.keys()))}[name]
        return torch.tensor([1.0, float(idx)])

    def _predict_time(self, name: str):
        return max(0.1, float((self._feature(name) * self.w).sum().item()))

    def _simulate_actual_time(self, name: str):
        base = {"S": 1.0, "M": 2.0, "L": 3.5}[name]
        noise = torch.rand(1, generator=self.rng).item() * 0.2 + 0.9  # [0.9,1.1]
        return base * noise

    def _update_time_model(self, name: str, observed: float):
        err = observed - (self._feature(name) * self.w).sum().item()
        lr = 0.1
        self.w += lr * err * self._feature(name)

    def compute_scores(self):
        C_rem = 100.0
        scores = {}
        for name in self.shapes:
            E = self._E(name, C_rem)
            t_hat = self._predict_time(name)
            scores[name] = E + self.lambda_t * t_hat * self.time_weight
        # emulate one step
        choice = min(scores, key=scores.get)
        actual_time = self._simulate_actual_time(choice)
        self._update_time_model(choice, actual_time)
        throughput = (1 - scores[choice]) / actual_time
        self.lambda_t = max(0.0, self.lambda_t + self.eta * (throughput - 0.01))
        return scores

    def update_after_epoch(self, val_acc):
        pass
