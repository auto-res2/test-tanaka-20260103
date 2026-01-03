"""src/train.py
Single-run training executor with Hydra configuration management, Optuna hyper-parameter
search (optional), comprehensive WandB logging, and critical assertions.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Local imports (absolute because executed as module)
from src.preprocess import build_dataloaders, get_tokenizer
from src.model import BaseScheduler, CassOCScheduler, load_model

# ------------------------------------------------------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:  # deterministic ish
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assert_gradients(model: torch.nn.Module) -> None:
    """CRITICAL ‑ ensure gradients exist and are non-zero before optimiser.step()."""
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"Gradient for parameter {n} is None."
        if torch.allclose(p.grad, torch.zeros_like(p.grad)):
            raise ValueError(f"Gradient for parameter {n} is all-zero.")


@torch.no_grad()
def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tok = 0
    correct_tok = 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item() * batch["input_ids"].size(0)
        logits = outputs.logits.argmax(dim=-1)
        mask = batch["labels"] != -100
        correct_tok += (logits[mask] == batch["labels"][mask]).sum().item()
        total_tok += mask.sum().item()
    return total_loss / len(val_loader.dataset), correct_tok / max(1, total_tok)


# ------------------------------------------------------------------------------------------------------------------
# Hydra main
# ------------------------------------------------------------------------------------------------------------------


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:  # noqa: C901  – complexity acceptable for single file
    # --------------------------------------------------------------------------------------------------------------
    # Resolve final configuration: load the YAML produced by main.py (cfg.cfg_path). If not present, fall back to cfg
    # as-is (allows standalone execution for debugging).
    # --------------------------------------------------------------------------------------------------------------
    if cfg.get("cfg_path"):
        final_cfg: DictConfig = OmegaConf.load(to_absolute_path(cfg.cfg_path))
        final_cfg = OmegaConf.merge(final_cfg, cfg)  # merge any CLI overrides
    else:
        final_cfg = cfg

    # Directory preparation
    results_dir = Path(to_absolute_path(final_cfg.results_dir)).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Seed everything
    set_seed(final_cfg.get("seed", 42))

    # ----------------------------------------------------------------------------------------------------------
    # WandB initialisation
    # ----------------------------------------------------------------------------------------------------------
    run_identifier = final_cfg.get("run_id") or (final_cfg.get("run") if isinstance(final_cfg.get("run"), str) else None)
    use_wandb = final_cfg.wandb.mode != "disabled"
    if use_wandb:
        wandb.init(
            entity=final_cfg.wandb.entity,
            project=final_cfg.wandb.project,
            id=run_identifier,
            resume="allow",
            mode=final_cfg.wandb.mode,
            config=OmegaConf.to_container(final_cfg, resolve=True),
        )

    # ----------------------------------------------------------------------------------------------------------
    # Data pipeline
    # ----------------------------------------------------------------------------------------------------------
    tokenizer = get_tokenizer(final_cfg)
    train_loader, val_loader = build_dataloaders(final_cfg, tokenizer)

    # ----------------------------------------------------------------------------------------------------------
    # Model & optimiser
    # ----------------------------------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(final_cfg, tokenizer).to(device)

    # Post-init assertions
    assert tokenizer.pad_token_id is not None, "Tokenizer has no pad_token_id after preparation."
    assert model.config.vocab_size == len(tokenizer), "Tokenizer / model vocab size mismatch."

    optim_cls = {"adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[final_cfg.training.optimizer.lower()]
    optimizer = optim_cls(model.parameters(), lr=final_cfg.training.learning_rate)
    total_steps = len(train_loader) * final_cfg.training.epochs
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    # ----------------------------------------------------------------------------------------------------------
    # Optional Optuna hyper-parameter search (only when enabled & n_trials>0)
    # ----------------------------------------------------------------------------------------------------------
    optuna_cfg = final_cfg.get("optuna")

    def build_scheduler_from_cfg(local_cfg: DictConfig):
        return CassOCScheduler(local_cfg) if "CASS-OC" in local_cfg.method else BaseScheduler(local_cfg)

    if optuna_cfg and optuna_cfg.get("n_trials", 0) > 0 and final_cfg.mode == "full":

        def objective(trial: optuna.Trial):  # type: ignore[override]
            sampled = {}
            for space in optuna_cfg.search_spaces:
                if space.distribution_type == "uniform":
                    sampled_val = trial.suggest_float(space.param_name, space.low, space.high)
                else:
                    sampled_val = trial.suggest_categorical(space.param_name, space.choices)
                sampled[space.param_name] = sampled_val
            patched_cfg = OmegaConf.merge(final_cfg, {"hyper_opt": sampled})
            local_sched = build_scheduler_from_cfg(patched_cfg)
            tmp_model = load_model(patched_cfg, tokenizer).to(device)
            tmp_optim = optim_cls(tmp_model.parameters(), lr=patched_cfg.training.learning_rate)
            tmp_model.train()
            n_batches = min(5, len(train_loader))
            for i, batch in enumerate(train_loader):
                if i >= n_batches:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                _ = local_sched.compute_scores()  # touch scheduler
                outputs = tmp_model(**batch)
                tmp_loss: torch.Tensor = outputs.loss
                tmp_optim.zero_grad()
                tmp_loss.backward()
                tmp_optim.step()
            val_loss, val_acc = evaluate(tmp_model, val_loader, device)
            return val_loss - val_acc  # minimise

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=int(optuna_cfg.n_trials))
        final_cfg = OmegaConf.merge(final_cfg, {"hyper_opt": study.best_trial.params})

    # Build (possibly tuned) scheduler object
    sched_obj = build_scheduler_from_cfg(final_cfg)

    # ----------------------------------------------------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------------------------------------------------
    global_step = 0
    for epoch in range(final_cfg.training.epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{final_cfg.training.epochs}")
        for step, batch in enumerate(progress):
            if step == 0:  # batch-start assertion once per epoch
                expected_bs = final_cfg.training.batch_size if final_cfg.mode == "full" else batch["input_ids"].shape[0]
                assert batch["input_ids"].shape[0] == expected_bs, "Unexpected batch size."            
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = sched_obj.compute_scores()  # scheduler side-effect
            outputs = model(**batch)
            loss: torch.Tensor = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            # Critical gradient integrity check
            assert_gradients(model)
            optimizer.step()
            scheduler_lr.step()
            global_step += 1
            if use_wandb:
                wandb.log({"train_loss": loss.item(), "lr": scheduler_lr.get_last_lr()[0]}, step=global_step)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        # -------------------------------- end-of-epoch validation --------------------------------
        val_loss, val_acc = evaluate(model, val_loader, device)
        if use_wandb:
            wandb.log({"val_loss": val_loss, "val_acc": val_acc}, step=global_step)
        sched_obj.update_after_epoch(val_acc)

    # ----------------------------------------------------------------------------------------------------------
    # Final metrics & teardown
    # ----------------------------------------------------------------------------------------------------------
    final_val_loss, final_val_acc = evaluate(model, val_loader, device)
    if use_wandb:
        wandb.summary["final_val_loss"] = final_val_loss
        wandb.summary["final_val_acc"] = final_val_acc
        print("WandB URL:", wandb.run.get_url())
        wandb.finish()

    # Optionally save model (guarded by config flag to avoid disk bloat)
    if final_cfg.get("save_model", False):
        save_path = results_dir / "model_final"
        save_path.mkdir(exist_ok=True, parents=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    main()
