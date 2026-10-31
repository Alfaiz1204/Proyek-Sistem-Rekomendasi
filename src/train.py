"""Langkah 4 - training dan evaluasi satu kombinasi (mode, seed).

Protokol mengikuti paper Bagian 4.1.3:
    full cross-entropy, batch 128, Adam lr 1e-3,
    2 blok transformer, 1 attention head, dropout 0.3, dim 128.

Early stopping memakai NDCG@10 pada validation.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines import content_knn_ranks
from config import (PROCESSED_DIR, RESULTS_DIR, data_cfg, ensure_dirs,
                    eval_cfg, model_cfg)
from data import EvalSequences, TrainSequences, build_splits
from metrics import ndcg_only, rank_targets, summarize
from progress import bar
from models import SASRec

SPLITS_CACHE = PROCESSED_DIR / "splits.pkl"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_splits(max_len: int, rebuild: bool = False):
    """Split di-cache karena dipakai ulang oleh semua mode dan seed."""
    if SPLITS_CACHE.exists() and not rebuild:
        with SPLITS_CACHE.open("rb") as fh:
            return pickle.load(fh)
    splits = build_splits(data_cfg, max_len=max_len)
    with SPLITS_CACHE.open("wb") as fh:
        pickle.dump(splits, fh)
    return splits


def fit(mode: str, seed: int, splits, cfg=model_cfg, device: str = "cuda",
        top_k: int = 10, verbose: bool = True):
    """Latih satu model sampai early stopping, kembalikan model terbaik.

    Dipisahkan dari pelaporan metrik supaya bisa dipakai ulang oleh run_single()
    (yang butuh ringkasan) dan analysis.py (yang butuh rank per sampel).
    """
    set_seed(seed)

    model = SASRec(splits.n_items, cfg,
                   content=None if mode == "random" else splits.content,
                   mode=mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_loader = DataLoader(TrainSequences(splits.train_seqs, cfg.max_len),
                              batch_size=cfg.batch_size, shuffle=True,
                              drop_last=False, num_workers=0)
    val_loader = DataLoader(EvalSequences(splits.val_inputs, splits.val_targets, cfg.max_len),
                            batch_size=cfg.eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(EvalSequences(splits.test_inputs, splits.test_targets, cfg.max_len),
                             batch_size=cfg.eval_batch_size, shuffle=False, num_workers=0)

    # SASRec biasa tidak punya representasi terlatih untuk item cold, jadi item
    # cold dilarang muncul di rekomendasinya. Ini yang menghasilkan baris 0.0000
    # pada kolom cold di Tabel 2 paper.
    forbid = torch.from_numpy(splits.cold_mask).to(device) if mode == "random" else None

    use_amp = cfg.amp and device == "cuda"
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp)

    best_ndcg, best_state, best_epoch, bad_epochs = -1.0, None, -1, 0
    t0 = time.time()

    ep_bar = bar(cfg.max_epochs, f"  {mode} seed={seed}", unit="epoch",
                 leave=False, position=1)
    for epoch in range(cfg.max_epochs):
        model.train()
        total_loss, n_batch = 0.0, 0
        for seq, target in train_loader:
            seq = seq.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast:
                loss = model.loss(seq, target)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batch += 1

        val_ranks = rank_targets(model, val_loader, device, forbid_items=forbid, amp=use_amp)
        val_ndcg = ndcg_only(val_ranks, top_k)

        if val_ndcg > best_ndcg:
            best_ndcg, best_epoch, bad_epochs = val_ndcg, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1

        ep_bar.update(1)
        ep_bar.set_postfix_str(
            f"loss={total_loss / max(n_batch, 1):.4f} "
            f"val={val_ndcg:.4f} terbaik={best_ndcg:.4f}@{best_epoch} "
            f"sabar={bad_epochs}/{cfg.patience}")

        if bad_epochs >= cfg.patience:
            break
    ep_bar.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    meta = {"model": mode, "seed": seed, "best_epoch": best_epoch,
            "val_ndcg": best_ndcg, "train_minutes": round((time.time() - t0) / 60, 2)}

    # Statistik delta: berapa banyak item yang benar-benar mentok di batas norm.
    if mode == "delta":
        norms = model.item_emb.delta_norms().cpu().numpy()[1:]  # lewati PAD
        meta["delta_norm_mean"] = float(norms.mean())
        meta["delta_at_cap_pct"] = float(
            100.0 * np.mean(norms >= cfg.max_delta_norm - 1e-6))

    return model, forbid, test_loader, use_amp, meta


def run_single(mode: str, seed: int, splits, cfg=model_cfg, device: str = "cuda",
               top_k: int = 10, verbose: bool = True) -> dict:
    """Latih lalu laporkan metrik ringkas (cold / warm / total)."""
    model, forbid, test_loader, use_amp, meta = fit(
        mode, seed, splits, cfg, device, top_k, verbose)
    ranks = rank_targets(model, test_loader, device, forbid_items=forbid, amp=use_amp)
    result = summarize(ranks, splits.test_is_cold, top_k)
    result.update(meta)
    return result


def run_single_ranks(mode: str, seed: int, splits, cfg=model_cfg, device: str = "cuda",
                     top_k: int = 10, verbose: bool = True) -> np.ndarray:
    """Latih lalu kembalikan rank ground truth per sampel (untuk analisis bucket)."""
    model, forbid, test_loader, use_amp, _ = fit(
        mode, seed, splits, cfg, device, top_k, verbose)
    return rank_targets(model, test_loader, device, forbid_items=forbid, amp=use_amp)


def run_content_knn(splits, cfg=model_cfg, device: str = "cuda", top_k: int = 10) -> dict:
    ranks = content_knn_ranks(splits, cfg.max_len, device=device)
    result = summarize(ranks, splits.test_is_cold, top_k)
    result.update(model="content_knn", seed=0, best_epoch=0, val_ndcg=float("nan"),
                  train_minutes=0.0)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="delta",
                    choices=["random", "content_init", "delta", "content_knn"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-delta-norm", type=float, default=model_cfg.max_delta_norm)
    ap.add_argument("--max-epochs", type=int, default=model_cfg.max_epochs)
    ap.add_argument("--batch-size", type=int, default=model_cfg.batch_size)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rebuild-splits", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ensure_dirs()
    model_cfg.max_delta_norm = args.max_delta_norm
    model_cfg.max_epochs = args.max_epochs
    model_cfg.batch_size = args.batch_size

    splits = get_splits(model_cfg.max_len, rebuild=args.rebuild_splits)
    print("Statistik data:")
    for key, value in splits.stats.items():
        print(f"  {key:22s} {value}")
    print()

    if args.mode == "content_knn":
        result = run_content_knn(splits, model_cfg, args.device, eval_cfg.top_k)
    else:
        result = run_single(args.mode, args.seed, splits, model_cfg,
                            args.device, eval_cfg.top_k)

    print("\nHasil test:")
    for key, value in result.items():
        print(f"  {key:22s} {value}")

    out = args.out or RESULTS_DIR / f"{args.mode}_seed{args.seed}.json"
    out.write_text(json.dumps({"stats": splits.stats, "result": result},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ntersimpan -> {out}")


if __name__ == "__main__":
    main()
