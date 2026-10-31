"""Metrik evaluasi: HR@K dan NDCG@K, dipecah antara ground truth cold dan warm.

Hanya ada satu item relevan per sampel (next-item prediction), jadi
    HR@K   = 1 kalau rank <= K
    NDCG@K = 1 / log2(rank + 1) kalau rank <= K, selain itu 0
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def _ranks_to_metrics(ranks: np.ndarray, k: int) -> tuple:
    hit = ranks <= k
    ndcg = np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0)
    return hit.astype(np.float64), ndcg


@torch.no_grad()
def rank_targets(model, loader: DataLoader, device: str,
                 forbid_items: torch.Tensor | None = None,
                 exclude_seen: bool = True,
                 amp: bool = False) -> np.ndarray:
    """Hitung rank (1-based) dari item ground truth untuk tiap sampel.

    forbid_items : mask bool [n_items]; item ber-True tidak boleh direkomendasikan.
                   Dipakai untuk SASRec biasa, yang secara definisi tidak punya
                   representasi terlatih untuk item cold.
    exclude_seen : buang item yang sudah ada di riwayat input dari kandidat.
    """
    model.eval()
    all_ranks = []
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda")

    for seq, target in loader:
        seq = seq.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with autocast:
            hidden = model.encode(seq)[:, -1, :]      # state setelah item terakhir
        logits = model.full_logits(hidden.float())

        if forbid_items is not None:
            logits[:, forbid_items] = -torch.inf

        if exclude_seen:
            # scatter -inf ke posisi item yang sudah pernah dilihat user
            logits.scatter_(1, seq, -torch.inf)
            logits[:, 0] = -torch.inf

        target_score = logits.gather(1, target.unsqueeze(1))
        # Rank = jumlah item berskor lebih tinggi, + 1.
        ranks = (logits > target_score).sum(dim=1) + 1
        all_ranks.append(ranks.cpu().numpy())

    return np.concatenate(all_ranks)


def summarize(ranks: np.ndarray, is_cold: np.ndarray, k: int = 10) -> dict:
    """Pecah metrik menjadi cold GT / warm GT / total."""
    hit, ndcg = _ranks_to_metrics(ranks, k)
    out = {}
    for name, sel in (("cold", is_cold), ("warm", ~is_cold), ("total", np.ones_like(is_cold))):
        sel = sel.astype(bool)
        if sel.sum() == 0:
            out[f"hr@{k}_{name}"] = float("nan")
            out[f"ndcg@{k}_{name}"] = float("nan")
            out[f"n_{name}"] = 0
            continue
        out[f"hr@{k}_{name}"] = float(hit[sel].mean())
        out[f"ndcg@{k}_{name}"] = float(ndcg[sel].mean())
        out[f"n_{name}"] = int(sel.sum())
    return out


def ndcg_only(ranks: np.ndarray, k: int = 10) -> float:
    """Dipakai untuk early stopping pada validation."""
    _, ndcg = _ranks_to_metrics(ranks, k)
    return float(ndcg.mean())


def ndcg_per_sample(ranks: np.ndarray, k: int = 10) -> np.ndarray:
    """NDCG@K per sampel, tanpa dirata-ratakan. Dipakai analysis.py untuk bucketing."""
    _, ndcg = _ranks_to_metrics(ranks, k)
    return ndcg
