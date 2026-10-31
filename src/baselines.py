"""Baseline Content-based KNN (paper 4.2.1).

Tidak ada training sama sekali: vektor user adalah rata-rata embedding konten
dari item di riwayatnya, lalu item diurutkan berdasarkan cosine similarity
terhadap vektor itu. Baseline ini bagus untuk item cold (semuanya punya konten)
tetapi lemah untuk item warm, karena sinyal kolaboratif diabaikan sepenuhnya.
"""
from __future__ import annotations

import numpy as np
import torch

from data import Splits, left_pad


@torch.no_grad()
def content_knn_ranks(splits: Splits, max_len: int, device: str = "cpu",
                      batch_size: int = 256, exclude_seen: bool = True) -> np.ndarray:
    content = torch.from_numpy(splits.content).to(device)  # baris sudah L2-normalized
    content = torch.nn.functional.normalize(content, dim=-1)

    seqs = np.stack([left_pad(s, max_len) for s in splits.test_inputs])
    targets = torch.from_numpy(splits.test_targets).to(device)
    all_ranks = []

    for start in range(0, len(seqs), batch_size):
        batch = torch.from_numpy(seqs[start:start + batch_size]).to(device)
        tgt = targets[start:start + batch_size]

        # Rata-rata hanya atas posisi non-padding.
        vecs = content[batch]                              # [B, L, D]
        valid = (batch != 0).unsqueeze(-1).float()
        user_vec = (vecs * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        user_vec = torch.nn.functional.normalize(user_vec, dim=-1)

        scores = user_vec @ content.t()                    # [B, n_items]
        scores[:, 0] = -torch.inf
        if exclude_seen:
            scores.scatter_(1, batch, -torch.inf)
            scores[:, 0] = -torch.inf

        target_score = scores.gather(1, tgt.unsqueeze(1))
        ranks = (scores > target_score).sum(dim=1) + 1
        all_ranks.append(ranks.cpu().numpy())

    return np.concatenate(all_ranks)
