"""SASRec dengan tiga mode embedding item.

    random        SASRec standar - embedding diinisialisasi acak dan dilatih penuh.
                  Tidak bisa menangani item cold sama sekali.

    content_init  Embedding diinisialisasi dari embedding konten lalu di-fine-tune
                  penuh. Item warm bergeser jauh, item cold tertinggal di posisi
                  awalnya sehingga ruang representasinya jadi tidak konsisten.

    delta         Usulan paper. Embedding konten DIBEKUKAN, yang dilatih hanya
                  vektor koreksi kecil d_i dengan batas norm delta_max:

                      e_i = c_i + d_i,   ||c_i|| = 1,   ||d_i|| <= delta_max

                  Batas ini menjamin cos(c_i, e_i) >= sqrt(1 - delta_max^2),
                  jadi representasi boleh beradaptasi tapi tidak lepas dari
                  struktur semantik aslinya. Item cold yang tidak pernah
                  menerima gradien tetap berada di c_i - satu ruang dengan item warm.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomItemEmbedding(nn.Module):
    """nn.Embedding biasa. Skala inisialisasi disamakan dengan embedding konten
    (norm sekitar 1) supaya perbandingan antar mode adil."""

    def __init__(self, n_items: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_items, dim, padding_idx=0)
        nn.init.normal_(self.emb.weight, mean=0.0, std=1.0 / math.sqrt(dim))
        with torch.no_grad():
            self.emb.weight[0].zero_()

    def weight(self) -> torch.Tensor:
        return self.emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)


class ContentInitEmbedding(nn.Module):
    """Diinisialisasi dari konten, lalu dilatih penuh tanpa batasan apa pun."""

    def __init__(self, content: torch.Tensor):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(content.clone(), freeze=False,
                                                padding_idx=0)

    def weight(self) -> torch.Tensor:
        return self.emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)


class DeltaItemEmbedding(nn.Module):
    """Konten beku + delta terlatih dengan norm dibatasi delta_max."""

    def __init__(self, content: torch.Tensor, max_delta_norm: float):
        super().__init__()
        # Konten disimpan sebagai buffer, bukan Parameter -> tidak pernah menerima gradien.
        self.register_buffer("content", F.normalize(content.clone(), dim=-1))
        self.delta = nn.Parameter(torch.zeros_like(content))
        self.max_delta_norm = float(max_delta_norm)

        pad = torch.ones(content.shape[0], 1)
        pad[0] = 0.0
        self.register_buffer("pad_mask", pad)  # baris PAD dipaksa nol

    def weight(self) -> torch.Tensor:
        norm = self.delta.norm(dim=-1, keepdim=True)
        # Baris yang melewati budget di-rescale; yang di bawahnya lewat apa adanya.
        # Rescale (bukan clamp in-place) supaya tetap bisa diturunkan gradiennya.
        scale = (self.max_delta_norm / norm.clamp_min(1e-8)).clamp(max=1.0)
        return (self.content + self.delta * scale) * self.pad_mask

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight())

    @torch.no_grad()
    def delta_norms(self) -> torch.Tensor:
        """Untuk analisis: distribusi norm delta aktual setelah training."""
        norm = self.delta.norm(dim=-1)
        return norm.clamp(max=self.max_delta_norm)


class PointWiseFFN(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(F.relu(self.fc1(x)))))


class SASRec(nn.Module):
    def __init__(self, n_items: int, cfg, content: np.ndarray | None = None,
                 mode: str = "random"):
        super().__init__()
        self.mode = mode
        self.n_items = n_items
        self.dim = cfg.hidden_dim
        self.max_len = cfg.max_len

        if mode == "random":
            self.item_emb = RandomItemEmbedding(n_items, self.dim)
        else:
            if content is None:
                raise ValueError(f"mode '{mode}' membutuhkan matriks konten")
            content_t = torch.from_numpy(np.ascontiguousarray(content)).float()
            if content_t.shape[1] != self.dim:
                raise ValueError(
                    f"dimensi konten {content_t.shape[1]} != hidden_dim {self.dim}")
            if mode == "content_init":
                self.item_emb = ContentInitEmbedding(content_t)
            elif mode == "delta":
                self.item_emb = DeltaItemEmbedding(content_t, cfg.max_delta_norm)
            else:
                raise ValueError(f"mode tidak dikenal: {mode}")

        self.pos_emb = nn.Embedding(cfg.max_len, self.dim)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        self.attn_norms = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(cfg.num_blocks):
            self.attn_norms.append(nn.LayerNorm(self.dim, eps=1e-8))
            self.attns.append(nn.MultiheadAttention(self.dim, cfg.num_heads,
                                                    dropout=cfg.dropout,
                                                    batch_first=True))
            self.ffn_norms.append(nn.LayerNorm(self.dim, eps=1e-8))
            self.ffns.append(PointWiseFFN(self.dim, cfg.dropout))
        self.last_norm = nn.LayerNorm(self.dim, eps=1e-8)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: [B, L] indeks item (0 = PAD) -> hidden state [B, L, D]."""
        B, L = seq.shape
        x = self.item_emb(seq) * math.sqrt(self.dim)
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.emb_dropout(x + self.pos_emb(positions))

        # Posisi padding dinolkan setelah tiap blok. Pendekatan ini (bukan
        # key_padding_mask) mengikuti implementasi SASRec asli dan menghindari
        # NaN pada baris yang seluruh key-nya ter-mask.
        pad = (seq == 0).unsqueeze(-1)
        x = x.masked_fill(pad, 0.0)

        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device),
                            diagonal=1)
        for attn_norm, attn, ffn_norm, ffn in zip(
                self.attn_norms, self.attns, self.ffn_norms, self.ffns):
            q = attn_norm(x)
            out, _ = attn(q, x, x, attn_mask=causal, need_weights=False)
            x = q + out
            x = x + ffn(ffn_norm(x))
            x = x.masked_fill(pad, 0.0)

        return self.last_norm(x)

    def item_weight(self) -> torch.Tensor:
        return self.item_emb.weight()

    def full_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Skor terhadap seluruh katalog. PAD selalu dibuat mustahil terpilih."""
        logits = hidden @ self.item_weight().t()
        logits[..., 0] = -torch.inf
        return logits

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.encode(seq)

    def loss(self, seq: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Full cross-entropy atas seluruh katalog (paper 4.1.3).

        Logits hanya dihitung pada posisi non-padding. Ini memangkas pemakaian
        VRAM kira-kira setengah dan mempercepat langkah training.
        """
        hidden = self.encode(seq)
        valid = target != 0
        h = hidden[valid]
        t = target[valid]
        logits = self.full_logits(h)
        return F.cross_entropy(logits, t)
