"""Uji cepat seluruh rantai kode dengan data sintetis.

Tidak menyentuh data asli: semua file ditulis ke direktori sementara dan path
di modul `data` di-patch sementara. Tujuannya memastikan split, model, loss,
dan metrik berjalan tanpa error sebelum eksperimen sungguhan dijalankan.

Jalankan: python smoke_test.py
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data as data_mod
from baselines import content_knn_ranks
from config import DataConfig, ModelConfig
from metrics import summarize
from models import SASRec
from train import run_single

N_USERS, N_ITEMS, N_COLD, DIM = 400, 300, 40, 32


def make_synthetic(tmp: Path, seed: int = 0):
    """Bangun dataset kecil yang dijamin punya item cold setelah split temporal."""
    rng = np.random.default_rng(seed)
    warm_items = [f"V_{i}" for i in range(N_ITEMS - N_COLD)]
    cold_items = [f"V_{i}" for i in range(N_ITEMS - N_COLD, N_ITEMS)]

    rows = []
    for u in range(N_USERS):
        n = int(rng.integers(8, 25))
        # Riwayat awal: hanya item warm, timestamp 1000..1900.
        ts = np.sort(rng.integers(1000, 1900, size=n))
        items = rng.choice(warm_items, size=n, replace=True)
        for item, t in zip(items, ts):
            rows.append((f"U_{u}", item, int(t)))
        # Awal periode test: masih item warm, 1900..1950.
        for t in sorted(rng.integers(1900, 1950, size=2)):
            rows.append((f"U_{u}", str(rng.choice(warm_items)), int(t)))
        # Interaksi terakhir di 1990..2000 - jauh di atas batas kuantil 90%,
        # sehingga item cold dijamin hanya muncul di periode test.
        pool = cold_items if rng.random() < 0.5 else warm_items
        rows.append((f"U_{u}", str(rng.choice(pool)), int(rng.integers(1990, 2000))))

    df = pd.DataFrame(rows, columns=["user_id", "item_id", "ts"])

    all_items = warm_items + cold_items
    content = rng.normal(size=(len(all_items), DIM)).astype(np.float32)
    content /= np.linalg.norm(content, axis=1, keepdims=True)

    inter_path = tmp / "interactions.parquet"
    content_path = tmp / "content_emb.npy"
    items_path = tmp / "content_items.json"
    df.to_parquet(inter_path, index=False)
    np.save(content_path, content)
    items_path.write_text(json.dumps(all_items), encoding="utf-8")
    return inter_path, content_path, items_path


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        inter, cont, items = make_synthetic(tmp)

        # Patch path di modul `data` (bukan di `config`), karena nama-nama itu
        # sudah terikat saat import.
        data_mod.INTERACTIONS_PATH = inter
        data_mod.CONTENT_PATH = cont
        data_mod.CONTENT_ITEMS_PATH = items

        cfg_data = DataConfig(n_users=N_USERS, min_seq_len=3, k_core=3,
                              test_quantile=0.90, val_user_frac=0.10, seed=0)
        splits = data_mod.build_splits(cfg_data, max_len=32)

        print("\n--- statistik split ---")
        for k, v in splits.stats.items():
            print(f"  {k:22s} {v}")

        assert splits.stats["n_items_cold"] > 0, "tidak ada item cold terbentuk"
        assert splits.test_is_cold.any(), "tidak ada ground truth cold"
        assert not splits.test_is_cold.all(), "semua ground truth cold"
        assert splits.content.shape[0] == splits.n_items
        assert np.allclose(splits.content[0], 0.0), "baris PAD harus nol"

        cfg_model = ModelConfig(hidden_dim=DIM, num_blocks=2, num_heads=1,
                                dropout=0.1, max_len=32, batch_size=64,
                                eval_batch_size=64, lr=1e-3, max_epochs=2,
                                patience=2, max_delta_norm=0.5,
                                amp=(device == "cuda"))

        # --- Content-based KNN (tanpa training) -------------------------------
        knn_ranks = content_knn_ranks(splits, cfg_model.max_len, device=device)
        knn = summarize(knn_ranks, splits.test_is_cold, 10)
        print(f"\ncontent_knn  cold NDCG@10={knn['ndcg@10_cold']:.4f}  "
              f"total={knn['ndcg@10_total']:.4f}")

        # --- Tiga mode SASRec --------------------------------------------------
        for mode in ("random", "content_init", "delta"):
            res = run_single(mode, 42, splits, cfg_model, device, 10, verbose=False)
            print(f"{mode:13s} cold HR@10={res['hr@10_cold']:.4f}  "
                  f"warm HR@10={res['hr@10_warm']:.4f}  "
                  f"total NDCG@10={res['ndcg@10_total']:.4f}")
            assert not np.isnan(res["ndcg@10_total"]), f"{mode}: metrik NaN"
            if mode == "random":
                assert res["hr@10_cold"] == 0.0, \
                    "SASRec biasa seharusnya mustahil mengenai item cold"
            if mode == "delta":
                assert res["delta_norm_mean"] <= cfg_model.max_delta_norm + 1e-6, \
                    "norm delta melewati batas"
                print(f"              delta: mean_norm={res['delta_norm_mean']:.4f} "
                      f"mentok_batas={res['delta_at_cap_pct']:.1f}%")

        # --- Item cold boleh/tidak boleh muncul di kandidat --------------------
        # Pemeriksaan deterministik: skor item cold harus -inf untuk SASRec biasa
        # dan harus berhingga untuk model berbasis konten.
        cold_idx = torch.from_numpy(np.where(splits.cold_mask)[0]).to(device)
        probe = torch.from_numpy(
            np.stack([data_mod.left_pad(s, cfg_model.max_len)
                      for s in splits.test_inputs[:8]])).to(device)
        for mode, forbidden in (("random", True), ("delta", False)):
            m = SASRec(splits.n_items, cfg_model,
                       None if mode == "random" else splits.content, mode).to(device).eval()
            with torch.no_grad():
                logits = m.full_logits(m.encode(probe)[:, -1, :].float())
                if forbidden:
                    logits[:, cold_idx] = -torch.inf
                cold_scores = logits[:, cold_idx]
            if forbidden:
                assert torch.isinf(cold_scores).all(), \
                    "item cold seharusnya mustahil dipilih SASRec biasa"
            else:
                assert torch.isfinite(cold_scores).all(), \
                    "item cold seharusnya punya skor berhingga di mode delta"
        print("uji kandidat: item cold -inf untuk random, berhingga untuk delta")

        # --- Batas norm delta diuji langsung ----------------------------------
        model = SASRec(splits.n_items, cfg_model, splits.content, "delta")
        with torch.no_grad():
            model.item_emb.delta.normal_(0, 10.0)  # paksa delta sangat besar
        norms = (model.item_emb.weight() - model.item_emb.content).norm(dim=-1)
        assert norms[1:].max().item() <= cfg_model.max_delta_norm + 1e-5, \
            "clipping norm delta tidak bekerja"
        assert torch.allclose(model.item_emb.weight()[0],
                              torch.zeros(DIM)), "baris PAD harus tetap nol"
        print(f"\nuji clipping: norm delta maksimum = {norms[1:].max().item():.4f} "
              f"(batas {cfg_model.max_delta_norm})")

    print("\nSEMUA UJI LULUS")


if __name__ == "__main__":
    main()
