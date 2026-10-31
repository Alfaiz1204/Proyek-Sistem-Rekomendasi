"""Langkah 7 - analisis tambahan: replikasi Gambar 4 dan Gambar 5 paper.

Gambar 4 (Bagian 4.2.3): NDCG@10 terhadap proporsi item warm di input sequence.
    Menunjukkan bahwa inisialisasi konten menolong bukan hanya ketika ground
    truth-nya cold, tetapi juga ketika RIWAYAT user mengandung item cold.

Gambar 5 (Bagian 4.2.4): NDCG@10 terhadap frekuensi item ground truth di data train.
    Menunjukkan perbaikan terbesar ada di item langka, lalu mengecil dan
    bertemu dengan SASRec biasa seiring item makin sering muncul.

Luaran berupa CSV supaya bisa langsung dibuat grafik di Excel/matplotlib.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from config import RESULTS_DIR, ensure_dirs, eval_cfg, model_cfg
from metrics import ndcg_per_sample
from progress import bar
from train import get_splits, run_single_ranks

BUCKET_LABELS_FREQ = ["0 (cold)", "1-2", "3-5", "6-10", "11-30", "31-100", ">100"]
FREQ_EDGES = [0, 1, 3, 6, 11, 31, 101, np.inf]

WARM_BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]


def input_warm_fraction(splits) -> np.ndarray:
    """Proporsi item warm di tiap input sequence test."""
    out = np.zeros(len(splits.test_inputs))
    for i, seq in enumerate(splits.test_inputs):
        if len(seq) == 0:
            out[i] = 1.0
            continue
        out[i] = float(np.mean(~splits.cold_mask[seq]))
    return out


def bucket_table(ranks_by_model: dict, values: np.ndarray, edges, labels,
                 k: int) -> list:
    """Rata-rata NDCG@k per bucket, satu kolom per model."""
    rows = []
    idx = np.digitize(values, edges[1:-1], right=False)
    for b, label in enumerate(labels):
        sel = idx == b
        row = {"bucket": label, "n": int(sel.sum())}
        for model, ranks in ranks_by_model.items():
            if sel.sum() == 0:
                row[model] = float("nan")
                continue
            row[model] = round(float(ndcg_per_sample(ranks[sel], k).mean()), 4)
        rows.append(row)
    return rows


def write_csv(rows: list, path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"tersimpan -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-epochs", type=int, default=model_cfg.max_epochs)
    ap.add_argument("--max-delta-norm", type=float, default=model_cfg.max_delta_norm,
                    help="delta_max untuk mode delta; pilih dari hasil sweep_delta.py")
    args = ap.parse_args()

    ensure_dirs()
    model_cfg.max_epochs = args.max_epochs
    model_cfg.max_delta_norm = args.max_delta_norm
    splits = get_splits(model_cfg.max_len)
    k = eval_cfg.top_k

    ranks_by_model = {}
    modes = ("random", "content_init", "delta")
    overall = bar(len(modes), "analisis", unit="model", position=0)
    for mode in modes:
        overall.set_description(f"analisis [{mode}]")
        ranks_by_model[mode] = run_single_ranks(mode, args.seed, splits, model_cfg,
                                                args.device, k, verbose=False)
        overall.update(1)
    overall.close()

    # --- Gambar 5 atas: terhadap frekuensi ground truth di train --------------
    gt_freq = splits.item_train_freq[splits.test_targets].astype(float)
    rows = bucket_table(ranks_by_model, gt_freq, FREQ_EDGES, BUCKET_LABELS_FREQ, k)
    print("\nNDCG@%d per frekuensi item ground truth di train:" % k)
    for r in rows:
        print("  ", r)
    write_csv(rows, RESULTS_DIR / "fig5_by_gt_frequency.csv")

    # --- Gambar 5 bawah: NDCG relatif terhadap SASRec -------------------------
    # Paper menampilkan panel kedua berisi rasio terhadap SASRec biasa. Pada
    # bucket cold, SASRec bernilai 0 sehingga rasionya tak terdefinisi; di situ
    # dilaporkan selisih absolut saja.
    rel_rows = []
    for r in rows:
        base = r.get("random")
        out = {"bucket": r["bucket"], "n": r["n"]}
        for model in ("content_init", "delta"):
            val = r.get(model)
            if val is None or np.isnan(val) or base is None or np.isnan(base):
                out[f"{model}_rel"] = float("nan")
                out[f"{model}_diff"] = float("nan")
            else:
                out[f"{model}_rel"] = (round(float(val / base), 4)
                                       if base > 0 else float("nan"))
                out[f"{model}_diff"] = round(float(val - base), 4)
        rel_rows.append(out)
    print("\nNDCG@%d relatif terhadap SASRec:" % k)
    for r in rel_rows:
        print("  ", r)
    write_csv(rel_rows, RESULTS_DIR / "fig5_relative_to_sasrec.csv")

    # --- Gambar 4: terhadap proporsi item warm di input ------------------------
    warm_frac = input_warm_fraction(splits)
    edges = [0.0] + [b[1] for b in WARM_BUCKETS]
    labels = [f"{lo:.0%}-{min(hi, 1.0):.0%}" for lo, hi in WARM_BUCKETS]
    rows = bucket_table(ranks_by_model, warm_frac, edges, labels, k)
    print("\nNDCG@%d per proporsi item warm di input sequence:" % k)
    for r in rows:
        print("  ", r)
    write_csv(rows, RESULTS_DIR / "fig4_by_input_warm_fraction.csv")


if __name__ == "__main__":
    main()
