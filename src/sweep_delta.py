"""Langkah 6 - sweep delta_max (replikasi Gambar 3 paper).

Paper menemukan nilai 0,3-0,6 memberi trade-off terbaik:
  * delta_max terlalu kecil -> model tidak punya ruang beradaptasi, metrik total turun,
  * delta_max terlalu besar -> embedding menjauh dari inisialisasi, metrik cold turun.
Apakah rentang optimal di domain LMS sama atau berbeda adalah temuan proyek ini.
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import torch

from config import RESULTS_DIR, ensure_dirs, eval_cfg, model_cfg
from progress import bar
from train import get_splits, run_single


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deltas", type=float, nargs="+", default=list(eval_cfg.delta_sweep))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-epochs", type=int, default=model_cfg.max_epochs)
    args = ap.parse_args()

    ensure_dirs()
    model_cfg.max_epochs = args.max_epochs
    splits = get_splits(model_cfg.max_len)
    k = eval_cfg.top_k

    # Gambar 3 di paper menampilkan SASRec with content initialization sebagai
    # garis acuan horizontal. Nilainya dihitung sekali di sini supaya CSV-nya
    # langsung bisa diplot tanpa menggabungkan file lain.
    ref_bar = bar(len(args.seeds), "acuan content_init", unit="run", position=0)
    ref_runs = []
    for seed in args.seeds:
        ref_runs.append(run_single("content_init", seed, splits, model_cfg,
                                   args.device, k, verbose=False))
        ref_bar.update(1)
    ref_bar.close()
    ref = {f"ref_ci_{part}": round(
        float(np.nanmean([r[f"ndcg@{k}_{part}"] for r in ref_runs])), 4)
        for part in ("cold", "warm", "total")}
    print(f"acuan content_init: cold={ref['ref_ci_cold']:.4f} "
          f"total={ref['ref_ci_total']:.4f}")

    rows = []
    overall = bar(len(args.deltas) * len(args.seeds), "sweep delta_max",
                  unit="run", position=0)
    for delta in args.deltas:
        model_cfg.max_delta_norm = delta
        runs = []
        for seed in args.seeds:
            overall.set_description(f"sweep [delta={delta} s{seed}]")
            runs.append(run_single("delta", seed, splits, model_cfg,
                                   args.device, k, verbose=False))
            overall.update(1)

        row = {"delta_max": delta,
               "min_cosine": round(float(np.sqrt(max(0.0, 1 - delta ** 2))), 4)}
        for part in ("cold", "warm", "total"):
            vals = np.array([r[f"ndcg@{k}_{part}"] for r in runs])
            row[f"ndcg_{part}_mean"] = round(float(np.nanmean(vals)), 4)
            row[f"ndcg_{part}_std"] = round(float(np.nanstd(vals)), 4)
        row["delta_at_cap_pct"] = round(
            float(np.mean([r.get("delta_at_cap_pct", np.nan) for r in runs])), 2)
        row.update(ref)  # garis acuan content_init, sama di tiap baris
        rows.append(row)

        # overall.write() mencetak tanpa merusak tampilan progress bar.
        overall.write(f"delta_max={delta:.2f}  cos_min={row['min_cosine']:.3f}  "
                      f"NDCG@{k} cold={row['ndcg_cold_mean']:.4f}  "
                      f"total={row['ndcg_total_mean']:.4f}  "
                      f"mentok batas={row['delta_at_cap_pct']:.1f}%")

    overall.close()

    csv_path = RESULTS_DIR / "delta_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS_DIR / "delta_sweep.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")

    best = max(rows, key=lambda r: r["ndcg_total_mean"])
    print(f"\ndelta_max terbaik untuk metrik total: {best['delta_max']}")
    print(f"tersimpan -> {csv_path}")


if __name__ == "__main__":
    main()
