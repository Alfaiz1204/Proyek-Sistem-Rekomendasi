"""Langkah 5 - jalankan seluruh eksperimen dan susun tabel hasil.

Mereplikasi struktur Tabel 2 paper: 4 model x 5 seed, dilaporkan mean +- std.
Content-based KNN deterministik sehingga hanya dijalankan sekali (std = 0).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import RESULTS_DIR, SEEDS, ensure_dirs, eval_cfg, model_cfg
from progress import StageTimer, bar
from train import get_splits, run_content_knn, run_single

MODELS = [
    ("content_knn", "Content-based KNN"),
    ("random", "SASRec"),
    ("content_init", "SASRec with c.i."),
    ("delta", "SASRec with t.d. (ours)"),
]


def aggregate(runs: list, metric: str) -> tuple:
    values = np.array([r[metric] for r in runs if not np.isnan(r[metric])])
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std())


def markdown_table(by_model: dict, k: int) -> str:
    lines = [
        "| Metric | Model | Cold GT | Warm GT | Total |",
        "|---|---|---|---|---|",
    ]
    for metric in (f"hr@{k}", f"ndcg@{k}"):
        label = metric.upper()
        for mode, name in MODELS:
            if mode not in by_model:
                continue
            cells = []
            for part in ("cold", "warm", "total"):
                mean, std = aggregate(by_model[mode], f"{metric}_{part}")
                cells.append(f"{mean:.4f}±{std:.3f}")
            lines.append(f"| {label} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--models", nargs="+",
                    default=[m for m, _ in MODELS])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-epochs", type=int, default=model_cfg.max_epochs)
    ap.add_argument("--rebuild-splits", action="store_true")
    ap.add_argument("--max-delta-norm", type=float, default=model_cfg.max_delta_norm,
                    help="delta_max untuk mode delta; pilih dari hasil sweep_delta.py")
    ap.add_argument("--tag", default="", help="akhiran nama file luaran")
    args = ap.parse_args()

    ensure_dirs()
    model_cfg.max_epochs = args.max_epochs
    model_cfg.max_delta_norm = args.max_delta_norm
    splits = get_splits(model_cfg.max_len, rebuild=args.rebuild_splits)

    print("Statistik data:")
    for key, value in splits.stats.items():
        print(f"  {key:22s} {value}")
    print()

    by_model: dict = {}
    all_runs: list = []
    timer = StageTimer()

    # Total run = 1 untuk content_knn (deterministik) + n_seed untuk tiap model neural.
    total_runs = sum(1 if m == "content_knn" else len(args.seeds) for m in args.models)
    overall = bar(total_runs, "eksperimen", unit="run", position=0)

    for mode in args.models:
        runs = []
        if mode == "content_knn":
            overall.set_description(f"eksperimen [{mode}]")
            runs.append(run_content_knn(splits, model_cfg, args.device, eval_cfg.top_k))
            overall.update(1)
        else:
            for seed in args.seeds:
                overall.set_description(f"eksperimen [{mode} s{seed}]")
                runs.append(run_single(mode, seed, splits, model_cfg,
                                       args.device, eval_cfg.top_k, verbose=False))
                overall.update(1)
        by_model[mode] = runs
        all_runs.extend(runs)
        timer.mark(mode)

        mean, std = aggregate(runs, f"ndcg@{eval_cfg.top_k}_cold")
        # overall.write() mencetak tanpa merusak tampilan progress bar.
        overall.write(f"--> {mode}: NDCG@{eval_cfg.top_k} cold = {mean:.4f} +- {std:.3f}")

    overall.close()
    timer.report()

    table = markdown_table(by_model, eval_cfg.top_k)
    print("\n" + table + "\n")

    (RESULTS_DIR / f"table2{args.tag}.md").write_text(
        "# Hasil utama (replikasi struktur Tabel 2)\n\n"
        + "## Statistik data\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in splits.stats.items())
        + "\n\n## Hasil\n\n" + table + "\n",
        encoding="utf-8")
    (RESULTS_DIR / f"all_runs{args.tag}.json").write_text(
        json.dumps({"stats": splits.stats, "runs": all_runs}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    print(f"tersimpan -> {RESULTS_DIR / ('table2' + args.tag + '.md')}")
    print(f"tersimpan -> {RESULTS_DIR / ('all_runs' + args.tag + '.json')}")


if __name__ == "__main__":
    main()
