"""Utilitas progress bar yang dipakai seluruh pipeline.

Semua tahap memakai format yang sama supaya progresnya langsung terbaca:
persentase, jumlah selesai/total, kecepatan, waktu berjalan, dan perkiraan sisa.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from tqdm import tqdm

BAR = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"


def bar(total, desc: str, unit: str = "it", **kw) -> tqdm:
    """Progress bar standar proyek ini."""
    return tqdm(total=total, desc=desc, unit=unit, bar_format=BAR,
                file=sys.stdout, dynamic_ncols=True, **kw)


def file_bar(path: Path, desc: str) -> tqdm:
    """Progress bar berbasis BYTE untuk file besar.

    Dipakai saat membaca JSON Lines yang jumlah barisnya tidak diketahui di muka
    (user-video.json 3,18 GB). Menghitung baris lebih dulu berarti membaca file
    dua kali, jadi progresnya diukur dari byte yang sudah dibaca.
    """
    return tqdm(total=path.stat().st_size, desc=desc, unit="B", unit_scale=True,
                unit_divisor=1024, bar_format=BAR, file=sys.stdout,
                dynamic_ncols=True)


class StageTimer:
    """Pencatat waktu per tahap, dicetak sebagai ringkasan di akhir."""

    def __init__(self) -> None:
        self.stages: list = []
        self._t0 = time.time()

    def mark(self, name: str) -> None:
        now = time.time()
        self.stages.append((name, now - self._t0))
        self._t0 = now

    def report(self) -> None:
        if not self.stages:
            return
        total = sum(d for _, d in self.stages)
        print("\nWaktu per tahap:")
        for name, dur in self.stages:
            print(f"  {name:36s} {dur / 60:7.2f} menit")
        print(f"  {'TOTAL':36s} {total / 60:7.2f} menit")


def fmt_eta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}j {m:02d}m"
    return f"{m}m {s:02d}d"
