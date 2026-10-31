"""Langkah 1 - ubah user-video.json (3,18 GB, JSON Lines) jadi tabel interaksi.

Format satu baris di file sumber:
    {"seq": [{"video_id": "V_1", "segment": [{"local_start_time": 1588431144, ...}, ...]}, ...],
     "user_id": "U_112"}

File tidak muat di RAM, jadi diproses streaming per baris. User dipilih dengan
reservoir sampling supaya sampelnya acak seragam tanpa perlu tahu jumlah baris
di muka dan tanpa dua kali baca file.

Output:
    data/processed/interactions.parquet   kolom: user_id, item_id, ts
    data/interim/item_ids.json            daftar video yang muncul di sampel
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from progress import file_bar

from config import (INTERACTIONS_PATH, ITEM_IDS_PATH, ITEM_TEXT_PATH,
                    USER_VIDEO_JSON, data_cfg, ensure_dirs)


def load_allowed_items(path: Path) -> set:
    """Baca daftar video berkonten, dari JSON list atau JSONL berfield item_id."""
    allowed = set()
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return set(json.loads(text))
    for line in text.splitlines():
        line = line.strip()
        if line:
            allowed.add(json.loads(line)["item_id"])
    return allowed


def parse_line(line: str, allowed: set | None = None):
    """Ambil (user_id, [(video_id, ts), ...]) dari satu baris JSONL.

    Satu video bisa ditonton dalam beberapa segmen; waktu tonton paling awal
    dipakai sebagai timestamp interaksi.

    `allowed` menyaring video yang tidak punya metadata konten. Penyaringan
    HARUS dilakukan di sini, sebelum sampling user - kalau disaring belakangan,
    sebagian besar sequence jadi bolong dan datanya rontok.
    """
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None

    user_id = rec.get("user_id")
    seq = rec.get("seq")
    if not user_id or not seq:
        return None

    events = []
    for entry in seq:
        vid = entry.get("video_id")
        if allowed is not None and vid not in allowed:
            continue
        segments = entry.get("segment") or []
        times = [s["local_start_time"] for s in segments
                 if isinstance(s.get("local_start_time"), (int, float))]
        if vid and times:
            events.append((vid, int(min(times))))

    if not events:
        return None

    events.sort(key=lambda x: x[1])

    # Buang video berturut-turut yang sama (mengikuti preprocessing paper 4.1.1).
    deduped = [events[0]]
    for vid, ts in events[1:]:
        if vid != deduped[-1][0]:
            deduped.append((vid, ts))

    return user_id, deduped


def reservoir_sample(path: Path, n_users: int, min_seq_len: int, seed: int,
                     allowed: set | None = None):
    """Algoritma R: pertahankan n_users sampel seragam dari stream baris."""
    rng = random.Random(seed)
    reservoir: list = []
    n_seen = 0          # jumlah user yang memenuhi syarat panjang sequence
    n_lines = 0

    pbar = file_bar(path, "scan user-video.json")
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n_lines += 1
            # Progres diukur dari byte, bukan baris: jumlah baris tidak diketahui
            # sebelum file dibaca habis.
            pbar.update(len(line.encode("utf-8", "ignore")))
            if n_lines % 2000 == 0:
                pbar.set_postfix_str(f"user layak={n_seen:,} dipilih={len(reservoir):,}")

            parsed = parse_line(line, allowed)
            if parsed is None or len(parsed[1]) < min_seq_len:
                continue

            n_seen += 1
            if len(reservoir) < n_users:
                reservoir.append(parsed)
            else:
                j = rng.randrange(n_seen)
                if j < n_users:
                    reservoir[j] = parsed
    pbar.close()

    return reservoir, n_lines, n_seen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=USER_VIDEO_JSON)
    ap.add_argument("--n-users", type=int, default=data_cfg.n_users)
    ap.add_argument("--min-seq-len", type=int, default=data_cfg.min_seq_len)
    ap.add_argument("--seed", type=int, default=data_cfg.seed)
    ap.add_argument("--restrict-to", type=Path, default=ITEM_TEXT_PATH,
                    help="daftar video berkonten (item_text.jsonl dari "
                         "build_content.py --texts-only)")
    args = ap.parse_args()

    ensure_dirs()
    if not args.source.exists():
        raise SystemExit(f"File sumber belum ada: {args.source}")

    allowed = None
    if args.restrict_to and args.restrict_to.exists():
        allowed = load_allowed_items(args.restrict_to)
        print(f"menyaring ke {len(allowed):,} video yang punya metadata konten")

    reservoir, n_lines, n_eligible = reservoir_sample(
        args.source, args.n_users, args.min_seq_len, args.seed, allowed)

    rows_user, rows_item, rows_ts = [], [], []
    for user_id, events in reservoir:
        for vid, ts in events:
            rows_user.append(user_id)
            rows_item.append(vid)
            rows_ts.append(ts)

    df = pd.DataFrame({"user_id": rows_user, "item_id": rows_item, "ts": rows_ts})
    df.to_parquet(INTERACTIONS_PATH, index=False)

    items = sorted(df["item_id"].unique().tolist())
    ITEM_IDS_PATH.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    print(f"\nbaris dibaca          : {n_lines:,}")
    print(f"user memenuhi syarat  : {n_eligible:,}")
    print(f"user disampel         : {df['user_id'].nunique():,}")
    print(f"video unik            : {len(items):,}")
    print(f"interaksi             : {len(df):,}")
    print(f"panjang sequence rata2: {len(df) / max(df['user_id'].nunique(), 1):.1f}")
    print(f"\ntersimpan -> {INTERACTIONS_PATH}")
    print(f"tersimpan -> {ITEM_IDS_PATH}")


if __name__ == "__main__":
    main()
