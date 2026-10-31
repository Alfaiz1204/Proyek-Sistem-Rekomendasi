"""Langkah 3 - pembagian data dan Dataset PyTorch.

Mengikuti protokol evaluasi paper (Bagian 4.1.2):
  * global temporal split pada kuantil 90% dari seluruh timestamp (mencegah kebocoran data),
  * k-core filtering k=3 HANYA pada data train/validation,
  * 10% user acak jadi validation,
  * interaksi terakhir = ground truth, sisanya jadi input.

Definisi cold item: video yang TIDAK pernah muncul di periode train.
Definisi ini harus lahir dari split temporal, bukan sampling acak - kalau tidak,
model sudah pernah melihat item tersebut saat training dan angka cold-start jadi palsu.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import CONTENT_ITEMS_PATH, CONTENT_PATH, INTERACTIONS_PATH, data_cfg

PAD = 0  # indeks 0 dicadangkan untuk padding


@dataclass
class Splits:
    n_items: int                      # termasuk PAD
    content: np.ndarray               # [n_items, dim], baris 0 = nol (PAD)
    train_seqs: list                  # sequence training per user
    val_inputs: list
    val_targets: np.ndarray
    test_inputs: list
    test_targets: np.ndarray
    test_is_cold: np.ndarray          # bool, True kalau target adalah cold item
    cold_mask: np.ndarray             # bool [n_items], True untuk item cold
    item_train_freq: np.ndarray       # int [n_items], jumlah kemunculan di train
    stats: dict


def _k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Buang user dan item dengan interaksi kurang dari k, berulang sampai stabil."""
    while True:
        n_before = len(df)
        df = df[df["item_id"].map(df["item_id"].value_counts()) >= k]
        if df.empty:
            return df
        df = df[df["user_id"].map(df["user_id"].value_counts()) >= k]
        if len(df) == n_before or df.empty:
            return df


def build_splits(cfg=data_cfg, max_len: int = 64) -> Splits:
    df = pd.read_parquet(INTERACTIONS_PATH)
    content_items = json.loads(CONTENT_ITEMS_PATH.read_text(encoding="utf-8"))
    content_raw = np.load(CONTENT_PATH)

    # Hanya pakai interaksi yang itemnya punya embedding konten.
    n_before = len(df)
    df = df[df["item_id"].isin(set(content_items))]
    df = df.sort_values(["user_id", "ts"], kind="mergesort").reset_index(drop=True)

    # --- global temporal split ------------------------------------------------
    boundary = df["ts"].quantile(cfg.test_quantile)
    train_df = _k_core(df[df["ts"] < boundary], cfg.k_core)
    test_df = df[df["ts"] >= boundary]
    if train_df.empty:
        raise SystemExit("Data train kosong setelah k-core. Turunkan k_core atau naikkan n_users.")

    train_items = set(train_df["item_id"].unique())
    test_items = set(test_df["item_id"].unique())
    cold_items = test_items - train_items

    # --- kosakata item --------------------------------------------------------
    # Item cold ikut masuk kosakata supaya model berbasis konten bisa
    # merekomendasikannya. SASRec biasa tidak boleh - item cold di-mask saat evaluasi.
    vocab = sorted(train_items | test_items)
    item2idx = {item: i + 1 for i, item in enumerate(vocab)}  # 0 = PAD
    n_items = len(vocab) + 1

    content_lookup = {item: i for i, item in enumerate(content_items)}
    content = np.zeros((n_items, content_raw.shape[1]), dtype=np.float32)
    for item, idx in item2idx.items():
        content[idx] = content_raw[content_lookup[item]]

    cold_mask = np.zeros(n_items, dtype=bool)
    for item in cold_items:
        cold_mask[item2idx[item]] = True

    # Frekuensi tiap item di data train - dipakai untuk analisis item langka (Gambar 5).
    item_train_freq = np.zeros(n_items, dtype=np.int64)
    for item, count in train_df["item_id"].value_counts().items():
        item_train_freq[item2idx[item]] = int(count)

    # --- sequence per user ----------------------------------------------------
    train_by_user = (train_df.groupby("user_id")["item_id"]
                     .apply(lambda s: np.array([item2idx[i] for i in s], dtype=np.int64)))

    rng = np.random.default_rng(cfg.seed)
    users = sorted(train_by_user.index)
    n_val = max(1, int(len(users) * cfg.val_user_frac))
    # Sampling lewat indeks, bukan lewat array string - lebih aman untuk dtype object.
    val_idx = rng.choice(len(users), size=n_val, replace=False)
    val_users = {users[i] for i in val_idx}

    train_seqs, val_inputs, val_targets = [], [], []
    for user in users:
        seq = train_by_user[user]
        if user in val_users and len(seq) >= 2:
            # Interaksi terakhir jadi target validation, sisanya jadi input.
            val_inputs.append(seq[:-1][-max_len:])
            val_targets.append(seq[-1])
            train_seqs.append(seq[:-1])
        else:
            train_seqs.append(seq)
    train_seqs = [s for s in train_seqs if len(s) >= 2]

    # --- data test ------------------------------------------------------------
    # Input = seluruh riwayat sebelum interaksi test terakhir (train + awal periode test).
    test_by_user = test_df.groupby("user_id")["item_id"].apply(list)
    test_inputs, test_targets, test_is_cold = [], [], []
    for user, items in test_by_user.items():
        history = train_by_user.get(user)
        if history is None or len(history) == 0:
            continue  # user baru tanpa riwayat - di luar cakupan masalah item cold start
        earlier = np.array([item2idx[i] for i in items[:-1]], dtype=np.int64)
        full_input = np.concatenate([history, earlier])[-max_len:]
        if len(full_input) == 0:
            continue
        test_inputs.append(full_input)
        test_targets.append(item2idx[items[-1]])
        test_is_cold.append(items[-1] in cold_items)

    if not test_inputs:
        raise SystemExit("Data test kosong. Naikkan n_users atau turunkan test_quantile.")

    stats = {
        "interaksi_awal": int(n_before),
        "interaksi_berkonten": int(len(df)),
        "boundary_ts": int(boundary),
        "n_users_train": int(len(train_seqs)),
        "n_users_test": int(len(test_inputs)),
        "n_items_total": int(n_items - 1),
        "n_items_train": int(len(train_items)),
        "n_items_cold": int(len(cold_items)),
        "pct_items_cold": round(100 * len(cold_items) / max(len(vocab), 1), 2),
        "pct_test_cold_gt": round(100 * float(np.mean(test_is_cold)), 2),
        "avg_seq_len": round(float(np.mean([len(s) for s in train_seqs])), 1),
    }

    return Splits(
        n_items=n_items,
        content=content,
        train_seqs=train_seqs,
        val_inputs=val_inputs,
        val_targets=np.array(val_targets, dtype=np.int64),
        test_inputs=test_inputs,
        test_targets=np.array(test_targets, dtype=np.int64),
        test_is_cold=np.array(test_is_cold, dtype=bool),
        cold_mask=cold_mask,
        item_train_freq=item_train_freq,
        stats=stats,
    )


def left_pad(seq: np.ndarray, max_len: int) -> np.ndarray:
    """SASRec memakai padding di kiri supaya item terbaru selalu di posisi terakhir."""
    seq = seq[-max_len:]
    out = np.zeros(max_len, dtype=np.int64)
    out[max_len - len(seq):] = seq
    return out


class TrainSequences(Dataset):
    """Next-item prediction: input = seq[:-1], target = seq[1:]."""

    def __init__(self, seqs: list, max_len: int):
        self.seqs = seqs
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int):
        seq = self.seqs[i][-(self.max_len + 1):]
        return (torch.from_numpy(left_pad(seq[:-1], self.max_len)),
                torch.from_numpy(left_pad(seq[1:], self.max_len)))


class EvalSequences(Dataset):
    """Satu input sequence, satu ground-truth item."""

    def __init__(self, inputs: list, targets: np.ndarray, max_len: int):
        self.inputs = inputs
        self.targets = targets
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, i: int):
        return (torch.from_numpy(left_pad(self.inputs[i], self.max_len)),
                torch.tensor(self.targets[i], dtype=torch.long))
