"""Konfigurasi terpusat untuk eksperimen cold-start LMS.

Nilai hyperparameter mengikuti Pembek et al., "Let It Go? Not Quite" (RecSys '25),
Bagian 4.1.3, kecuali yang ditandai khusus untuk adaptasi domain MOOCCubeX.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "mooccubex"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

COURSE_JSON = RAW_DIR / "course.json"
USER_VIDEO_JSON = RAW_DIR / "user-video.json"

ITEM_TEXT_PATH = INTERIM_DIR / "item_text.jsonl"
ITEM_IDS_PATH = INTERIM_DIR / "item_ids.json"
CONTENT_RAW_PATH = INTERIM_DIR / "content_e5_raw.npy"
CONTENT_PATH = PROCESSED_DIR / "content_emb.npy"
CONTENT_ITEMS_PATH = PROCESSED_DIR / "content_items.json"
INTERACTIONS_PATH = PROCESSED_DIR / "interactions.parquet"


@dataclass
class ContentConfig:
    """Pembuatan embedding konten dari metadata course/video."""

    model_name: str = "intfloat/multilingual-e5-base"  # teks MOOCCubeX berbahasa Mandarin
    max_text_len: int = 256  # cukup untuk judul video + konteks course yang sudah dipotong
    encode_batch_size: int = 64
    target_dim: int = 128  # dimensi setelah PCA, = dimensi embedding SASRec


@dataclass
class DataConfig:
    """Sampling user, filtering, dan pembagian data."""

    n_users: int = 20_000   # paper menurunkan Zvuk ke 10.000 user; di sini lebih besar karena sequence lebih pendek
    min_seq_len: int = 5    # panjang minimum sequence agar user dipakai
    k_core: int = 3         # k-core, hanya pada data train/validation (paper 4.1.1)
    test_quantile: float = 0.90  # batas global temporal split (paper 4.1.2)
    val_user_frac: float = 0.10  # 10% user acak jadi validation (paper 4.1.2)
    seed: int = 42


@dataclass
class ModelConfig:
    """Arsitektur SASRec dan optimasi (paper 4.1.3)."""

    hidden_dim: int = 128
    num_blocks: int = 2
    num_heads: int = 1
    dropout: float = 0.3
    max_len: int = 64
    batch_size: int = 128
    eval_batch_size: int = 128
    lr: float = 1e-3
    max_epochs: int = 100
    patience: int = 5          # early stopping pada NDCG@10 validation
    max_delta_norm: float = 0.5  # delta_max; 0.5 -> cosine similarity minimum 0.87
    amp: bool = True           # bf16 autocast, penting di GPU 6 GB


@dataclass
class EvalConfig:
    top_k: int = 10
    # Daftar delta_max untuk sweep (replikasi Gambar 3 paper).
    delta_sweep: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


content_cfg = ContentConfig()
data_cfg = DataConfig()
model_cfg = ModelConfig()
eval_cfg = EvalConfig()

SEEDS = (42, 43, 44, 45, 46)  # 5 seed, sesuai paper


def ensure_dirs() -> None:
    for d in (INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
