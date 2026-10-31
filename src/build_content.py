"""Langkah 2 - bangun embedding konten per video dari course.json.

Pipeline mengikuti paper Bagian 4.1.3:
    teks -> encoder E5 -> standardisasi per-komponen -> PCA ke 128 dim -> L2-normalize

Teks MOOCCubeX berbahasa Mandarin, jadi dipakai multilingual-E5, bukan E5 biasa.

Format satu baris course.json:
    {"id": "C_584313", "name": "...", "field": ["..."], "about": "...",
     "resource": [{"titles": [...], "resource_id": "V_849", "chapter": "1.1.1"}, ...]}
"""
from __future__ import annotations

import os

# transformers otomatis mencoba mengimpor TensorFlow kalau terpasang. TF versi
# lama di environment ini tidak kompatibel dengan numpy 2.x dan crash saat impor,
# jadi backend dikunci ke PyTorch. Harus diset SEBELUM transformers diimpor.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from progress import bar, file_bar
from transformers import AutoModel, AutoTokenizer

from config import (CONTENT_ITEMS_PATH, CONTENT_PATH, CONTENT_RAW_PATH,
                    COURSE_JSON, ITEM_IDS_PATH, ITEM_TEXT_PATH, content_cfg,
                    ensure_dirs)

# Deskripsi course dipotong supaya tidak mendominasi teks item.
MAX_ABOUT_CHARS = 120


def clean_titles(titles) -> str:
    """Gabungkan judul bab/sub-bab/video, buang None dan duplikat berurutan."""
    out: list[str] = []
    for t in titles or []:
        if not t:
            continue
        t = str(t).strip()
        if t and (not out or out[-1] != t):
            out.append(t)
    return " / ".join(out)


def build_texts(course_path: Path, keep_items: set[str] | None):
    """Hasilkan {video_id: teks}. Video yang muncul di banyak course diambil yang pertama."""
    texts: dict[str, str] = {}

    pbar = file_bar(course_path, "parse course.json")
    with course_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            pbar.update(len(line.encode("utf-8", "ignore")))
            pbar.set_postfix_str(f"video={len(texts):,}")
            line = line.strip()
            if not line:
                continue
            try:
                course = json.loads(line)
            except json.JSONDecodeError:
                continue

            name = (course.get("name") or "").strip()
            fields = "、".join(course.get("field") or [])
            about = (course.get("about") or "").strip()

            for res in course.get("resource") or []:
                vid = res.get("resource_id")
                # resource bisa berupa video (V_) atau latihan (Ex_); hanya video dipakai.
                if not vid or not vid.startswith("V_") or vid in texts:
                    continue
                if keep_items is not None and vid not in keep_items:
                    continue

                chapter = (res.get("chapter") or "").strip()
                title = clean_titles(res.get("titles"))
                # Urutan field PENTING. Deskripsi course (`about`) identik untuk
                # semua video dalam satu course dan bisa sampai 600+ karakter.
                # Kalau ditaruh di depan, judul video yang justru membedakan antar
                # item akan terpotong oleh batas token - hasilnya 45% item punya
                # kembaran dengan cosine similarity > 0,99 dan sinyal kontennya mati.
                # Jadi: info spesifik video dulu, konteks course menyusul, dan
                # `about` dipotong supaya tidak menenggelamkan sisanya.
                about_short = about[:MAX_ABOUT_CHARS]
                # Prefix "passage: " adalah format wajib E5 untuk sisi dokumen.
                texts[vid] = (
                    f"passage: 视频：{title}。章节：{chapter}。"
                    f"课程：{name}。领域：{fields}。简介：{about_short}"
                )

    pbar.close()
    return texts


@torch.no_grad()
def encode(texts: list[str], cfg, device: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModel.from_pretrained(cfg.model_name).to(device).eval()

    vectors = []
    pbar = bar(len(texts), "encode multilingual-E5", unit="teks")
    for i in range(0, len(texts), cfg.encode_batch_size):
        batch = texts[i:i + cfg.encode_batch_size]
        enc = tokenizer(batch, max_length=cfg.max_text_len, padding=True,
                        truncation=True, return_tensors="pt").to(device)
        out = model(**enc).last_hidden_state

        # Average pooling bermasking - prosedur pooling resmi E5.
        mask = enc["attention_mask"].unsqueeze(-1).bool()
        pooled = out.masked_fill(~mask, 0.0).sum(1) / enc["attention_mask"].sum(1, keepdim=True)
        vectors.append(F.normalize(pooled, dim=-1).float().cpu().numpy())
        pbar.update(len(batch))

    pbar.close()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(vectors, axis=0)


def reduce_dim(raw: np.ndarray, target_dim: int) -> np.ndarray:
    """Standardisasi per-komponen -> PCA -> L2-normalize (paper 4.1.3)."""
    if raw.shape[0] <= target_dim:
        raise SystemExit(
            f"Jumlah item ({raw.shape[0]}) harus lebih besar dari target_dim ({target_dim}).")

    print(f"  standardisasi {raw.shape[0]:,} x {raw.shape[1]} ...", flush=True)
    scaled = StandardScaler().fit_transform(raw)
    print(f"  PCA -> {target_dim} dimensi ...", flush=True)
    reduced = PCA(n_components=target_dim, svd_solver="randomized",
                  random_state=0).fit_transform(scaled)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    return (reduced / np.clip(norms, 1e-8, None)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--course", type=Path, default=COURSE_JSON)
    ap.add_argument("--items-from", type=Path, default=ITEM_IDS_PATH,
                    help="JSON berisi daftar video yang dipakai; "
                         "kalau tidak ada, semua video di course.json di-encode")
    ap.add_argument("--target-dim", type=int, default=content_cfg.target_dim)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--reuse-raw", action="store_true",
                    help="pakai ulang embedding E5 mentah yang sudah tersimpan")
    ap.add_argument("--texts-only", action="store_true",
                    help="berhenti setelah menulis item_text.jsonl, tanpa encoding. "
                         "Dipakai di langkah 0 untuk menyediakan daftar video "
                         "berkonten bagi prepare_sequences.py")
    args = ap.parse_args()

    ensure_dirs()
    if not args.course.exists():
        raise SystemExit(f"course.json belum ada: {args.course}")

    keep_items = None
    if args.items_from and args.items_from.exists():
        keep_items = set(json.loads(args.items_from.read_text(encoding="utf-8")))
        print(f"membatasi ke {len(keep_items):,} video yang muncul di interaksi")

    texts = build_texts(args.course, keep_items)
    if not texts:
        raise SystemExit("Tidak ada video yang cocok. Cek item_ids.json / course.json.")

    item_ids = sorted(texts)
    with ITEM_TEXT_PATH.open("w", encoding="utf-8") as fh:
        for vid in item_ids:
            fh.write(json.dumps({"item_id": vid, "text": texts[vid]},
                                ensure_ascii=False) + "\n")
    print(f"{len(item_ids):,} video punya teks konten -> {ITEM_TEXT_PATH}")

    if args.texts_only:
        print("--texts-only: berhenti di sini, encoding dilewati.")
        return

    if args.reuse_raw and CONTENT_RAW_PATH.exists():
        raw = np.load(CONTENT_RAW_PATH)
        if raw.shape[0] != len(item_ids):
            raise SystemExit("Jumlah baris embedding mentah tidak cocok; jalankan tanpa --reuse-raw.")
    else:
        raw = encode([texts[v] for v in item_ids], content_cfg, args.device)
        np.save(CONTENT_RAW_PATH, raw)

    emb = reduce_dim(raw, args.target_dim)
    np.save(CONTENT_PATH, emb)
    CONTENT_ITEMS_PATH.write_text(json.dumps(item_ids, ensure_ascii=False), encoding="utf-8")

    print(f"\nembedding mentah : {raw.shape}")
    print(f"setelah PCA      : {emb.shape}  (norm rata2 {np.linalg.norm(emb, axis=1).mean():.4f})")
    print(f"tersimpan -> {CONTENT_PATH}")
    print(f"tersimpan -> {CONTENT_ITEMS_PATH}")


if __name__ == "__main__":
    main()
