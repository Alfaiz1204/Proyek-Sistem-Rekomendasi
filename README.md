# Cold-Start Materi Pembelajaran pada LMS dengan Bounded Content-Embedding Adaptation

Reproduksi dan penerapan lintas-domain metode **"Let It Go? Not Quite: Addressing Item
Cold Start in Sequential Recommendations with Content-Based Initialization"**
(Pembek, Fatkulin, Klenitskiy, Vasilev — RecSys '25, hlm. 626–631, DOI `10.1145/3705328.3748038`)
dari domain e-commerce/musik ke domain **Learning Management System**.

Dataset: **MOOCCubeX** (XuetangX / Tsinghua KEG).

## Inti metode

Embedding item SASRec diinisialisasi dari embedding konten `c_i`, lalu `c_i` **dibekukan**
dan yang dilatih hanya vektor koreksi `d_i` dengan norm dibatasi:

```
e_i = c_i + d_i ,   ‖c_i‖ = 1 ,   ‖d_i‖ ≤ δ_max
```

Batas ini menjamin `cos(c_i, e_i) ≥ √(1 − δ_max²)`. Dengan `δ_max = 0.5`, cosine
similarity minimum adalah 0.87. Akibatnya item cold — yang tidak pernah menerima
gradien dan tetap berada di `c_i` — hidup di ruang representasi yang sama dengan
item warm, sehingga bisa direkomendasikan tanpa merusak performa item warm.

## Prasyarat

**Wajib pakai Python 3.10 64-bit.** Python 3.8 32-bit yang ada di PATH tidak bisa:
torch tidak tersedia untuk Windows 32-bit, dan batas memori prosesnya tidak cukup.

```powershell
$PY = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
& $PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Paket yang dibutuhkan sudah terpasang di environment ini (torch 2.5.1+cu121,
transformers, scikit-learn, pandas, pyarrow, tqdm). Kalau pindah mesin:

```powershell
& $PY -m pip install -r requirements.txt
```

## Data

Unduh dari MOOCCubeX (`https://lfs.aminer.cn/misc/moocdata/data/mooccube2/`) ke
`data/raw/mooccubex/`:

| File | Ukuran | Wajib | Isi |
|---|---|---|---|
| `entities/course.json` | 44,6 MB | ✅ | `id`, `name`, `field`, `about`, `resource[].titles`, `chapter` |
| `relations/user-video.json` | 3,18 GB | ✅ | sequence per user + `local_start_time` |
| `prerequisites/{cs,math,psy}.json` | ~228 MB | opsional | relasi prasyarat antar konsep, untuk analisis tambahan |

Keduanya berformat **JSON Lines** (satu objek per baris), bukan satu array JSON besar.

## Menjalankan pipeline

```powershell
$PY = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
cd src

# 0. Ekstrak teks konten SEMUA video di course.json (tanpa encoding, ~3 detik)
& $PY build_content.py --texts-only --items-from nonexistent

# 1. Sampling user, DISARING ke video yang punya metadata konten (streaming, hemat RAM)
& $PY prepare_sequences.py --n-users 20000

# 2. Encode hanya video yang lolos: teks -> multilingual-E5 -> standardisasi -> PCA 128 -> L2
& $PY build_content.py

# 3. Smoke test satu model dulu sebelum eksperimen penuh
& $PY train.py --mode delta --seed 42 --max-epochs 3 --rebuild-splits

# 4. Eksperimen penuh: 4 model x 5 seed -> results/table2.md
& $PY run_all.py

# 5. Sweep delta_max (replikasi Gambar 3 paper) -> results/delta_sweep.csv
& $PY sweep_delta.py

# 6. Replikasi Gambar 4 dan 5 -> results/fig4_*.csv, results/fig5_*.csv
& $PY analysis.py
```

**Langkah 0 tidak boleh dilewati.** `course.json` hanya mencakup sebagian katalog
XuetangX: dari 53.159 video yang benar-benar ditonton pada sampel user acak, hanya
7.896 (14,9%) punya metadata. Kalau user disampel dulu baru kontennya disaring,
85% interaksi rontok dan sequence jadi bolong. Dengan penyaringan di depan,
user yang disampel adalah user yang riwayatnya memang tercakup, sehingga
sequence-nya utuh. `entities/video.json` tidak bisa menambal ini — file itu
berkunci `ccid` (hash), bukan `V_xxx`, dan tidak ada tabel pemetaan di MOOCCubeX.

Atau sekaligus: `powershell -ExecutionPolicy Bypass -File run_pipeline.ps1`

## Struktur kode

| File | Peran |
|---|---|
| `src/config.py` | Seluruh hyperparameter dan path, terpusat di satu tempat |
| `src/prepare_sequences.py` | Streaming JSONL 3,18 GB + reservoir sampling user |
| `src/build_content.py` | Teks konten → E5 → standardisasi → PCA → L2-normalize |
| `src/data.py` | Global temporal split, k-core, penentuan item cold, Dataset PyTorch |
| `src/models.py` | SASRec + tiga mode embedding (`random`, `content_init`, `delta`) |
| `src/metrics.py` | HR@K dan NDCG@K, dipecah cold GT / warm GT / total |
| `src/baselines.py` | Content-based KNN (tanpa training) |
| `src/train.py` | Training satu kombinasi (mode, seed) + early stopping |
| `src/run_all.py` | 4 model × 5 seed → tabel hasil |
| `src/sweep_delta.py` | Sweep `δ_max` |

## Model yang dibandingkan

| Mode | Keterangan | Bisa menangani item cold? |
|---|---|---|
| `content_knn` | Rata-rata embedding konten riwayat user, cosine similarity. Tanpa training. | Ya, tapi lemah di item warm |
| `random` | SASRec standar, embedding acak | **Tidak** — item cold di-mask saat evaluasi |
| `content_init` | Embedding diinisialisasi dari konten, lalu di-fine-tune penuh | Sebagian |
| `delta` | **Usulan paper** — konten beku + delta ber-norm terbatas | Ya |

## Keputusan metodologis yang penting

**Item cold ditentukan oleh split temporal, bukan sampling acak.** Interaksi dipotong
di kuantil 90% dari seluruh timestamp; item cold adalah video yang hanya muncul
setelah batas itu. Kalau interaksi dibuang secara acak lalu disebut "cold", model
sudah pernah melihat item tersebut saat training dan angka cold-start jadi palsu.

**SASRec biasa dilarang merekomendasikan item cold.** Model itu memang tidak punya
representasi terlatih untuk item tersebut. Inilah yang menghasilkan nilai tepat
`0.0000` pada kolom cold di Tabel 2 paper, dan kode ini mereproduksinya dengan
mask eksplisit, bukan kebetulan numerik.

**k-core hanya diterapkan pada data train/validation.** Kalau diterapkan ke data
test, item cold justru ikut terbuang — padahal item itulah objek penelitiannya.

**Teks MOOCCubeX berbahasa Mandarin**, jadi dipakai `intfloat/multilingual-e5-base`.
E5 versi bahasa Inggris tidak akan menghasilkan representasi yang bermakna.

## Catatan performa

GPU acuan: RTX 4050 Laptop, 6 GB VRAM.

- Full cross-entropy dihitung hanya pada posisi non-padding — memangkas VRAM
  sekitar setengah dibanding menghitung seluruh posisi.
- `amp=True` memakai autocast bf16 (didukung Ada Lovelace).
- Kalau tetap kehabisan VRAM: turunkan `--batch-size` ke 64, atau `max_len` ke 32
  di `config.py`.
- Kalau `n_users` dinaikkan, katalog ikut membesar dan biaya cross-entropy naik linear.

## Luaran

| File | Isi |
|---|---|
| `results/table2.md` | Tabel utama, mean ± std atas 5 seed |
| `results/all_runs.json` | Metrik mentah tiap run |
| `results/delta_sweep.csv` | Hasil sweep `δ_max` |

## Arah pengembangan

Paper menyebut keterbatasan **memory overhead 2×** (satu vektor konten + satu delta
per item). Celah yang bisa diklaim sebagai kontribusi:

1. **`δ_max` adaptif per item** — item langka diberi `δ` kecil (percaya konten),
   item populer diberi `δ` besar (percaya sinyal kolaboratif).
2. **Delta low-rank** `D = AB` dengan `r ≪ d` — menjawab langsung limitasi memori.
3. **Ganti backbone** ke GRU4Rec/BERT4Rec untuk menguji apakah efeknya generik.
4. **Analisis prasyarat** — pakai `prerequisites/*.json` untuk mengukur berapa
   sering rekomendasi melanggar urutan prasyarat pada tiap nilai `δ_max`.
   Analisis ini tidak mungkin dilakukan di domain paper aslinya.
