# Temuan Eksperimen

Penerapan metode *Let It Go? Not Quite* (RecSys '25) ke domain LMS, dataset MOOCCubeX.

> **Status data: PARSIAL.** Angka di bawah berasal dari 134.342 user pertama
> `user-video.json` (sekitar 11% file). Setelah unduhan penuh selesai, jalankan
> ulang `run_pipeline.ps1`. Test set saat ini hanya 269 user (112 cold, 157 warm),
> jadi semua selisih di bawah perlu dikonfirmasi ulang pada data penuh.

## Karakteristik data

| Properti | Nilai | Pembanding di paper |
|---|---|---|
| User train | 2.544 | Beauty 21.029 |
| Item total | 3.977 | Beauty 11.733 |
| Interaksi | 84.056 | Beauty 149.147 |
| Panjang sequence rata-rata | 25,6 | Beauty 4,3 |
| **Item cold** | **21,6%** | — |
| **Ground truth test cold** | **41,6%** | Beauty 25%, Amazon-M2 7% |

Cold start di LMS memang jauh lebih ekstrem daripada domain di paper. Ini
memperkuat alasan pemilihan domain, bukan sekadar mengganti dataset.

## Hasil utama (NDCG@10, 5 seed, mean ± std)

| Model | Cold GT | Warm GT | Total |
|---|---|---|---|
| Content-based KNN | 0,1976±0,000 | 0,1933±0,000 | 0,1951±0,000 |
| SASRec | **0,0000**±0,000 | 0,5084±0,004 | 0,2967±0,003 |
| SASRec + content init | 0,1663±0,010 | 0,5294±0,003 | **0,3782**±0,005 |
| SASRec + trainable delta, δ=0,5 *(setelan paper)* | 0,1221±0,013 | **0,5325**±0,004 | 0,3617±0,007 |
| SASRec + trainable delta, δ=0,1 *(setelan domain ini)* | **0,2115**±0,014 | 0,4619±0,014 | 0,3577±0,007 |

HR@10 untuk δ=0,1: cold 0,4536±0,017 · warm 0,6917±0,006 · total 0,5926±0,010.

### Yang tereproduksi

1. SASRec biasa mencetak **tepat 0,0000** pada item cold. Ini bukan kebetulan
   numerik — model itu memang tidak punya representasi terlatih untuk item cold,
   dan kode melarangnya secara eksplisit.
2. Inisialisasi konten mengangkat cold dari 0 ke 0,1663 **tanpa mengorbankan warm**;
   warm justru naik dari 0,5084 ke 0,5294. Persis klaim paper.
3. Trainable delta mengungguli semua baseline pada item cold — **tetapi hanya
   setelah δ_max disetel ulang untuk domain ini**.

### Yang TIDAK tereproduksi

Pada δ_max = 0,5 (nilai yang dipakai paper), trainable delta **kalah** dari
content init pada item cold: 0,1221 vs 0,1663. Rekomendasi paper bahwa
δ_max 0,3–0,6 memberi trade-off terbaik **tidak berlaku di domain LMS**.

## Sweep δ_max — inti temuan

| δ_max | cos minimum | Cold | Warm | Total | % item mentok batas |
|---|---|---|---|---|---|
| 0,05 | 0,9987 | **0,2210** | 0,4070 | 0,3295 | 100% |
| 0,10 | 0,9950 | 0,2155 | 0,4676 | 0,3626 | 100% |
| 0,20 | 0,9798 | 0,1602 | 0,5239 | 0,3725 | 99,98% |
| 0,30 | 0,9539 | 0,1325 | 0,5338 | 0,3667 | 98,63% |
| 0,40 | 0,9165 | 0,1165 | 0,5315 | 0,3587 | 85,68% |
| 0,50 | 0,8660 | 0,1265 | 0,5321 | 0,3632 | 60,97% |
| 0,70 | 0,7141 | 0,1638 | **0,5323** | **0,3789** | 17,05% |
| 0,90 | 0,4359 | 0,1619 | 0,5301 | 0,3768 | 0,44% |

Tiga pembacaan penting:

**Kurva cold berbentuk U, bukan berpuncak di tengah.** Performa cold terbaik di
δ sangat kecil (0,05–0,1), terburuk di 0,3–0,5 — justru rentang yang
direkomendasikan paper. Lalu naik lagi di δ besar.

**Ujung kanan kurva memvalidasi implementasi.** Pada δ=0,9 hanya 0,44% item
menyentuh batas, jadi secara matematis mode delta meluruh menjadi content init.
Angkanya memang bertemu: cold 0,1619 vs 0,1663, total 0,3768 vs 0,3782. Ini
pemeriksaan kebenaran kode yang kuat — bukan kebetulan.

**Trade-off-nya tajam dan terbaca.** δ kecil = model terkunci pada semantik
konten (cold bagus, warm jatuh ke 0,407). δ besar = model bebas mengejar sinyal
kolaboratif (warm 0,532, cold anjlok). Inilah dial "kurikulum-driven ↔
perilaku-driven" yang jadi argumen utama presentasi.

## Analisis per frekuensi item (replikasi Gambar 5, δ=0,1)

| Frekuensi GT di train | n | SASRec | + content init | + trainable delta |
|---|---|---|---|---|
| 0 (cold) | 112 | 0,0000 | 0,1567 | **0,2178** |
| 3–5 | 22 | 0,4374 | 0,4315 | **0,4746** |
| 6–10 | 43 | 0,5405 | **0,5523** | 0,4619 |
| 11–30 | 40 | 0,7123 | **0,7267** | 0,6187 |
| 31–100 | 32 | 0,2627 | **0,3674** | 0,2887 |
| >100 | 20 | **0,4747** | 0,4368 | 0,4112 |

Pola paper muncul jelas: keunggulan terbesar ada di item langka, lalu mengecil
dan berbalik seiring item makin sering muncul di train.

## Analisis per komposisi riwayat (replikasi Gambar 4, δ=0,1)

| Proporsi item warm di input | n | SASRec | + content init | + trainable delta |
|---|---|---|---|---|
| 40–60% | 11 | 0,0000 | 0,0888 | **0,2069** |
| 60–80% | 29 | 0,0345 | 0,3103 | **0,3614** |
| 80–100% | 229 | 0,3417 | **0,3946** | 0,3674 |

Makin banyak item cold di riwayat user, makin besar keunggulan trainable delta —
2,3× lipat atas content init saat riwayatnya separuh cold. Ini relevan langsung
untuk LMS di awal semester, ketika sebagian besar materi yang diakses masih baru.

## Keterbatasan metodologis yang perlu disebut di presentasi

**δ_max optimal tidak bisa dipilih lewat protokol validation standar.** Target
validation diambil dari periode train, jadi seluruhnya item warm. Early stopping
dan pemilihan hyperparameter karenanya condong ke performa warm, dan tidak akan
pernah memilih δ=0,05 yang justru terbaik untuk cold. Paper tidak membahas ini.
Menyelesaikannya butuh validation set yang sengaja memuat item cold — misalnya
dengan memotong periode validation secara temporal juga, bukan mengambil
interaksi terakhir tiap user.

**Test set masih kecil.** 269 user, 112 di antaranya cold-GT. Selisih 0,045
antara δ=0,1 dan δ=0,5 setara sekitar 3 standar deviasi, jadi kemungkinan nyata,
tapi wajib dikonfirmasi pada data penuh.

**Cakupan metadata terbatas.** `course.json` hanya mencakup sebagian katalog
XuetangX. Dari sampel user acak, hanya 14,9% video yang ditonton punya metadata.
Pipeline menyaring ke user yang riwayatnya tercakup, sehingga sequence utuh, tapi
konsekuensinya hanya 3,1% user yang memenuhi syarat.

## Berkas luaran

| File | Isi |
|---|---|
| `table2.md` | Tabel utama, δ_max = 0,5 (setelan paper) |
| `table2_delta0.1.md` | Tabel pada δ_max = 0,1 |
| `delta_sweep.csv` | Sweep δ_max lengkap |
| `fig5_by_gt_frequency.csv` | Replikasi Gambar 5 |
| `fig4_by_input_warm_fraction.csv` | Replikasi Gambar 4 |
| `all_runs*.json` | Metrik mentah tiap run |
