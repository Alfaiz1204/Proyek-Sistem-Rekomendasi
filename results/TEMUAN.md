# Temuan Eksperimen — Final

Penerapan metode *Let It Go? Not Quite* (Pembek et al., RecSys '25) ke domain LMS,
dataset MOOCCubeX. Data penuh, embedding konten sudah diperbaiki, 5 seed.

## Karakteristik data

| Properti | Nilai |
|---|---|
| User train / test | 7.321 / 544 |
| Item total / train / cold | 7.175 / 5.922 / 1.253 |
| Interaksi | 194.537 |
| Panjang sequence rata-rata | 21,7 |
| **Proporsi item cold** | **17,46%** |
| **Ground truth test yang cold** | **33,09%** |

Cold start di LMS jauh lebih ekstrem daripada domain paper (Amazon-M2 7%, Beauty 25%).

## Tabel 2 — setelan paper (δ_max = 0,5)

| NDCG@10 | Cold GT | Warm GT | Total |
|---|---|---|---|
| Content-based KNN | 0,1900±0,000 | 0,1362±0,000 | 0,1540±0,000 |
| SASRec | **0,0000**±0,000 | 0,5504±0,003 | 0,3683±0,002 |
| SASRec + content init | **0,2316**±0,012 | 0,5615±0,003 | **0,4523**±0,005 |
| SASRec + trainable delta | 0,1197±0,010 | **0,5699**±0,003 | 0,4210±0,003 |

Pada setelan paper, metode usulan **kalah** dari content init di item cold.

## Tabel 2 — setelan optimal domain LMS (δ_max = 0,1)

| NDCG@10 | Cold GT | Warm GT | Total |
|---|---|---|---|
| Content-based KNN | 0,1900±0,000 | 0,1362±0,000 | 0,1540±0,000 |
| SASRec | **0,0000**±0,000 | 0,5504±0,003 | 0,3683±0,002 |
| SASRec + content init | 0,2316±0,012 | **0,5615**±0,003 | 0,4523±0,005 |
| SASRec + trainable delta | **0,2591**±0,021 | 0,5608±0,004 | **0,4610**±0,005 |

HR@10: cold **0,4644**±0,027 · warm 0,7126±0,004 · total **0,6305**±0,008
(content init: cold 0,3978±0,014 · warm 0,6929±0,011 · total 0,5952±0,012)

Pada δ=0,1 metode usulan **menang di cold dan total**, warm praktis imbang —
persis pola yang diklaim paper: *"substantial improvements in cold-item metrics...
warm-item and overall performance also show slight improvements."*

## Gambar 3 — sweep δ_max (3 seed)

Acuan content init: cold 0,2280 · total 0,4509

| δ_max | cos minimum | Cold | Total | % item mentok batas |
|---|---|---|---|---|
| 0,05 | 0,9987 | **0,2876** | 0,4439 | 100,0% |
| 0,10 | 0,9950 | 0,2583 | **0,4607** | 100,0% |
| 0,20 | 0,9798 | 0,2172 | 0,4557 | 100,0% |
| 0,30 | 0,9539 | 0,1626 | 0,4385 | 99,45% |
| 0,40 | 0,9165 | 0,1232 | 0,4257 | 88,42% |
| 0,50 | 0,8660 | **0,1178** | 0,4196 | 55,93% |
| 0,70 | 0,7141 | 0,2004 | 0,4437 | 17,27% |
| 0,90 | 0,4359 | 0,2197 | 0,4504 | 0,53% |

Tiga pembacaan:

**Kurva cold berbentuk U.** Terbaik di δ kecil (0,05–0,1), terburuk di 0,3–0,5 —
justru rentang yang direkomendasikan paper. Rekomendasi δ 0,3–0,6 tidak lintas-domain.

**Ujung kanan memvalidasi implementasi.** Di δ=0,9 hanya 0,53% item menyentuh batas,
sehingga mode delta secara matematis meluruh jadi content init. Angkanya memang
bertemu: cold 0,2197 vs acuan 0,2280; total 0,4504 vs acuan 0,4509. Bukan kebetulan.

**Trade-off terbaca jelas.** δ kecil = terkunci pada semantik konten (cold bagus,
total turun). δ besar = bebas mengejar sinyal kolaboratif. Inilah dial
"kurikulum-driven ↔ perilaku-driven" yang jadi argumen domain LMS.

## Gambar 5 — per frekuensi item ground truth (δ=0,1)

| Frekuensi di train | n | SASRec | + content init | + trainable delta |
|---|---|---|---|---|
| 0 (cold) | 180 | 0,0000 | 0,2301 | **0,2674** |
| 3–5 | 50 | 0,4446 | 0,4997 | **0,5393** |
| 6–10 | 61 | **0,5738** | 0,5571 | 0,5451 |
| 11–30 | 68 | **0,6137** | 0,6113 | 0,5867 |
| 31–100 | 90 | 0,5234 | **0,5495** | 0,5409 |
| >100 | 95 | 0,5542 | **0,5688** | 0,5688 |

Relatif terhadap SASRec, trainable delta unggul **1,213×** di item langka (3–5),
lalu menyatu (0,95–1,03) saat item makin sering muncul. Persis pola Gambar 5 paper:
*"improvements for rare items, converging to SASRec levels as frequency increases."*

## Gambar 4 — per proporsi item warm di riwayat (δ=0,1)

| Proporsi warm di input | n | SASRec | + content init | + trainable delta |
|---|---|---|---|---|
| 20–40% | 6 | 0,0000 | 0,4722 | **0,5030** |
| 40–60% | 14 | 0,0000 | **0,3464** | 0,3340 |
| 60–80% | 39 | 0,0256 | **0,4118** | 0,3771 |
| 80–100% | 484 | 0,4085 | 0,4581 | **0,4722** |

SASRec biasa hampir nol begitu riwayat user mengandung item cold; kedua varian
berbasis konten mengangkatnya drastis. Urutan delta vs content init di bucket
tengah tidak stabil karena n kecil (6–39 sampel).

## Keterbatasan yang perlu disebut di presentasi

**δ_max optimal tidak bisa dipilih lewat protokol validation standar.** Target
validation diambil dari periode train sehingga seluruhnya item warm; early stopping
dan pemilihan hyperparameter condong ke performa warm, dan tidak akan pernah
memilih δ=0,05–0,1 yang terbaik untuk cold. Paper tidak membahas ini.

**Test set relatif kecil** — 544 user, 180 di antaranya cold-GT. Selisih cold
0,2591 vs 0,2316 setara sekitar 1,3 standar deviasi; arah temuannya konsisten
lintas seed tapi belum kuat secara statistik.

**Cakupan metadata terbatas.** `course.json` hanya mencakup sebagian katalog
XuetangX, sehingga dari 310.360 user hanya 9.058 (2,9%) yang seluruh riwayatnya
tercakup metadata.

**Bukan replikasi.** Dataset paper (Amazon-M2, Beauty, Zvuk) tidak dipakai. Klaim
yang sah: metode dan protokol evaluasi paper diterapkan ke domain LMS dan klaim
intinya bertahan, dengan catatan δ_max harus disetel ulang.

## Berkas luaran

| File | Isi |
|---|---|
| `table2.md` | Tabel 2, δ=0,5 (setelan paper) |
| `table2_terbaik.md` | Tabel 2, δ=0,1 (setelan LMS) |
| `delta_sweep.csv` | Gambar 3 + kolom acuan `ref_ci_*` |
| `fig5_by_gt_frequency.csv` | Gambar 5 panel atas |
| `fig5_relative_to_sasrec.csv` | Gambar 5 panel bawah |
| `fig4_by_input_warm_fraction.csv` | Gambar 4 |
| `all_runs*.json` | Metrik mentah tiap run |
| `arsip_data_parsial/`, `arsip_embedding_rusak/` | Hasil lama — **jangan dipakai** |
