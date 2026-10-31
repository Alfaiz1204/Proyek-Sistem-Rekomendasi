# Hasil utama (replikasi struktur Tabel 2)

## Statistik data

- interaksi_awal: 84056
- interaksi_berkonten: 84056
- boundary_ts: 1603790118
- n_users_train: 2544
- n_users_test: 269
- n_items_total: 3977
- n_items_train: 3117
- n_items_cold: 860
- pct_items_cold: 21.62
- pct_test_cold_gt: 41.64
- avg_seq_len: 25.6

## Hasil

| Metric | Model | Cold GT | Warm GT | Total |
|---|---|---|---|---|
| HR@10 | SASRec with t.d. (ours) | 0.4536±0.017 | 0.6917±0.006 | 0.5926±0.010 |
| NDCG@10 | SASRec with t.d. (ours) | 0.2115±0.014 | 0.4619±0.014 | 0.3577±0.007 |
