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
| HR@10 | Content-based KNN | 0.3661±0.000 | 0.3376±0.000 | 0.3494±0.000 |
| HR@10 | SASRec | 0.0000±0.000 | 0.6484±0.013 | 0.3784±0.008 |
| HR@10 | SASRec with c.i. | 0.3554±0.021 | 0.6994±0.008 | 0.5561±0.011 |
| HR@10 | SASRec with t.d. (ours) | 0.2714±0.024 | 0.7083±0.010 | 0.5264±0.009 |
| NDCG@10 | Content-based KNN | 0.1976±0.000 | 0.1933±0.000 | 0.1951±0.000 |
| NDCG@10 | SASRec | 0.0000±0.000 | 0.5084±0.004 | 0.2967±0.003 |
| NDCG@10 | SASRec with c.i. | 0.1663±0.010 | 0.5294±0.003 | 0.3782±0.005 |
| NDCG@10 | SASRec with t.d. (ours) | 0.1221±0.013 | 0.5325±0.004 | 0.3617±0.007 |
