# Hasil utama (replikasi struktur Tabel 2)

## Statistik data

- interaksi_awal: 194537
- interaksi_berkonten: 194537
- boundary_ts: 1603808855
- n_users_train: 7321
- n_users_test: 544
- n_items_total: 7175
- n_items_train: 5922
- n_items_cold: 1253
- pct_items_cold: 17.46
- pct_test_cold_gt: 33.09
- avg_seq_len: 21.7

## Hasil

| Metric | Model | Cold GT | Warm GT | Total |
|---|---|---|---|---|
| HR@10 | Content-based KNN | 0.3333±0.000 | 0.3104±0.000 | 0.3180±0.000 |
| HR@10 | SASRec | 0.0000±0.000 | 0.6626±0.007 | 0.4434±0.005 |
| HR@10 | SASRec with c.i. | 0.3978±0.014 | 0.6929±0.011 | 0.5952±0.012 |
| HR@10 | SASRec with t.d. (ours) | 0.4644±0.027 | 0.7126±0.004 | 0.6305±0.008 |
| NDCG@10 | Content-based KNN | 0.1900±0.000 | 0.1362±0.000 | 0.1540±0.000 |
| NDCG@10 | SASRec | 0.0000±0.000 | 0.5504±0.003 | 0.3683±0.002 |
| NDCG@10 | SASRec with c.i. | 0.2316±0.012 | 0.5615±0.003 | 0.4523±0.005 |
| NDCG@10 | SASRec with t.d. (ours) | 0.2591±0.021 | 0.5608±0.004 | 0.4610±0.005 |
