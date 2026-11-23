# Results

All results of the paper, produced by `scripts/run_all.py` (Kaggle, NVIDIA T4).

| Path | Content |
|---|---|
| `<fold>/<setting>/<model>/` | `test_predictions.csv`, `val_predictions.csv` (MW, every seed and the 5-seed ensemble), `meta.json` |
| `tables/` | paper tables (A_main_comparison = Table II, B = Table III, C/D = Sec. V-E (operational errors and intervals), E = Table V, F = Table VI, G = Fig. 4, H = Eid, J = Table I) |
| `tables/mcs_*`, `tables/dm_*` | MCS and DM tests over all models (paper); `*_hybrids_*` = M1–M8 + strongest baseline (heatmaps only) |
| `tables/R1_*` … `R8_*` | review analyses: tests in every test year, pooled 3-year DM, M7 vs ablations, matched-feature 168 h, weather lagged 72 h, tuning gain, R7 = Table VII (weather availability), R8 = Table IV (best hybrid vs baseline) |
| `tuning/` | random-search trials (8 configs per neural model and setting, validation MAPE), Sec. IV-D |
| `figures/` | DM heatmaps, monthly MAPE (Fig. 3), grouped importance (Fig. 4), first test week |
| `data_audit/` | dataset description, outage events, imputation fingerprints, holidays, MERRA-2 grid cells |
| `reproducibility.json` | library and GPU versions, seeds, folds, hyperparameters |

`<fold>`: `last` = main test year (10 Mar 2025 – 10 Mar 2026), `end20250310`, `end20240310` = earlier test years.
`<setting>`: `H24`, `HDA`, `H168`, `H720` (`_longON` / `_longOFF` = long-lag switch).

Model weights (`seed*.pt`, `*.pkl`) are not included (about 1.5 GB). All tables can be rebuilt from the
predictions without the PGCB data: `python scripts/run_all.py --steps tables`.
