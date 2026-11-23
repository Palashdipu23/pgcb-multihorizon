# Multi-Horizon Benchmark of Hybrid Deep Learning and Gradient-Boosting Models for PGCB Electricity Demand Forecasting

Code for the paper *"Multi-Horizon Benchmark of Hybrid Deep Learning and Gradient-Boosting Models for
Day-, Week- and Month-Ahead Electricity Demand Forecasting in Bangladesh Using PGCB Data"*.

Eight CNN-based hybrid models (M1–M8) are compared with XGBoost, LightGBM, Ridge, DLinear, an MLP on the
known-feature branch (K-MLP) and naive baselines for direct hourly forecasts of national PGCB demand at

* a rolling **24 h** lead, a fixed-issue **day-ahead** forecast (issued 10:00 on D−1 for all hours of D),
* **168 h** (week ahead) and **720 h** (month ahead),

with **five seeds (43–47)** and **three rolling test years** (ending 10 Mar 2024, 2025, 2026).
One frozen pipeline (same split, preprocessing and seeds) is used for every model, horizon and test year.

## Models

| ID | Architecture |
|----|--------------|
| M1 | CNN → 2-layer LSTM → mean‖max pooling → head |
| M2 | CNN → 2-layer BiLSTM → mean‖max pooling → head |
| M3 | CNN → Linear → Transformer encoder → mean‖max pooling → head |
| M4 | CNN → patching (len 12, stride 6) → Transformer encoder → pooling → head |
| M5 | CNN → 2-layer BiGRU → Transformer encoder → pooling → head |
| M6 | CNN → parallel BiGRU ‖ BiLSTM → fusion layer → head |
| M7 | CNN + spatial weather encoder (SWE) → BiGRU → Transformer with locality bias → head |
| M8 | CNN → 2-layer BiLSTM → Transformer encoder → pooling → head |

SWE = two graph-convolution layers over the 8 divisional weather points (`StationGCN` in `src/pgcb_lib.py`);
the graph connects weather points, not load nodes.

Baselines: XGBoost, LightGBM, Ridge, a classical MLR (hour × weekday, month, holiday × hour and lagged load; no weather), DLinear, K-MLP, persistence, seasonal naive, last-year-scaled and, for DA, the same hour on the latest observed day (D−1 if the hour is ≤ 10:00, else D−2).
Ablations: M7-MLP, M7-noLB, M7-noStation, M2+S, M5+S.

## Repository layout

```
pgcb-multihorizon/
├── README.md
├── LICENSE                     MIT
├── requirements.txt            versions used for the paper
├── configs/default.json        seeds 43-47, test-year folds, horizons
├── data/
│   ├── README.md               sources, licence, expected csv format
│   ├── stations.csv            8 divisional points (lat/lon) for NASA POWER
│   └── weather/download_nasa_power.py
├── src/
│   └── pgcb_lib.py             the whole pipeline (features, split, models, training, metrics, tables)
├── scripts/
│   ├── run_all.py              runs every step of the paper
│   └── run_all.sh              audit -> train -> evaluate -> tables
└── results/                    all paper results (predictions, metric tables, figures)
```

## Setup

```bash
git clone https://github.com/Palashdipu23/pgcb-multihorizon.git
cd pgcb-multihorizon
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the input file `PGCB_Demand_With_Weather_Holidays.csv` in `data/` (see [data/README.md](data/README.md)),
or pass its path with `--data`.

## Reproduce the paper

```bash
# quick check of the whole pipeline (1 seed, 2 epochs, short val/test): a few minutes
python scripts/run_all.py --smoke

# full paper run (GPU recommended; the paper used one NVIDIA T4)
bash scripts/run_all.sh

# parts of it
python scripts/run_all.py --horizons 24 DA --folds 2025-26
python scripts/run_all.py --steps main --models M7 XGBoost
python scripts/run_all.py --steps tables          # rebuild tables from saved predictions
```

| Step (`--steps`) | What it does | Paper |
|---|---|---|
| `audit` | outage detection, imputation fingerprints, holiday fix (Eid), MERRA-2 grid cells | Sec. IV-A, IV-B |
| `tune` | optional: equal random-search budget (8 configs) per neural model | Sec. IV-D (sensitivity) |
| `main` | M1–M8 + baselines, 5 seeds, 3 test years, 4 settings | Tables II, III |
| `ablation` | M7 without graph / locality bias / station input, M2/M5 with station input | Table V |
| `longlag` | long-lag features switched on at 168 h and off at 720 h | Table VI |
| `weatherlag` | 24 h and DA re-run with the window's weather lagged 72 h (NASA POWER meteorology arrives 2–3 days late) | Table VII, Sec. V-D |
| `tables` | MAE/RMSE/MAPE (seed mean ± SD), ensemble, MASE, R², MCS, DM vs strongest baseline (lowest validation MAPE, Holm), intervals, figures | Tables II, III, V, VI; Sec. V-E; Fig. 3 |
| `importance` | grouped permutation importance and Eid analysis (load shedding only if `load_shedding_csv` is set; not used in the paper) | Sec. V-F, V-G (Fig. 4) |
| `params` | parameter counts per horizon, `reproducibility.json` | Table I |
| `review` | review tables R1–R8 from saved predictions: MCS/DM in every test year, pooled 3-year DM with CIs, M7 vs ablations, matched-feature 168 h, weather lag, tuning gain, weather availability over three years, best hybrid vs baseline (no training) | Table IV (R8), Table VII (R7), Sec. IV-D, V-B–V-D |

Fig. 1 and Fig. 2 are schematic diagrams and are not generated by the code.

Finished runs are skipped on re-run, so an interrupted run can simply be started again.

Outputs:

```
results/<fold>/<setting>/<model>/    test_predictions.csv, val_predictions.csv, meta.json
                                     (seed*.pt / *.pkl model weights are written locally but not in the repo)
results/tables/                      A_main_comparison, B_rolling_origin_MAPE, ... (.csv, .tex, .md)
                                     mcs_<fold>_<setting>.csv, dm_vs_strongest_baseline_*.csv: all models (paper, Table II)
                                     mcs_hybrids_*.csv, dm_*_hybrids_*.csv: M1-M8 + strongest baseline (DM heatmaps only)
results/figures/                     DM p-value heatmaps, monthly MAPE, first test week, importance
results/data_audit/                  outage events, holiday counts, station grid cells
results/reproducibility.json         library / GPU versions, seeds, folds, hyperparameters
```

`<fold>` is `last` for the main test year (10 Mar 2025 – 10 Mar 2026) and `end20250310`, `end20240310`
for the earlier years; `<setting>` is `H24`, `HDA`, `H168`, `H720` (`_longON` / `_longOFF` for the switch).

## Protocol (summary)

* Look-back window 168 h ending at the forecast origin t − H; target is the scaled log-ratio to the demand at the origin.
* Validation: the 365 days before each test year; training: everything earlier. Scalers fitted on training rows only.
* Smooth-L1 loss, AdamW (lr 3e-4, weight decay 1e-2), batch 256 from 10 % of training origins per epoch,
  input noise 0.05, early stopping (patience 12), ensemble of 5 seeds.
* An automated leakage audit (24–27 checks per setting) runs before every training run.
* 95 % intervals: constant radius from validation residuals, hour-block radius, adaptive conformal (ACI).

## Citation

```bibtex
@inproceedings{pgcb_multihorizon,
  title  = {Multi-Horizon Benchmark of Hybrid Deep Learning and Gradient-Boosting Models for Day-, Week- and
            Month-Ahead Electricity Demand Forecasting in Bangladesh Using PGCB Data},
  author = {<Authors>},
  year   = {2026}
}
```

## Licence

Code: MIT (see `LICENSE`). Data are not covered by this licence; see `data/README.md`.
