# Data

The models read one hourly csv: **`data/PGCB_Demand_With_Weather_Holidays.csv`**
(or any path given with `--data` / the `PGCB_CSV` environment variable).

## Sources

| Part | Source | Licence / access |
|---|---|---|
| Hourly national demand (MW), 28 Oct 2015 – 10 Mar 2026, 90,383 h | Power Grid Company of Bangladesh (PGCB), obtained through the authors' institution | Not redistributed here. Contact PGCB or the authors. |
| Hourly weather at 8 divisional points | NASA POWER (https://power.larc.nasa.gov), meteorology from MERRA-2 (0.5° × 0.625°) | Free to use, cite NASA POWER |
| Holidays | Bangladesh fixed-date national holidays; Eid / Christmas are added in code (`holiday_calendar` in `src/pgcb_lib.py`) | — |

Download the weather with:

```bash
python data/weather/download_nasa_power.py --start 20151028 --end 20260310
```

Station coordinates are in `stations.csv` (BAR, CTG, DHA, KHU, MYM, RAJ, RAN, SYL).

## Expected csv format

| Column | Meaning |
|---|---|
| `Date`, `Time` | local date (`YYYY-MM-DD`) and hour (`HH:MM[:SS]`) |
| `Sequence_ID` | id of each gap-free hourly segment (1 = 28 Oct 2015 – 19 Jan 2017, 2 = 9 Feb 2017 – 10 Mar 2026; 20-day gap in between) |
| `Demand` | hourly demand in MW (> 0) |
| `Holiday_Category` | 0 working day, 1 weekend (Fri/Sat), 2 public holiday |
| `Days_Since_Holiday`, `Days_Until_Holiday` | days to the previous / next public holiday |
| `{VAR}_{STATION}` | 7 weather variables × 8 stations = 56 columns, e.g. `T2M_DHA`; VAR ∈ T2M, RH2M, WS10M, PS, PRECTOTCORR, PSC, WSC |

Rows must be hourly and contiguous inside each `Sequence_ID`; the loader checks this.

Optional files (set in `configs/default.json`): `extra_holidays_csv` (column `date`, official gazette
holidays such as Durga Puja) and `load_shedding_csv` (columns `date`, `load_shedding_MW`).
