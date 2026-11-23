"""
PGCB multi-horizon load forecasting -- full experiment runner for the paper.

Every step of the experiments is one `--steps` entry, so the whole paper can be reproduced with one command:

    python scripts/run_all.py                           # everything, all horizons, all folds
    python scripts/run_all.py --horizons 24             # one horizon
    python scripts/run_all.py --folds 2025-26           # main test year only
    python scripts/run_all.py --steps main tables       # only training + tables
    python scripts/run_all.py --smoke                   # 1 seed, 2 epochs: checks the pipeline in minutes

Steps (in order):
    audit       data audit: outages, imputation fingerprints, holiday fix, NASA POWER grid
    tune        (optional) equal random-search budget per neural model
    main        M1-M8 + baselines, all seeds, all folds          -> results/<fold>/<setting>/<model>/
    ablation    M7 ablations and M2/M5 with station input
    longlag     long-lag feature switch at 168 h and 720 h
    tables      paper tables A-F, MCS, DM tests, figures           -> results/tables, results/figures
    importance  grouped permutation importance, Eid and load-shedding analysis
    params      parameter counts per horizon + reproducibility.json
    weatherlag  24 h and DA again with the window's weather lagged 72 h (NASA POWER arrives 2-3 days late)
    review      review tables R1-R8 from saved predictions (tests in every test year, pooled DM,
                M7 vs ablations, matched-feature 168 h, weather lag, tuning gain); no training

Finished runs are skipped on re-run (resume after a crash / Kaggle restart).
"""
import argparse
import gc
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib
matplotlib.use("Agg")

from pgcb_lib import *          # noqa: E402,F401,F403  (shared pipeline)
import pgcb_lib as L            # noqa: E402

ALL_STEPS = ["audit", "tune", "main", "ablation", "longlag", "weatherlag", "tables", "review", "importance", "params"]
DEFAULT_STEPS = ["audit", "main", "ablation", "longlag", "tables", "importance", "params"]


def parse_h(h):
    return "DA" if str(h).upper() == "DA" else int(h)


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "default.json"))
    p.add_argument("--data", default=None, help="path to PGCB_Demand_With_Weather_Holidays.csv (default: data/)")
    p.add_argument("--steps", nargs="+", default=DEFAULT_STEPS, choices=ALL_STEPS)
    p.add_argument("--horizons", nargs="+", default=None, help="e.g. 24 168 720 DA")
    p.add_argument("--folds", nargs="+", default=None, help="e.g. 2025-26 2024-25 2023-24")
    p.add_argument("--models", nargs="+", default=None, help="restrict the main step to these model ids")
    p.add_argument("--out", default=None, help="output root (default from config: results/)")
    p.add_argument("--tuned", action="store_true", help="use the hyperparameters found by --steps tune")
    p.add_argument("--smoke", action="store_true", help="1 seed, 2 epochs, short val/test (pipeline check)")
    p.add_argument("--weather-lag", type=int, default=72, help="hours of weather delay for the weatherlag step")
    return p.parse_args()


def main():
    a = get_args()
    C = json.load(open(a.config))
    SEEDS = tuple(C["seeds"])
    FOLDS = C["folds"]
    MAIN_FOLD = C["main_fold"]
    HORIZONS = [parse_h(h) for h in (a.horizons or C["horizons"])]
    RUN_FOLDS = a.folds or list(FOLDS)
    OUT_ROOT = a.out or C["out_root"]
    if not os.path.isabs(OUT_ROOT):
        OUT_ROOT = os.path.join(ROOT, OUT_ROOT)
    if a.smoke:
        OUT_ROOT += "_smoke"
    EXTRA_HOL = C.get("extra_holidays_csv")
    SHED_CSV = C.get("load_shedding_csv")
    os.makedirs(OUT_ROOT, exist_ok=True)
    for f in RUN_FOLDS:
        assert f in FOLDS, f"unknown fold {f}; choose from {list(FOLDS)}"

    device = get_device()
    # the PGCB csv is needed only for steps that build features; "tables" works from saved predictions alone
    needs_data = any(s in a.steps for s in ("audit", "tune", "main", "ablation", "longlag", "weatherlag",
                                            "importance", "params"))
    raw = load_data(a.data) if needs_data else None
    _FS = {}

    def get_split(h, long_feats="auto", fold=MAIN_FOLD, verbose=False, weather_lag=0):
        cfg = Config(horizon=h, long_feats=long_feats, test_end=FOLDS[fold], seeds=SEEDS,
                     out_root=OUT_ROOT, smoke=a.smoke, weather_lag=weather_lag)
        key = (str(h), str(long_feats)) if not weather_lag else (str(h), str(long_feats), weather_lag)
        if key not in _FS:
            _FS[key] = FeatureSet(raw, cfg, EXTRA_HOL)
        return cfg, Split(_FS[key], cfg, device, verbose=verbose)

    def hp_of(mid, cfg):
        return tuned_hp(mid, replace(cfg, test_end=FOLDS[MAIN_FOLD])) if a.tuned else None

    def free():
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def nn_ok(mid):
        if not HAS_TORCH and mid not in OTHER_MODELS:
            print(f"  {mid}: skipped (PyTorch not installed)")
            return False
        return True

    print("device:", device)
    print(json.dumps(environment_info(), indent=1))
    print(f"output: {OUT_ROOT} | horizons {HORIZONS} | folds {RUN_FOLDS} | steps {a.steps}")

    # ------------------------------------------------------------------ Step 1: data audit
    if "audit" in a.steps:
        print("\n========== Step 1: data audit")
        AUD = f"{OUT_ROOT}/data_audit"
        os.makedirs(AUD, exist_ok=True)
        fold_windows = {}
        for name in FOLDS:
            _cfg, _sp = get_split(24, fold=name)
            fold_windows[name] = (_sp.dts.iloc[_sp.test_t[0]], _sp.dts.iloc[_sp.test_t[-1]])
        data_quality_report(raw, AUD, fold_windows, ratio_thr=Config().outage_ratio)

        fs24 = _FS[("24", "auto")]
        d = fs24.data.assign(year=fs24.data["datetime"].dt.year, date=fs24.data["datetime"].dt.normalize())
        raw_y = raw.assign(year=raw["datetime"].dt.year, date=raw["datetime"].dt.normalize())
        hol_tab = pd.DataFrame({
            "raw holiday days": raw_y[raw_y["Holiday_Category"] == 2].groupby("year")["date"].nunique(),
            "fixed holiday days": d[d["IsPublicHoliday"] == 1].groupby("year")["date"].nunique(),
            "Eid-break days": d[d["IsEidBreak"] == 1].groupby("year")["date"].nunique()}).fillna(0).astype(int)
        hol_tab.to_csv(f"{AUD}/holiday_counts_by_year.csv")
        print(hol_tab.to_string())
        for name, (s, e) in fold_windows.items():
            m = (d["datetime"] >= s) & (d["datetime"] <= e)
            raw_h = int(((raw["datetime"] >= s) & (raw["datetime"] <= e) & (raw["Holiday_Category"] == 2)).sum())
            print(f"test {name}: public-holiday hours raw {raw_h} -> fixed {int(d.loc[m, 'IsPublicHoliday'].sum())}, "
                  f"Eid-break hours {int(d.loc[m, 'IsEidBreak'].sum())}")
        if fs24.eid_table is not None:
            fs24.eid_table.to_csv(f"{AUD}/eid_dates_tabular_calendar.csv", index=False)
        grid = nasa_power_grid_check()
        grid.to_csv(f"{AUD}/station_coordinates_merra2_cells.csv", index=False)
        print(grid.to_string(index=False))

    # ------------------------------------------------------------------ Step 2: tuning (optional)
    if "tune" in a.steps:
        print("\n========== Step 2: equal tuning budget")
        for H in HORIZONS:
            cfg, sp = get_split(H)
            for mid in MAIN_IDS + ["K-MLP"]:
                if not nn_ok(mid):
                    continue
                best, _ = tune_model(mid, sp, cfg, n_trials=C.get("tuning_trials", 8))
                print(mid, H, best)
                free()

    # ------------------------------------------------------------------ Step 3: main comparison
    if "main" in a.steps:
        print("\n========== Step 3: main comparison")
        for fold in RUN_FOLDS:
            for H in HORIZONS:
                cfg, sp = get_split(H, fold=fold, verbose=(fold == RUN_FOLDS[0]))
                for mid in (a.models or MAIN_IDS + baselines_for(cfg)):
                    if not nn_ok(mid):
                        continue
                    print(f"--- {fold} | {setting_tag(cfg)} | {mid}")
                    run_model(mid, sp, cfg, hp_overrides=hp_of(mid, cfg))
                    free()

    # ------------------------------------------------------------------ Step 4: ablation
    if "ablation" in a.steps:
        print("\n========== Step 4: ablation")
        for H in [parse_h(h) for h in C.get("ablation_horizons", [24, 720])]:
            cfg, sp = get_split(H, fold=MAIN_FOLD)
            for mid in ABLATION_IDS:
                if not nn_ok(mid):
                    continue
                print(f"--- ablation | {setting_tag(cfg)} | {mid}")
                run_model(mid, sp, cfg, hp_overrides=hp_of(mid, cfg))
                free()

    # ------------------------------------------------------------------ Step 5: long-lag switch
    if "longlag" in a.steps:
        print("\n========== Step 5: long-lag feature switch")
        for H, lf in C.get("long_lag_switch", [[168, True], [720, False]]):
            cfg, sp = get_split(parse_h(H), long_feats=lf, fold=MAIN_FOLD)
            for mid in MAIN_IDS + ["XGBoost", "LightGBM", "Ridge", "K-MLP", "Persistence", "LastYearScaled"]:
                if not nn_ok(mid):
                    continue
                print(f"--- long-lag switch | {setting_tag(cfg)} | {mid}")
                run_model(mid, sp, cfg, hp_overrides=hp_of(mid, cfg))
                free()

    # ------------------------------------------------------------------ Step 5b: weather lag
    if "weatherlag" in a.steps:
        print(f"\n========== Step 5b: weather lagged {a.weather_lag} h (24 h and DA)")
        import shutil
        for fold in RUN_FOLDS:
            for H in [h for h in HORIZONS if h == 24 or h == "DA"]:
                cfg, sp = get_split(H, fold=fold, weather_lag=a.weather_lag)
                cfg0 = replace(cfg, weather_lag=0)
                ids = MAIN_IDS + ["XGBoost", "LightGBM", "Ridge"] + [b for b in baselines_for(cfg) if b in NO_WEATHER_IDS]
                if fold == MAIN_FOLD:
                    ids += ["M5+S", "M7-noStation"]
                for mid in (a.models or ids):
                    if not nn_ok(mid):
                        continue
                    src, dst = run_dir(cfg0, mid), run_dir(cfg, mid)
                    if mid in NO_WEATHER_IDS and os.path.exists(f"{src}/meta.json") and not os.path.exists(f"{dst}/meta.json"):
                        shutil.copytree(src, dst, dirs_exist_ok=True)        # no weather input: identical result
                        m = json.load(open(f"{dst}/meta.json"))
                        m["setting"], m["copied_from"] = setting_tag(cfg), setting_tag(cfg0)
                        json.dump(m, open(f"{dst}/meta.json", "w"), indent=2)
                        print(f"  {mid}: uses no weather, copied from {setting_tag(cfg0)}")
                        continue
                    print(f"--- {fold} | {setting_tag(cfg)} | {mid}")
                    run_model(mid, sp, cfg, hp_overrides=hp_of(mid, cfg))
                    free()

    # ------------------------------------------------------------------ Step 6: tables + figures
    main_tag = fold_tag(Config(test_end=FOLDS[MAIN_FOLD]))
    FIG = f"{OUT_ROOT}/figures"
    best = {}
    df = None
    if any(s in a.steps for s in ("tables", "importance")):
        df = collect_results(OUT_ROOT, n_boot=Config(smoke=a.smoke).n_boot)
        ens = df[(df["member"] == "ensemble") & (df["fold"] == main_tag)]
        for st in ["H24", "H168", "H720", "HDA"]:
            e = ens[(ens["setting"] == st) & (ens["family"] == "main")]
            if len(e):
                best[st] = e.sort_values("MAPE")["model"].iloc[0]

    if "tables" in a.steps:
        print("\n========== Step 6: evaluation tables and figures")
        tag2name = {fold_tag(Config(test_end=v)): k for k, v in FOLDS.items()}
        tables = evaluation_tables(df, OUT_ROOT, main_fold=main_tag)
        tables["B_rolling_origin_MAPE"] = tables["B_rolling_origin_MAPE"].rename(columns=tag2name)
        tables["B_rolling_origin_MAPE"].to_csv(f"{OUT_ROOT}/tables/B_rolling_origin_MAPE.csv", index=False)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 40)
        for k, t in tables.items():
            print(f"\n==================== {k}")
            print(t.round(3).to_string())

        os.makedirs(FIG, exist_ok=True)
        ens = df[(df["member"] == "ensemble") & (df["fold"] == main_tag)]
        for st in ["H24", "H168", "H720", "HDA"]:
            e = ens[ens["setting"] == st]
            if e.empty:
                continue
            sb = strongest_baseline(f"{OUT_ROOT}/{main_tag}/{st}", [m for m in BASELINE_IDS if m in set(e["model"])])
            bl = [sb] if sb else []                                   # strongest baseline on validation MAPE
            r = mcs_and_dm(OUT_ROOT, main_tag, st, models=MAIN_IDS + bl, tag="hybrids")   # heatmap only; paper MCS/DM = all models (Table A)
            if r:
                plot_dm_heatmap(r["pairwise_p"], f"Pairwise DM (APE loss, Holm), {st}", f"{FIG}/dm_pvalues_{st}.png")
                print(st, "MCS:\n", r["mcs"].round(3).to_string(index=False))
        print("best main model per setting:", best)
        if best:
            plot_monthly_mape(OUT_ROOT, main_tag, best, f"{FIG}/monthly_mape_best.png")
            for st, mid in best.items():
                plot_first_week(OUT_ROOT, main_tag, st, mid, f"{FIG}/first_week_{st}_{mid}.png")

    # ------------------------------------------------------------------ Step 6b: review tables
    if "review" in a.steps:
        print("\n========== Step 6b: review tables R1-R8")
        tag2name = {fold_tag(Config(test_end=v)): k for k, v in FOLDS.items()}
        rv = review_tables(OUT_ROOT, tag2name, main_fold=main_tag, n_boot=Config(smoke=a.smoke).n_boot)
        rv.update(review_tables_78(OUT_ROOT, tag2name, n_boot=Config(smoke=a.smoke).n_boot))
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 40)
        for k, t in rv.items():
            print(f"\n==================== {k}")
            print(t.round(3).to_string(index=False))
            if t.attrs.get("note"):
                print(t.attrs["note"])

    # ------------------------------------------------------------------ Step 7: importance, Eid, shedding
    if "importance" in a.steps and best:
        print("\n========== Step 7: grouped importance, Eid and load shedding")
        os.makedirs(f"{OUT_ROOT}/tables", exist_ok=True)
        os.makedirs(FIG, exist_ok=True)
        gi_all = []
        for st, mid in best.items():
            H = "DA" if st == "HDA" else int(st[1:])
            cfg, sp = get_split(H, fold=MAIN_FOLD)
            for m in [mid, "XGBoost"]:
                if os.path.exists(f"{run_dir(cfg, m)}/meta.json") and nn_ok(m):
                    gi_all.append(grouped_importance(m, sp, cfg).assign(setting=st))
                    free()
        if gi_all:
            gi_all = pd.concat(gi_all)
            gi_all.to_csv(f"{OUT_ROOT}/tables/G_grouped_importance.csv", index=False)
            print(gi_all.round(1).to_string(index=False))
            plot_grouped_importance(gi_all[gi_all["model"] != "XGBoost"], f"{FIG}/grouped_importance_best_models.png")

        get_split(24)
        eid_table = _FS[("24", "auto")].eid_table
        eid_rows, shed_rows = [], []
        for st, mid in best.items():
            te = pd.read_csv(f"{OUT_ROOT}/{main_tag}/{st}/{mid}/test_predictions.csv")
            eid_rows.append(eid_analysis(te, eid_table=eid_table).assign(setting=st, model=mid))
            r = load_shedding_analysis(te, SHED_CSV)
            if r:
                shed_rows.append(dict(setting=st, model=mid, **r))
        eid_tab = pd.concat(eid_rows)
        eid_tab.to_csv(f"{OUT_ROOT}/tables/H_eid_analysis.csv")
        print(eid_tab.round(2).to_string())
        if shed_rows:
            pd.DataFrame(shed_rows).to_csv(f"{OUT_ROOT}/tables/I_load_shedding.csv", index=False)
            print(pd.DataFrame(shed_rows).round(3).to_string(index=False))

    # ------------------------------------------------------------------ Step 8: params + reproducibility
    if "params" in a.steps:
        print("\n========== Step 8: parameter counts and reproducibility")
        os.makedirs(f"{OUT_ROOT}/tables", exist_ok=True)
        if HAS_TORCH:
            pc = []
            for H in [24, 168, 720, "DA"]:
                cfg, sp = get_split(H, fold=MAIN_FOLD)
                pc.append(param_count_table(sp, cfg, MAIN_IDS + ["M7-MLP", "M7-noLB", "M7-noStation",
                                                                 "M2+S", "M5+S", "K-MLP", "DLinear"]))
            pc = pd.concat(pc)
            if len(pc):
                pc = pc.pivot_table(index=["model", "name"], columns="setting", values="params (M)").round(3)
                pc.to_csv(f"{OUT_ROOT}/tables/J_parameter_counts.csv")
                print(pc.to_string())
        repro = {"environment": environment_info(), "seeds": list(SEEDS), "folds": FOLDS,
                 "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(Config()).items()},
                 "default_hyperparameters": DEFAULT_HP, "station_coordinates": STATION_COORDS,
                 "weather_source": "NASA POWER hourly, meteorology from MERRA-2 (0.5 deg x 0.625 deg)"}
        json.dump(repro, open(f"{OUT_ROOT}/reproducibility.json", "w"), indent=2, default=str)
        print(json.dumps(repro["environment"], indent=1))

    print(f"\nDone. Everything is in {OUT_ROOT}")


if __name__ == "__main__":
    main()
