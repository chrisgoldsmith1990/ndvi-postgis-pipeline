"""Validate the crop-type curve-shape classifier against real ground truth
-- USDA's Cropland Data Layer (CDL) -- instead of just proxy metrics
(silhouette score, KMeans assignment confidence, acreage plausibility).

Why a *past* season, not this one: CDL for the current growing season
doesn't exist yet (it's produced after harvest), and comparing this
season's classification against the most recently *published* CDL (whichever
year that is) is confounded by crop rotation -- corn/soybean farming
typically alternates annually, so a field's most likely current-year crop
is the *opposite* of last year's label, not a match. Comparing against a
year's CDL using that *same* year's imagery sidesteps the problem entirely:
genuine same-season ground truth, no rotation ambiguity.

Why capped at a day-of-year cutoff: the validation season's imagery is
capped at the same day-of-year the *current* season's data actually
reaches (see CURRENT_SEASON_MAX_DOY below) -- so this never validates using
information we wouldn't actually have "in real time" partway through a
live season, which would be an unrealistically easy test.

Why 2021: Planetary Computer's `usda-cdl` mirror's own declared temporal
extent is 2008-01-01 to 2021-12-31 (checked directly against the
collection's metadata, not assumed) -- USDA/GMU's own CropScape REST API
has a genuinely expired SSL certificate as of this writing, and USDA's
direct-download portal blocks scripted requests with a real 403. 2021 is
simply the most recent year actually available through a free, keyless
source. This doesn't undermine the validation: corn/soybean growth
*physics* don't change year to year, only which specific field grows which
crop (driven by rotation) -- what's being validated here is the
classification *method*, which should generalize.

Result (subset area, 161 CDL-labeled Corn/Soybean parcels, 5-fold
cross-validated): a plain peak_doy threshold sweep picked 186 in *every*
fold with zero variance. NDVI scored 80.1% (+/- 7.3%) at that threshold;
EVI2 scored 87.6% (+/- 5.1%) -- a real, ground-truth-validated result
(not just a proxy-metric preference), which is why EVI2 + threshold=186
are now crop_clusters.py's defaults. A 5-feature logistic regression
*underperformed* the single-feature threshold rule for both indices,
confirming the simple rule is the right amount of complexity here, not an
oversimplification.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.cdl_ground_truth import load_cdl_ground_truth
from src.compute_evi2 import compute_evi2
from src.compute_ndvi import compute_ndvi
from src.crop_clusters import extract_features, fit_splines, load_series
from src.fetch_cdl import fetch_cdl_raster
from src.fetch_hls import build_date as hls_build_date
from src.fetch_hls import find_season_items as hls_find_season_items
from src.fetch_timeseries import MAX_BAD_FRACTION, SUBSET_BBOX
from src.fetch_timeseries import RAW_DIR as S2_RAW_DIR
from src.fetch_timeseries import build_date as s2_build_date
from src.fetch_timeseries import find_season_items as s2_find_season_items
from src.zonal_stats import compute_zonal_stats, load_zonal_stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURE_COLS = ["early_ndvi", "peak_ndvi", "peak_doy", "green_up_rate", "decline_rate"]

VALIDATION_YEAR = 2021
# Matches this season's actual max day-of-year reach (2026-09-03, DOY 246
# -- checked directly against ndvi_zonal_stats_subset_clipped) so the
# validation season never has more late-season information than we'd
# actually have partway through a live season.
CURRENT_SEASON_MAX_DOY = 246


def fetch_validation_imagery(year=VALIDATION_YEAR, max_doy=CURRENT_SEASON_MAX_DOY, bbox=SUBSET_BBOX):
    """Sentinel-2 + HLS imagery for the validation season, capped at
    max_doy -- reuses fetch_timeseries.py/fetch_hls.py's existing
    functions unmodified (bbox/datetime_range/raw_dir were already
    parameters)."""
    end_date = (pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=max_doy - 1)).strftime("%Y-%m-%d")
    date_range = f"{year}-04-01/{end_date}"

    written = []
    by_date = s2_find_season_items(bbox=bbox, datetime_range=date_range)
    for date, items in by_date.items():
        if s2_build_date(items, date, bbox=bbox, raw_dir=S2_RAW_DIR, max_bad_fraction=MAX_BAD_FRACTION) is not None:
            written.append(str(date))
    print(f"Sentinel-2 {year}: wrote {len(written)}/{len(by_date)} dates", flush=True)

    written_hls = []
    by_date_hls = hls_find_season_items(bbox=bbox, datetime_range=date_range)
    for date, items in by_date_hls.items():
        if hls_build_date(items, date, bbox=bbox, raw_dir=S2_RAW_DIR, max_bad_fraction=MAX_BAD_FRACTION) is not None:
            written_hls.append(str(date))
    print(f"HLS {year}: wrote {len(written_hls)}/{len(by_date_hls)} dates", flush=True)
    return written + written_hls


def process_validation_imagery(year=VALIDATION_YEAR, parcels_table="parcels_clipped"):
    """NDVI + EVI2 zonal stats for every fetched validation-season date,
    into their own _{year} tables."""
    ndvi_table = f"ndvi_zonal_stats_subset_{year}"
    evi2_table = f"evi2_zonal_stats_subset_{year}"
    raw_dirs = sorted(d for d in (DATA_DIR / "raw").iterdir() if d.name.startswith(f"{year}-"))
    for scene_dir in raw_dirs:
        date = scene_dir.name
        red, nir = scene_dir / "red.tif", scene_dir / "nir.tif"
        if not (red.exists() and nir.exists()):
            continue
        ndvi_path = DATA_DIR / "processed" / date / "ndvi.tif"
        compute_ndvi(red, nir, ndvi_path)
        load_zonal_stats(compute_zonal_stats(ndvi_path, parcels_table, prefix="ndvi"), date, table_name=ndvi_table)

        evi2_path = DATA_DIR / "processed_evi2" / date / "evi2.tif"
        compute_evi2(red, nir, evi2_path)
        load_zonal_stats(compute_zonal_stats(evi2_path, parcels_table, prefix="evi2"), date, table_name=evi2_table)
    return ndvi_table, evi2_table


def _build_feats(table_name, value_column):
    df = load_series(table_name=table_name, bad_dates=frozenset(), value_column=value_column)
    splines, kept_series = fit_splines(df)
    return extract_features(splines, kept_series)


def _best_threshold(feats, labels, thresholds):
    best_thr, best_acc = None, -1
    for thr in thresholds:
        acc = (np.where(feats["peak_doy"] < thr, "Corn", "Soybean") == labels.values).mean()
        if acc > best_acc:
            best_thr, best_acc = thr, acc
    return best_thr, best_acc


def _percentile_of(series, value):
    return (series < value).mean() * 100


def validate(ndvi_table, evi2_table, bbox=SUBSET_BBOX, year=VALIDATION_YEAR,
             thresholds=range(170, 250, 2)):
    """Cross-validated peak_doy threshold tuning against real CDL labels,
    for both NDVI and EVI2 -- prints a full report and returns the winning
    (index_name, threshold, cv_accuracy) for whichever index validates
    best."""
    gt = load_cdl_ground_truth(fetch_cdl_raster(bbox, str(year))).set_index("pin")

    results = {}
    for index_name, table, value_col in [("NDVI", ndvi_table, "ndvi_mean"), ("EVI2", evi2_table, "evi2_mean")]:
        feats = _build_feats(table, value_col)
        joined = feats.join(gt, how="inner")
        row_crop = joined[joined["cdl_label"].isin(["Corn", "Soybean"])].copy()
        print(f"\n{index_name}: {len(row_crop)} CDL-labeled Corn/Soybean parcels "
              f"({len(joined) - len(row_crop)} Other)", flush=True)

        full_thr, full_acc = _best_threshold(row_crop, row_crop["cdl_label"], thresholds)
        full_pct = _percentile_of(row_crop["peak_doy"], full_thr)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        X = row_crop.reset_index(drop=True)
        fold_thr, fold_acc, fold_pct, fold_pct_acc = [], [], [], []
        for train_idx, test_idx in cv.split(X, X["cdl_label"]):
            train, test = X.iloc[train_idx], X.iloc[test_idx]
            thr, _ = _best_threshold(train, train["cdl_label"], thresholds)
            acc = (np.where(test["peak_doy"] < thr, "Corn", "Soybean") == test["cdl_label"].values).mean()
            fold_thr.append(thr)
            fold_acc.append(acc)

            # Percentile version: the actually-deployed rule
            # (crop_clusters.cluster()) uses a percentile of *that season's
            # own* row-crop population, not a fixed absolute day (an
            # absolute day-of-year threshold, even cross-validated here,
            # turned out not to transfer to a different season's different
            # cloud-driven observation calendar -- see
            # crop_clusters.CORN_SOYBEAN_PEAK_DOY_PERCENTILE's comment for
            # the full story). Simulated here as: derive the percentile
            # from the train fold's own threshold, then apply that
            # percentile to the *test* fold's own distribution -- the same
            # operation as applying a percentile tuned on last season to
            # this season's own distribution.
            pct = _percentile_of(train["peak_doy"], thr)
            test_thr_from_pct = np.percentile(test["peak_doy"], pct)
            pct_acc = (np.where(test["peak_doy"] < test_thr_from_pct, "Corn", "Soybean")
                       == test["cdl_label"].values).mean()
            fold_pct.append(pct)
            fold_pct_acc.append(pct_acc)

        print(f"{index_name}: full-data best threshold={full_thr} "
              f"({full_pct:.1f}th percentile, in-sample acc={full_acc:.3f})", flush=True)
        print(f"{index_name}: 5-fold CV (absolute day): thresholds={fold_thr}, "
              f"test accuracy={np.mean(fold_acc):.3f} +/- {np.std(fold_acc):.3f}", flush=True)
        print(f"{index_name}: 5-fold CV (percentile, transferred fold-to-fold): "
              f"percentiles={[round(p, 1) for p in fold_pct]}, "
              f"test accuracy={np.mean(fold_pct_acc):.3f} +/- {np.std(fold_pct_acc):.3f}", flush=True)

        clf_scores = cross_val_score(LogisticRegression(max_iter=1000),
                                      row_crop[FEATURE_COLS].values,
                                      (row_crop["cdl_label"] == "Corn").values.astype(int),
                                      cv=cv, scoring="accuracy")
        print(f"{index_name}: 5-feature logistic regression CV accuracy: "
              f"{clf_scores.mean():.3f} +/- {clf_scores.std():.3f} (for comparison)", flush=True)

        results[index_name] = {"threshold": full_thr, "percentile": full_pct,
                                "cv_accuracy": np.mean(fold_acc), "cv_pct_accuracy": np.mean(fold_pct_acc)}

    winner = max(results, key=lambda k: results[k]["cv_accuracy"])
    print(f"\nWinner: {winner}, threshold={results[winner]['threshold']} "
          f"({results[winner]['percentile']:.1f}th percentile), "
          f"cv_accuracy={results[winner]['cv_accuracy']:.3f}, "
          f"cv_percentile_accuracy={results[winner]['cv_pct_accuracy']:.3f}", flush=True)
    return winner, results[winner]["percentile"], results[winner]["cv_accuracy"]


if __name__ == "__main__":
    fetch_validation_imagery()
    ndvi_table, evi2_table = process_validation_imagery()
    validate(ndvi_table, evi2_table)
