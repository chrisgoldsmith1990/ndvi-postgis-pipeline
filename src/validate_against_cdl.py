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

This went through four real iterations, each one found by actually
deploying the previous version's result, not by trusting a validation
number in isolation -- see crop_clusters.CORN_SOYBEAN_COEF's comment for
the full story:

- v1: subset-only (163 parcels), absolute peak_doy threshold. 87.6%
  cross-validated (EVI2). Deployed, then found not to transfer across
  seasons (2026 identified 2 of 150 as corn, not 40-60).
- v2: subset-only, percentile of peak_doy instead of an absolute day.
  Fixed the season-transfer problem, but the subset's own true CDL corn
  fraction (41.6%) turned out not to represent the *county's* real acreage
  split (~52%) -- a small rural cluster chosen for fetch cost, not
  representativeness. Deployed at county scale: 31.7% corn, far off.
- v3: county-wide (8,202 CDL-labeled parcels), multi-feature logistic
  regression (peak_doy alone only reached 63.8% at this scale -- the
  subset's 87.6% was never representative). 80.5% cross-validated. Fit on
  raw feature values; deployed to 2026 it produced only 37.6% corn.
- v4 (current): same model, refit on features standardized against each
  season's *own* row-crop population instead of raw values -- found a
  real season-to-season shift (2026's peak_ndvi ran ~0.5 std higher than
  2021's, biasing every prediction toward soybean). Standardizing
  preserved 2021 accuracy (80.65%) and moved the 2026 deployment result to
  56.1% corn, much closer to the real ~52%.
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
from src.fetch_cdl import fetch_cdl_mosaic, fetch_cdl_raster
from src.fetch_hls import build_date as hls_build_date
from src.fetch_hls import find_season_items as hls_find_season_items
from src.fetch_timeseries import COUNTY_BBOX, MAX_BAD_FRACTION, SUBSET_BBOX
from src.fetch_timeseries import RAW_DIR as BASE_RAW_DIR
from src.fetch_timeseries import build_date as s2_build_date
from src.fetch_timeseries import find_season_items as s2_find_season_items
from src.zonal_stats import compute_zonal_stats, load_zonal_stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURE_COLS = ["early_ndvi", "peak_ndvi", "peak_doy", "green_up_rate", "decline_rate"]

VALIDATION_YEAR = 2021
# Matches each season's actual max day-of-year reach (checked directly:
# 2026-09-03/DOY 246 for the subset, 2026-09-06/DOY 249 for the county) so
# the validation season never has more late-season information than we'd
# actually have partway through a live season.
CURRENT_SEASON_MAX_DOY = {"subset": 246, "county": 249}


def fetch_validation_imagery(scope="subset", year=VALIDATION_YEAR):
    """Sentinel-2 + HLS imagery for the validation season, capped at that
    scope's max day-of-year -- reuses fetch_timeseries.py/fetch_hls.py's
    existing functions unmodified (bbox/datetime_range/raw_dir were
    already parameters)."""
    bbox = SUBSET_BBOX if scope == "subset" else COUNTY_BBOX
    raw_dir = BASE_RAW_DIR if scope == "subset" else BASE_RAW_DIR / f"county_{year}"
    max_doy = CURRENT_SEASON_MAX_DOY[scope]
    end_date = (pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=max_doy - 1)).strftime("%Y-%m-%d")
    date_range = f"{year}-04-01/{end_date}"

    written = []
    by_date = s2_find_season_items(bbox=bbox, datetime_range=date_range)
    for date, items in by_date.items():
        if s2_build_date(items, date, bbox=bbox, raw_dir=raw_dir, max_bad_fraction=MAX_BAD_FRACTION) is not None:
            written.append(str(date))
    print(f"Sentinel-2 {year} ({scope}): wrote {len(written)}/{len(by_date)} dates", flush=True)

    written_hls = []
    by_date_hls = hls_find_season_items(bbox=bbox, datetime_range=date_range)
    for date, items in by_date_hls.items():
        if hls_build_date(items, date, bbox=bbox, raw_dir=raw_dir, max_bad_fraction=MAX_BAD_FRACTION) is not None:
            written_hls.append(str(date))
    print(f"HLS {year} ({scope}): wrote {len(written_hls)}/{len(by_date_hls)} dates", flush=True)
    return written + written_hls


def process_validation_imagery(scope="subset", year=VALIDATION_YEAR):
    """NDVI + EVI2 zonal stats for every fetched validation-season date,
    into their own tables."""
    raw_dir = BASE_RAW_DIR if scope == "subset" else BASE_RAW_DIR / f"county_{year}"
    parcels_table = "parcels_clipped" if scope == "subset" else "parcels_clipped_county"
    ndvi_table = f"ndvi_zonal_stats_{scope}_{year}"
    evi2_table = f"evi2_zonal_stats_{scope}_{year}"

    raw_dirs = sorted(d for d in raw_dir.iterdir() if d.name.startswith(f"{year}-"))
    for scene_dir in raw_dirs:
        date = scene_dir.name
        red, nir = scene_dir / "red.tif", scene_dir / "nir.tif"
        if not (red.exists() and nir.exists()):
            continue
        ndvi_path = DATA_DIR / "processed" / f"{scope}_{year}" / date / "ndvi.tif"
        compute_ndvi(red, nir, ndvi_path)
        load_zonal_stats(compute_zonal_stats(ndvi_path, parcels_table, prefix="ndvi"), date, table_name=ndvi_table)

        evi2_path = DATA_DIR / "processed_evi2" / f"{scope}_{year}" / date / "evi2.tif"
        compute_evi2(red, nir, evi2_path)
        load_zonal_stats(compute_zonal_stats(evi2_path, parcels_table, prefix="evi2"), date, table_name=evi2_table)
    return ndvi_table, evi2_table


def fetch_ground_truth(scope="subset", year=VALIDATION_YEAR):
    """CDL ground truth for the given scope -- a single-tile fetch for the
    subset, a 2-tile mosaic for the county (McLean County straddles a CDL
    tile boundary -- checked directly, not assumed)."""
    bbox = SUBSET_BBOX if scope == "subset" else COUNTY_BBOX
    parcels_table = "parcels_clipped" if scope == "subset" else "parcels_clipped_county"
    cdl_path = fetch_cdl_raster(bbox, str(year)) if scope == "subset" else fetch_cdl_mosaic(bbox, str(year))
    return load_cdl_ground_truth(cdl_path, parcels_table=parcels_table)


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


def validate_threshold(ndvi_table, evi2_table, gt, thresholds=range(150, 260, 2)):
    """v1/v2: cross-validated single-feature peak_doy threshold tuning
    against real CDL labels, for both NDVI and EVI2. Prints a full report;
    kept for the historical record (this is what first showed EVI2 beats
    NDVI, and what first showed the transfer/representativeness problems
    that motivated v3/v4) -- see validate_multi_feature for the model
    actually deployed."""
    gt = gt.set_index("pin") if gt.index.name != "pin" else gt
    results = {}
    for index_name, table, value_col in [("NDVI", ndvi_table, "ndvi_mean"), ("EVI2", evi2_table, "evi2_mean")]:
        feats = _build_feats(table, value_col)
        joined = feats.join(gt, how="inner")
        row_crop = joined[joined["cdl_label"].isin(["Corn", "Soybean"])].copy()
        print(f"\n{index_name}: {len(row_crop)} CDL-labeled Corn/Soybean parcels "
              f"({len(joined) - len(row_crop)} Other), true corn fraction "
              f"{(row_crop['cdl_label']=='Corn').mean():.3f}", flush=True)

        full_thr, full_acc = _best_threshold(row_crop, row_crop["cdl_label"], thresholds)
        full_pct = _percentile_of(row_crop["peak_doy"], full_thr)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        X = row_crop.reset_index(drop=True)
        fold_thr, fold_acc = [], []
        for train_idx, test_idx in cv.split(X, X["cdl_label"]):
            train, test = X.iloc[train_idx], X.iloc[test_idx]
            thr, _ = _best_threshold(train, train["cdl_label"], thresholds)
            acc = (np.where(test["peak_doy"] < thr, "Corn", "Soybean") == test["cdl_label"].values).mean()
            fold_thr.append(thr)
            fold_acc.append(acc)

        print(f"{index_name}: full-data best threshold={full_thr} "
              f"({full_pct:.1f}th percentile, in-sample acc={full_acc:.3f})", flush=True)
        print(f"{index_name}: 5-fold CV: thresholds={fold_thr}, "
              f"test accuracy={np.mean(fold_acc):.3f} +/- {np.std(fold_acc):.3f}", flush=True)
        results[index_name] = {"threshold": full_thr, "percentile": full_pct, "cv_accuracy": np.mean(fold_acc)}

    winner = max(results, key=lambda k: results[k]["cv_accuracy"])
    print(f"\nWinner: {winner}, threshold={results[winner]['threshold']} "
          f"({results[winner]['percentile']:.1f}th percentile), "
          f"cv_accuracy={results[winner]['cv_accuracy']:.3f}", flush=True)
    return results


def validate_multi_feature(evi2_table, gt, standardize=True):
    """v3/v4: multi-feature logistic regression against real CDL labels --
    the model actually deployed in crop_clusters.CORN_SOYBEAN_COEF.
    standardize=True fits on each feature z-scored against the row-crop
    population's own mean/std (v4); False fits on raw values (v3, kept for
    comparison -- this is what was found not to transfer to a new season).
    Prints a full report and returns the fitted (coef, intercept), fit on
    *all* labeled data (cross-validation is for honestly estimating
    accuracy, not for the deployed coefficients)."""
    gt = gt.set_index("pin") if gt.index.name != "pin" else gt
    feats = _build_feats(evi2_table, "evi2_mean")
    joined = feats.join(gt, how="inner")
    row_crop = joined[joined["cdl_label"].isin(["Corn", "Soybean"])].copy()
    print(f"{len(row_crop)} CDL-labeled Corn/Soybean parcels, true corn fraction "
          f"{(row_crop['cdl_label']=='Corn').mean():.3f}", flush=True)

    if standardize:
        row_crop_all = feats[feats["early_ndvi"] <= 0.19]  # NON_ROW_CROP_EARLY_NDVI
        mean, std = row_crop_all[FEATURE_COLS].mean(), row_crop_all[FEATURE_COLS].std()
        X = ((row_crop[FEATURE_COLS] - mean) / std).values
    else:
        X = row_crop[FEATURE_COLS].values
    y = (row_crop["cdl_label"] == "Corn").values.astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=cv, scoring="accuracy")
    print(f"{'Standardized' if standardize else 'Raw'}-feature CV accuracy: "
          f"{scores.mean():.4f} +/- {scores.std():.4f}", flush=True)

    clf = LogisticRegression(max_iter=2000).fit(X, y)
    print("Coefficients:", dict(zip(FEATURE_COLS, clf.coef_[0].round(4))), flush=True)
    print("Intercept:", round(clf.intercept_[0], 4), flush=True)
    print(f"Predicted corn fraction (training data): {clf.predict(X).mean():.4f} "
          f"(true: {y.mean():.4f})", flush=True)
    return clf.coef_[0], clf.intercept_[0]


if __name__ == "__main__":
    import sys

    scope = sys.argv[1] if len(sys.argv) > 1 else "subset"
    fetch_validation_imagery(scope=scope)
    ndvi_table, evi2_table = process_validation_imagery(scope=scope)
    gt = fetch_ground_truth(scope=scope)

    if scope == "subset":
        validate_threshold(ndvi_table, evi2_table, gt)
    else:
        validate_multi_feature(evi2_table, gt, standardize=False)
        print()
        validate_multi_feature(evi2_table, gt, standardize=True)
