"""Relative yield-potential ranking within each crop-type cluster.

Season-integrated NDVI (area under the fitted curve across the observed
date range) is the standard remote-sensing proxy for accumulated
photosynthetic biomass, which is what drives grain fill -- this is the
basis of real NDVI-based yield forecasting research. But two things stop
this from producing an actual bushels/acre number:

1. NDVI saturates once canopy closes. Two fields both pegged near 0.90 at
   peak can have genuinely different yields -- exactly the range that
   matters most -- and look nearly identical from NDVI alone.
2. No ground truth. There's no yield-monitor or per-field harvest data for
   this subset to calibrate a conversion factor against, so any
   NDVI-to-bushels formula applied here would be invented, not measured.

What IS defensible: a *relative* ranking within a crop type. Comparing a
corn-like field's season-integrated NDVI only against other corn-like
fields (never against soybean-like fields, which run on a different NDVI
scale entirely) says "this field accumulated more/less seasonal
greenness than its crop-type peers" -- the same neighbor-relative logic
neighbor_comparison.py already uses for anomaly detection, just applied
within a crop-type peer group instead of a spatial one.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.crop_clusters import cluster, extract_features, fit_splines, load_series
from src.visualize_subset import label_cluster

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def seasonal_ndvi_integral(splines, doy):
    """Area under each parcel's fitted curve across the observed date
    range (trapezoidal integration of a fine grid) -- the season-long
    biomass-accumulation proxy, not a single date's value."""
    dense_doy = np.arange(int(doy.min()), int(doy.max()) + 1)
    return {pin: np.trapezoid(cs(dense_doy), dense_doy) for pin, cs in splines.items()}


def rank_within_cluster(feats):
    feats = feats.copy()
    feats["percentile_in_cluster"] = feats.groupby("cluster")["seasonal_ndvi_integral"].rank(pct=True) * 100
    return feats


if __name__ == "__main__":
    df = load_series()
    splines, doy, pivot = fit_splines(df)
    feats = extract_features(splines, doy, pivot)
    feats, best_k = cluster(feats)

    integrals = seasonal_ndvi_integral(splines, doy)
    feats["seasonal_ndvi_integral"] = feats.index.map(integrals)
    feats = rank_within_cluster(feats)

    cluster_means = feats.groupby("cluster").mean(numeric_only=True)
    label_by_id = {cid: label_cluster(row) for cid, row in cluster_means.iterrows()}
    feats["cluster_label"] = feats["cluster"].map(label_by_id)

    print("Season-integrated NDVI stats by cluster:")
    print(feats.groupby("cluster")["seasonal_ndvi_integral"].describe()[["count", "mean", "std", "min", "max"]].round(1))

    print("\nTop 5 relative performers per cluster:")
    for cid, group in feats.groupby("cluster"):
        top = group.sort_values("percentile_in_cluster", ascending=False).head(5)
        print(f"\nCluster {cid} (n={len(group)}):")
        print(top[["seasonal_ndvi_integral", "percentile_in_cluster"]].round(1))

    out = feats[["cluster", "cluster_label", "seasonal_ndvi_integral", "percentile_in_cluster", "confidence"]]
    out.to_csv(REPORTS_DIR / "yield_ranking.csv")
    print(f"\nWrote {REPORTS_DIR / 'yield_ranking.csv'}", flush=True)
