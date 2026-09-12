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

## Literature-calibrated absolute estimate (approximate, clearly labeled)

Real published models exist for exactly this NDVI-integral-to-yield
relationship, so a rough absolute number is possible -- with real caveats
attached, not a made-up conversion factor:

- Johnson et al. 2021, "USA Crop Yield Estimation with MODIS NDVI: Are
  Remotely Sensed Models Better Than Simple Trend Analyses?", Remote
  Sensing 13(21):4227 (USDA NASS + NASA GSFC, open access, CC BY 4.0).
  Their "accumulated NDVI" method -- summing NDVI above an optimized
  threshold across the season, the same idea as seasonal_ndvi_integral
  here -- gets R2=0.91, SE=7.7 bu/ac for Illinois corn (state level) and
  R2=0.54, SE=4.8 bu/ac for Illinois soybean. Soybean is explicitly
  weaker: the paper notes soybean's accumulated-NDVI model was "only
  marginally better than using trend alone" nationally. Most of our
  subset (78/125 parcels) is soybean-like -- this estimate should be
  trusted less for that group.
- Xu & Katchova 2019, Journal of Agricultural and Applied Economics
  51(3):402-416: a 10% July NDVI increase corresponds to a 4.5% (1.94
  bu/ac) soybean yield increase nationally.

What's missing to apply either model exactly: the actual fitted
slope/intercept (the papers report R2/SE, not the regression equation
itself), and calibration specific to this subset rather than
Illinois/national aggregates. The approximation used here: anchor each
cluster's mean to McLean County's actual 2025 NASS yield (243.1 bu/ac
corn, 73.95 bu/ac soybean -- computed from published county
production/acreage; 2026's county yield won't be published until after
this harvest, same reporting lag as the CDL discussion elsewhere in this
project), then scale each parcel's deviation from its cluster mean using
the literature model's own coefficient of variation (CV = SE/mean, which
transfers across different yield baselines better than raw SE) as a
stand-in for "how much yield spread this method typically explains."
This is explicitly a rough approximation, not a validated per-field
prediction -- it borrows a real, cited relationship's *spread*, not its
exact fitted equation.

## Per-acre rate isn't the estimate -- total bushels is

A bu/ac rate alone doesn't say what a field actually produces: parcels in
this subset range from about a dozen to well over a hundred acres. The
number that matters is bu/ac x the parcel's own acreage (`computed_ac`
from the parcels table, the same acreage field load_boundaries.py loads
from the county's parcel layer).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.crop_clusters import cluster, extract_features, fit_splines, label_cluster, load_series
from src.db import get_engine

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# 2025 McLean County actual NASS yields (source: farmdoc daily / USDA-NASS
# county estimates). Corn: 77.31M bu / 318,000 harvested acres. Soybean:
# 21.742M bu / 294,000 harvested acres (McLean led Illinois in soybean
# production that year).
COUNTY_YIELD_ANCHOR_BU_AC = {
    "Corn-like (early peak, fast decline)": 77_310_000 / 318_000,
    "Soybean-like (later peak, slower decline)": 21_742_000 / 294_000,
}
# Illinois state-level accumulated-NDVI model CV (SE/mean) from Johnson et
# al. 2021, Table 1 -- used as a spread proxy, not a fitted slope.
LITERATURE_CV = {
    "Corn-like (early peak, fast decline)": 0.045,
    "Soybean-like (later peak, slower decline)": 0.095,
}


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


def estimate_yield_bu_ac(feats):
    """Approximate bu/ac per parcel: county-actual anchor for its crop-type
    cluster, scaled by its within-cluster z-score times the literature
    model's CV. Returns NaN for the non-row-crop cluster, which has no
    corresponding crop-yield literature or NASS anchor to use."""
    feats = feats.copy()
    z = feats.groupby("cluster")["seasonal_ndvi_integral"].transform(lambda s: (s - s.mean()) / s.std())
    anchor = feats["cluster_label"].map(COUNTY_YIELD_ANCHOR_BU_AC)
    cv = feats["cluster_label"].map(LITERATURE_CV)
    feats["estimated_yield_bu_ac"] = anchor * (1 + cv * z)
    return feats


def fetch_acreage(pins):
    """computed_ac per parcel from the parcels table -- the same acreage
    field load_boundaries.py loads from the county's parcel layer."""
    engine = get_engine()
    query = text("SELECT pin, computed_ac FROM parcels WHERE pin = ANY(:pins)")
    df = pd.read_sql(query, engine, params={"pins": list(pins)})
    return df.set_index("pin")["computed_ac"]


def estimate_total_bushels(feats, acreage):
    """The actual estimate: bu/ac x the parcel's own acreage. A rate alone
    doesn't say what a field produces -- these parcels range from a dozen
    to well over a hundred acres."""
    feats = feats.copy()
    feats["acres"] = feats.index.map(acreage)
    feats["estimated_total_bushels"] = feats["estimated_yield_bu_ac"] * feats["acres"]
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
    feats = estimate_yield_bu_ac(feats)

    acreage = fetch_acreage(feats.index)
    feats = estimate_total_bushels(feats, acreage)

    print("Season-integrated NDVI stats by cluster:")
    print(feats.groupby("cluster")["seasonal_ndvi_integral"].describe()[["count", "mean", "std", "min", "max"]].round(1))

    print("\nEstimated total bushels by cluster (approximate -- see module docstring):")
    print(feats.groupby("cluster_label")["estimated_total_bushels"].describe()[["count", "mean", "std", "min", "max"]].round(0))

    print("\nTop 5 relative performers per cluster:")
    for cid, group in feats.groupby("cluster"):
        top = group.sort_values("percentile_in_cluster", ascending=False).head(5)
        print(f"\nCluster {cid} (n={len(group)}):")
        print(top[["seasonal_ndvi_integral", "percentile_in_cluster", "acres",
                    "estimated_yield_bu_ac", "estimated_total_bushels"]].round(1))

    out = feats[["cluster", "cluster_label", "seasonal_ndvi_integral", "percentile_in_cluster",
                 "acres", "estimated_yield_bu_ac", "estimated_total_bushels", "confidence"]]
    out.to_csv(REPORTS_DIR / "yield_ranking.csv")
    print(f"\nWrote {REPORTS_DIR / 'yield_ranking.csv'}", flush=True)
