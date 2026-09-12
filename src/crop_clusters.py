"""Crop-type work, unsupervised: cluster parcels by NDVI curve *shape*
rather than a single date's value -- the whole premise from the start of
this thread is that corn and soybean (and other land uses) differ in when
and how fast they green up and senesce, not in what NDVI they happen to
hit on any one day.

No ground truth exists for this subset, so clusters are reported as
behavioral groups (A/B/C), not labeled "corn"/"soybean" -- the peak-timing
split does match the literature direction (corn peaks earlier than
soybean), but that's a prior, not a validated label. Labeling is a
separate, later decision (e.g. cross-referencing CDL, itself lagged a
full year -- see README).

Only 8 dates survived cloud/shadow filtering out of 41 candidate Sentinel-2
passes this season (confirmed by re-running the search with no whole-tile
cloud pre-filter at all -- the other 33 are genuinely too cloudy over this
specific AOI, not an artifact of an overly strict filter). That's a real,
irregularly-spaced 8 points per parcel (a 45-day gap between May 9 and
June 23 is the worst of it, then two closely-spaced points right at the
end -- Sept 1 and Sept 3), which makes naive two-point differences a poor
way to estimate a "rate", so each parcel's series is fit with a smooth
curve instead and features are read off that curve: interpolated peak
timing/height, and the curve's own derivative for green-up/decline rates.

A plain natural cubic spline (scipy.interpolate.CubicSpline) was the first
attempt, and it visibly overshoots on this data -- some parcels' fitted
curves swung above NDVI 1.2, which is physically impossible, because two
closely-spaced points (Sept 1, Sept 3) right after a gap is exactly the
shape that makes an unconstrained cubic spline ring between points. Caught
by plotting the full 163-parcel population, not the smaller sanity-check
sample, which happened not to include an affected parcel. Fixed by
switching to scipy.interpolate.PchipInterpolator -- a shape-preserving
cubic Hermite spline built specifically to not overshoot the data's own
range, at the cost of continuous second derivative (C1 instead of C2),
which is the right trade for bounded physical data like NDVI. The spline
is only evaluated within the observed date range either way -- no
extrapolation past April 9 or September 3.

Features:
- early_ndvi: raw April baseline -- separates anything already green in
  early spring (winter cover, pasture) from bare tilled soil.
- peak_ndvi, peak_doy: height and day-of-year of the spline's maximum.
- green_up_rate: the spline derivative's maximum (steepest rise, wherever
  in the season it actually falls, not fixed to one date pair).
- decline_rate: the spline derivative's minimum after the peak (steepest
  fall found so far in the observed window -- for parcels still rising at
  the last measured date, this reports the smallest available late-season
  slope rather than a fabricated decline).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from src.db import get_engine

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def load_series():
    engine = get_engine()
    df = pd.read_sql(text("""
        SELECT pin, date, ndvi_mean FROM ndvi_zonal_stats_subset
        WHERE ndvi_mean IS NOT NULL ORDER BY pin, date
    """), engine)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fit_splines(df):
    """One shape-preserving cubic (PCHIP) fit per parcel, over day-of-year.
    Returns the spline dict and the shared doy array (all parcels share the
    same 8 observed dates, so one grid works for all)."""
    pivot = df.pivot(index="pin", columns="date", values="ndvi_mean").dropna()
    dates = sorted(pivot.columns)
    doy = np.array([d.dayofyear for d in dates])

    splines = {
        pin: PchipInterpolator(doy, row.values.astype(float))
        for pin, row in pivot.iterrows()
    }
    return splines, doy, pivot


def extract_features(splines, doy, pivot, oversample_days=1):
    dense_doy = np.arange(doy.min(), doy.max() + 1, oversample_days)

    rows = {}
    for pin, cs in splines.items():
        curve = cs(dense_doy)
        deriv = cs(dense_doy, 1)

        peak_idx = int(np.argmax(curve))
        peak_doy = dense_doy[peak_idx]

        post_peak = deriv[peak_idx:]
        decline_rate = post_peak.min() if len(post_peak) else deriv.min()

        rows[pin] = {
            "early_ndvi": pivot.loc[pin].iloc[0],
            "peak_ndvi": curve[peak_idx],
            "peak_doy": peak_doy,
            "green_up_rate": deriv.max(),
            "decline_rate": decline_rate,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def cluster(feats, k_range=range(2, 6), min_cluster_frac=0.05):
    X = StandardScaler().fit_transform(feats.values)

    # Silhouette alone rewards isolating a single extreme outlier as its own
    # "cluster" -- k=4 here technically scores marginally higher than k=3,
    # but only because it peels off one parcel with an extreme decline_rate,
    # not because it finds a fourth real behavioral group. Restricting to
    # k's where every cluster holds a meaningful share of the data (>=5%)
    # picks the interpretable split instead of the technically-optimal one.
    min_size = max(3, int(min_cluster_frac * len(feats)))
    scores, labels_by_k = {}, {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        counts = np.bincount(labels)
        if counts.min() < min_size:
            print(f"  k={k}: rejected, smallest cluster only {counts.min()} parcels", flush=True)
            continue
        scores[k] = silhouette_score(X, labels)
        labels_by_k[k] = labels
    best_k = max(scores, key=scores.get)
    print("Silhouette scores by k (viable only):", {k: round(v, 3) for k, v in scores.items()})
    print(f"Best k = {best_k}", flush=True)

    feats = feats.copy()
    feats["cluster"] = labels_by_k[best_k]
    return feats, best_k


def plot_spline_sample(splines, doy, pivot, out_path, n=12, seed=0):
    """Raw points + fitted spline for a sample of parcels -- a visual check
    that the spline is tracking the real shape, not overshooting/oscillating
    in the wide gaps (worst case: 45 days, May 9 -> June 23)."""
    import random

    import matplotlib.pyplot as plt

    dense_doy = np.linspace(doy.min(), doy.max(), 300)
    pins = random.Random(seed).sample(list(splines.keys()), min(n, len(splines)))

    fig, ax = plt.subplots(figsize=(9, 6))
    for pin in pins:
        cs = splines[pin]
        line, = ax.plot(dense_doy, cs(dense_doy), linewidth=1, alpha=0.7)
        ax.scatter(doy, pivot.loc[pin].values, color=line.get_color(), s=15, zorder=3)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("NDVI")
    ax.set_title(f"Cubic spline fit, {len(pins)} sample parcels (dots = observed dates)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}", flush=True)


def plot_clusters(splines, doy, feats, out_path):
    import matplotlib.pyplot as plt

    dense_doy = np.linspace(doy.min(), doy.max(), 300)
    colors = plt.cm.Set1.colors

    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster_id, group in feats.groupby("cluster"):
        color = colors[cluster_id % len(colors)]
        for pin in group.index:
            ax.plot(dense_doy, splines[pin](dense_doy), color=color, alpha=0.15, linewidth=1)
        mean_curve = np.mean([splines[pin](dense_doy) for pin in group.index], axis=0)
        ax.plot(dense_doy, mean_curve, color=color, linewidth=3,
                 label=f"Group {cluster_id} (n={len(group)})")

    ax.set_xlabel("Day of year")
    ax.set_ylabel("NDVI")
    ax.set_title("NDVI season curves by behavioral cluster (spline-smoothed)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    df = load_series()
    splines, doy, pivot = fit_splines(df)
    print(f"{len(splines)} parcels with complete 8-date series, spline-fit over day-of-year {list(doy)}", flush=True)

    plot_spline_sample(splines, doy, pivot, REPORTS_DIR / "subset_spline_fit.png")

    feats = extract_features(splines, doy, pivot)
    feats, best_k = cluster(feats)
    print("\nCluster sizes:")
    print(feats["cluster"].value_counts().sort_index())
    print("\nCluster feature means:")
    print(feats.groupby("cluster").mean().round(3))

    plot_clusters(splines, doy, feats, REPORTS_DIR / "crop_clusters.png")

    feats.to_csv(REPORTS_DIR / "crop_clusters.csv")
    print(f"\nWrote {REPORTS_DIR / 'crop_clusters.csv'}", flush=True)
