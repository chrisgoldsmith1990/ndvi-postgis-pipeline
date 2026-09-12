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

Defaults to ndvi_zonal_stats_subset_clipped (see clip_parcels.py), computed
against parcel geometries with roads/waterways subtracted out via
ST_Difference -- a parcel's tax boundary can include a road or stream
running through the field itself, and those pixels read as pavement/water,
not crop. Confirmed as a real, physically sensible effect (pavement drags
summer NDVI down, green ditch-banks drag spring NDVI up, both up to ~0.05
NDVI on the most-affected parcel) but not one that changes the crop-type
story: clustering on clipped vs. unclipped data reassigns only 2 of 125
parcels and leaves silhouette/confidence essentially unchanged -- the
correction matters for per-parcel precision (feeds directly into
yield_ranking.py's acreage-based estimate), not for the aggregate split.

Only 8 dates survived cloud/shadow filtering out of 41 candidate Sentinel-2
passes this season at the strict (20% bad-pixel) threshold -- confirmed by
re-running the search with no whole-tile cloud pre-filter at all, the other
33 are genuinely too cloudy over this specific AOI, not an artifact of an
overly strict filter. The worst gap: 45 days between May 9 and June 23,
covering almost the entire green-up transition with zero observations --
which matters, because peak-timing and green-up-rate alone can't
distinguish "fast burst early, then plateau" from "steady climb the whole
way" if there's no data inside the window where that difference would show.

Loosening the threshold to 35% recovered two dates inside that exact gap
(May 14, June 15) plus a bonus (July 28) -- but July 28 turned out to be a
bad date, not just a noisier one: county-mean NDVI dropped to 0.43 on that
date alone, sandwiched between 0.83 (July 18) and 0.89 (Aug 22), and every
sampled parcel showed the same implausible crash-and-recover pattern
(stdev 0.26 vs ~0.03-0.08 on its neighbors). That's whole-scene residual
haze/thin cirrus the binary SCL mask didn't catch, not real phenology --
excluded explicitly below rather than let it corrupt every curve fit with
a spurious dip. May 14 and June 15 both integrate smoothly into their
neighbors' trajectories and are kept.

10 dates total, still irregularly spaced (two closely-spaced points right
at the end -- Sept 1 and Sept 3), which makes naive two-point differences
a poor way to estimate a "rate", so each parcel's series is fit with a
smooth curve instead and features are read off that curve: interpolated
peak timing/height, and the curve's own derivative for green-up/decline
rates.

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

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import bindparam, text

from src.db import get_engine

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


# 2026-07-28 passed the per-AOI SCL cloud/shadow check (74% clear) but is a
# bad date, not just a noisy one: county-mean NDVI dropped to 0.43 that day
# alone, between 0.83 (Jul 18) and 0.89 (Aug 22), with every sampled parcel
# showing the same implausible crash-and-recover shape (stdev 0.26 vs
# ~0.03-0.08 on neighboring dates) -- whole-scene residual haze the binary
# SCL mask didn't catch. Excluded explicitly rather than let it corrupt
# every curve fit with a spurious dip.
BAD_DATES = {"2026-07-28"}


def load_series(table_name="ndvi_zonal_stats_subset_clipped", bad_dates=None):
    """bad_dates defaults to BAD_DATES (the subset's known-bad Sentinel-2
    date) -- pass a different set (or empty) for a different table, since a
    bad date found for one AOI/date-set isn't necessarily meaningful for
    another. Check the new table's own date-by-date trajectory rather than
    assuming this default applies."""
    if bad_dates is None:
        bad_dates = BAD_DATES
    engine = get_engine()
    # An empty bad_dates (the county table's default -- no known-bad dates
    # yet) can't go through the expanding "NOT IN :bad_dates" bindparam:
    # SQLAlchemy renders an empty expanding IN/NOT IN as a typed NULL
    # subquery, and with no values to infer a type from it defaults to
    # INTEGER, which then fails to compare against a date/text column
    # ("operator does not exist: text = integer"). Skip the clause entirely
    # instead of exercising that edge case.
    if bad_dates:
        query = text(
            f"SELECT pin, date, ndvi_mean FROM {table_name} "
            f"WHERE ndvi_mean IS NOT NULL AND date NOT IN :bad_dates ORDER BY pin, date"
        ).bindparams(bindparam("bad_dates", expanding=True))
        params = {"bad_dates": list(bad_dates)}
    else:
        query = text(f"SELECT pin, date, ndvi_mean FROM {table_name} WHERE ndvi_mean IS NOT NULL ORDER BY pin, date")
        params = {}
    df = pd.read_sql(query, engine, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df


MIN_VALID_DATES = 6
MIN_SEASON_SPAN_DAYS = 60


def fit_splines(df, min_valid_dates=MIN_VALID_DATES, min_season_span_days=MIN_SEASON_SPAN_DAYS):
    """One shape-preserving cubic (PCHIP) fit per parcel, over day-of-year.

    At subset scale every parcel shared the exact same handful of dates
    (the county-wide cloud check was a whole-AOI aggregate, so a date either
    cleared for everyone or nobody), and a strict dropna() complete-case
    pivot was the right, simple choice. That whole-AOI assumption is what
    the county-scale fetch fixed (see fetch_timeseries.build_date's
    docstring): cloud cover is now checked and masked per-pixel, so which
    dates are usable now genuinely varies parcel to parcel -- some sit under
    a clear sky on 40 of the 43 fetched dates, others clear on 8. Requiring
    one shared date list for all ~9,571 parcels would mean either dropping
    almost every parcel (to the dates *everyone* has) or almost every date
    (to parcels that happen to have all of them) -- neither is right when
    the actual cloud pattern is genuinely parcel-specific.

    So each parcel is fit on its own valid dates instead of a shared pivot,
    and a parcel is excluded (not fit at all) rather than guessed at when
    its own data can't support a real curve:
    - min_valid_dates: below this, PCHIP is just connecting too few dots to
      resolve the shape features below (green-up rate, decline rate) from
      genuine curvature rather than noise between two points.
    - min_season_span_days: count alone isn't enough -- a parcel with 8
      clear dates all bunched into a 3-week window in June has no data
      anywhere near green-up or senescence, so a peak-timing/decline-rate
      read off that curve would be pure extrapolation dressed up as a fit.
      Requiring the parcel's own first-to-last valid date to span most of
      the season keeps only parcels whose curve actually covers the
      transitions the crop-type signal depends on.
    Excluded parcels are dropped silently here; the caller reports the
    excluded count so it's visible, not swallowed.
    """
    long = df.dropna(subset=["ndvi_mean"]).copy()
    long["doy"] = long["date"].dt.dayofyear

    splines, kept_doy = {}, {}
    excluded_count, excluded_short_span = 0, 0
    for pin, group in long.groupby("pin"):
        group = group.drop_duplicates(subset="doy").sort_values("doy")
        if len(group) < min_valid_dates:
            excluded_count += 1
            continue
        span = group["doy"].iloc[-1] - group["doy"].iloc[0]
        if span < min_season_span_days:
            excluded_count += 1
            excluded_short_span += 1
            continue
        doy = group["doy"].to_numpy()
        # extrapolate=False: features and plots below only ever evaluate a
        # parcel's spline within its own observed range, but plot_clusters
        # shares one dense grid across parcels with different ranges for the
        # overlay -- outside its own domain that must come back NaN, not a
        # silently extrapolated (and, for PCHIP past the boundary segment,
        # potentially out-of-[0,1]) value.
        splines[pin] = PchipInterpolator(doy, group["ndvi_mean"].to_numpy(dtype=float), extrapolate=False)
        kept_doy[pin] = doy

    print(f"  fit_splines: {len(splines)} parcels kept, {excluded_count} excluded "
          f"({excluded_short_span} for season span, "
          f"{excluded_count - excluded_short_span} for date count)", flush=True)

    all_doy = np.array(sorted({d for arr in kept_doy.values() for d in arr}))
    # Pivoted on the actual calendar date, not day-of-year: visualize_subset.py
    # reads real dates back off these columns for the hover chart's raw-point
    # labels, and day-of-year alone can't round-trip to a date without also
    # knowing the year.
    pivot = long[long["pin"].isin(splines)].pivot(index="pin", columns="date", values="ndvi_mean")
    return splines, all_doy, pivot


def extract_features(splines, doy, pivot, oversample_days=1):
    """doy is accepted for backward compatibility with existing callers but
    is no longer used to build one shared grid: since fit_splines now fits
    each parcel on its own valid dates (see its docstring), a shared grid
    spanning the full season would evaluate parcels with a narrower own
    range outside their domain, where extrapolate=False returns NaN. Each
    parcel's dense grid is instead built from its own domain, read directly
    off its PchipInterpolator's own breakpoints (cs.x) -- the pivot/doy
    args are accepted for backward compatibility with existing callers but
    no longer used."""
    rows = {}
    for pin, cs in splines.items():
        own_doy = cs.x
        dense_doy = np.arange(own_doy.min(), own_doy.max() + 1, oversample_days)
        curve = cs(dense_doy)
        deriv = cs(dense_doy, 1)

        peak_idx = int(np.argmax(curve))
        peak_doy = dense_doy[peak_idx]

        post_peak = deriv[peak_idx:]
        decline_rate = post_peak.min() if len(post_peak) else deriv.min()

        rows[pin] = {
            "early_ndvi": curve[0],
            "peak_ndvi": curve[peak_idx],
            "peak_doy": peak_doy,
            "green_up_rate": deriv.max(),
            "decline_rate": decline_rate,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


NON_ROW_CROP_EARLY_NDVI = 0.3


def label_cluster(row):
    """Behavioral label from a cluster's own feature means -- not a fixed
    ID mapping, since KMeans cluster numbering is arbitrary per run."""
    if row["early_ndvi"] > NON_ROW_CROP_EARLY_NDVI:
        return "Non-row-crop (already green in April)"
    if row["peak_doy"] < 210:
        return "Corn-like (early peak, fast decline)"
    return "Soybean-like (later peak, slower decline)"


def _fit_kmeans(feats, k_range, min_cluster_frac):
    """One silhouette-selected KMeans fit, factored out of cluster() so it
    can run twice (see cluster()'s docstring for why)."""
    X = StandardScaler().fit_transform(feats.values)

    # Silhouette alone rewards isolating a single extreme outlier as its own
    # "cluster" -- k=4 here technically scores marginally higher than k=3,
    # but only because it peels off one parcel with an extreme decline_rate,
    # not because it finds a fourth real behavioral group. Restricting to
    # k's where every cluster holds a meaningful share of the data (>=5%)
    # picks the interpretable split instead of the technically-optimal one.
    min_size = max(3, int(min_cluster_frac * len(feats)))
    scores, labels_by_k, model_by_k = {}, {}, {}
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        labels = model.labels_
        counts = np.bincount(labels)
        if counts.min() < min_size:
            print(f"  k={k}: rejected, smallest cluster only {counts.min()} parcels", flush=True)
            continue
        scores[k] = silhouette_score(X, labels)
        labels_by_k[k] = labels
        model_by_k[k] = model
    best_k = max(scores, key=scores.get)
    print("Silhouette scores by k (viable only):", {k: round(v, 3) for k, v in scores.items()})
    print(f"Best k = {best_k}", flush=True)

    feats = feats.copy()
    feats["cluster"] = labels_by_k[best_k]
    feats["confidence"] = assignment_confidence(X, model_by_k[best_k])
    return feats, best_k


def cluster(feats, k_range=range(2, 6), min_cluster_frac=0.05,
            non_row_crop_early_ndvi=NON_ROW_CROP_EARLY_NDVI):
    """Two-stage split: first pull out non-row-crop parcels (already green
    in April -- winter cover, pasture, hay) with the same deterministic
    early_ndvi threshold label_cluster() itself checks first, then cluster
    only the remaining row-crop parcels to find the corn/soybean
    peak-timing split.

    An earlier version tried to find *both* splits from one joint KMeans
    fit across the whole population, which does not reliably work: on the
    full McLean County run (~9,565 parcels), silhouette-based k-selection
    over everyone together picked k=2 and lumped every row-crop parcel into
    one undifferentiated cluster, identifying zero as corn-like, despite
    the county being close to half corn by actual harvested acreage
    (318,000 corn vs. 294,000 soybean acres, 2025 NASS) -- the early_ndvi
    outlier split is a strong, high-variance single-axis signal that
    silhouette prefers over the comparatively subtler peak-timing split. A
    second attempt tried using KMeans itself (forced to k=2) to *find* the
    non-row-crop split before re-clustering the rest, which fixed the
    county run but doesn't generalize: at the ~163-parcel subset scale that
    first k=2 fit split on a different, unrelated axis instead, leaving
    almost nothing for the second stage. A fixed threshold on early_ndvi
    sidesteps both failure modes -- it's already how label_cluster() names
    the resulting clusters after the fact, so applying it before clustering
    too, to actually perform the split, is consistent rather than hoping
    an unsupervised fit rediscovers the same rule on its own.

    Confidence for the non-row-crop decision isn't from a KMeans margin
    (there's no fit backing this split), so it's read directly off how far
    a parcel's own early_ndvi sits from the threshold relative to the
    population's spread -- clipped to the same [0.5, 1.0] bound
    assignment_confidence uses, so the two are on a comparable scale even
    though they come from different mechanisms.
    """
    is_non_row_crop = feats["early_ndvi"] > non_row_crop_early_ndvi
    std = feats["early_ndvi"].std()
    margin = (feats["early_ndvi"] - non_row_crop_early_ndvi).abs() / (std if std > 0 else 1)

    non_row_crop = feats[is_non_row_crop].copy()
    non_row_crop["cluster"] = 0
    non_row_crop["confidence"] = np.clip(0.5 + margin[is_non_row_crop], 0.5, 1.0)

    row_crop = feats.loc[~is_non_row_crop, feats.columns]
    stage2, best_k = _fit_kmeans(row_crop, k_range=k_range, min_cluster_frac=min_cluster_frac)
    stage2["cluster"] = stage2["cluster"] + 1  # +1 so ids never collide with non-row-crop's 0

    print(f"Non-row-crop split (early_ndvi > {non_row_crop_early_ndvi}): "
          f"{len(non_row_crop)} non-row-crop, {len(stage2)} row-crop "
          f"-> KMeans best_k={best_k} on row-crop", flush=True)

    result = pd.concat([non_row_crop, stage2])
    return result.loc[feats.index], best_k


def popup_curve_data(splines, pivot):
    """Per-parcel dense curve + raw observed points, as GeoJSON-ready JSON
    strings -- shared by visualize_subset.py and visualize_county_crops.py's
    interactive maps. Built per pin over that pin's own observed range
    (cs.x) rather than one shared range for every parcel: since fit_splines
    now fits each parcel on its own valid dates (parcels no longer all
    share one global date set -- see its docstring), a shared range would
    show fabricated values on dates a given parcel never actually cleared."""
    rows = {}
    for pin, cs in splines.items():
        dense_doy = np.arange(int(cs.x.min()), int(cs.x.max()) + 1)
        own_dates = pivot.loc[pin].dropna().sort_index().index
        rows[pin] = {
            "ndvi_dense_start_doy": int(dense_doy[0]),
            "ndvi_dense": json.dumps([round(v, 3) for v in cs(dense_doy)]),
            "ndvi_raw_doy": json.dumps([int(d) for d in cs.x]),
            "ndvi_raw_dates": json.dumps([d.strftime("%Y-%m-%d") for d in own_dates]),
            "ndvi_raw_values": json.dumps([round(v, 3) for v in cs(cs.x)]),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def assignment_confidence(X, model):
    """How much closer each point is to its assigned cluster's centroid than
    to the next-closest one, in the same standardized space KMeans itself
    uses -- not a calibrated probability, but a direct, honest read of how
    ambiguous a call actually was. distance_to_own is always <= the second-
    smallest distance by construction (KMeans assigns to the nearest
    centroid), so this is bounded [0.5, 1.0]: 0.5 is an exact tie between
    two groups, 1.0 is unambiguous. Generalizes to k>2 by comparing against
    whichever *other* centroid is nearest, not an average of all of them --
    a point near the boundary of two of three groups is genuinely
    ambiguous even if it's far from the third.
    """
    dists = np.linalg.norm(X[:, None, :] - model.cluster_centers_[None, :, :], axis=2)
    sorted_dists = np.sort(dists, axis=1)
    d_own, d_next = sorted_dists[:, 0], sorted_dists[:, 1]
    return d_next / (d_own + d_next)


def plot_spline_sample(splines, doy, pivot, out_path, n=12, seed=0):
    """Raw points + fitted spline for a sample of parcels -- a visual check
    that the spline is tracking the real shape, not overshooting/oscillating
    in the wide gaps (worst case: 45 days, May 9 -> June 23)."""
    import random

    import matplotlib.pyplot as plt

    pins = random.Random(seed).sample(list(splines.keys()), min(n, len(splines)))

    fig, ax = plt.subplots(figsize=(9, 6))
    for pin in pins:
        cs = splines[pin]
        own_doy = cs.x
        dense_doy = np.linspace(own_doy.min(), own_doy.max(), 300)
        line, = ax.plot(dense_doy, cs(dense_doy), linewidth=1, alpha=0.7)
        # cs(own_doy) reproduces the original measured values exactly (PCHIP
        # interpolates through its own data points) -- reading straight off
        # the spline's own breakpoints instead of pivot avoids any mismatch
        # from the dedup a duplicate-doy date could cause in fit_splines.
        ax.scatter(own_doy, cs(own_doy), color=line.get_color(), s=15, zorder=3)
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
            # extrapolate=False (fit_splines) means this is NaN, hence a
            # plotted gap, outside this parcel's own observed range -- a
            # true reflection of what's actually known about that parcel,
            # not a manufactured curve on unobserved dates.
            ax.plot(dense_doy, splines[pin](dense_doy), color=color, alpha=0.15, linewidth=1)
        mean_curve = np.nanmean([splines[pin](dense_doy) for pin in group.index], axis=0)
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
    print(f"{len(splines)} parcels kept, {len(doy)} distinct days-of-year observed "
          f"across them (day-of-year range {doy.min()}-{doy.max()})", flush=True)

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
