"""Crop-type work: cluster parcels by curve *shape* rather than a single
date's value -- the whole premise from the start of this thread is that
corn and soybean (and other land uses) differ in when and how fast they
green up and senesce, not in what index value they happen to hit on any
one day.

Started fully unsupervised (no ground truth, clusters reported as
behavioral groups rather than validated "corn"/"soybean" labels) but is
no longer: this module's corn/soybean split (CORN_SOYBEAN_COEF, in
cluster() below) is a logistic regression fit against real USDA CDL
ground truth -- see validate_against_cdl.py and the README's "Validating
against real ground truth" section for the full methodology and the
several real, evidence-driven corrections that produced it. The
non-row-crop split (NON_ROW_CROP_EARLY_NDVI) remains an unvalidated,
threshold-based prior in the sense described below.

Defaults to evi2_zonal_stats_subset_clipped (see clip_parcels.py for the
parcels_clipped geometry, load_series() for the EVI2-vs-NDVI table
default -- EVI2 outperformed NDVI once validated against real CDL labels,
see the README), computed
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
which is the right trade for bounded physical data like NDVI.

Scaling to full-county interactive popups later surfaced a second, related
problem: PCHIP interpolates *exactly* through every point by construction,
so a single residual cloud-shadow/haze pixel that survives per-pixel
masking on just one date, for just one parcel, showed up in the fitted
curve as a sharp, physically implausible spike or dip -- correct behavior
for an interpolator, wrong behavior once that kind of per-parcel artifact
was understood to be a real, recurring thing at this scale rather than an
occasional whole-date event (see BAD_DATES above, and fit_splines'
docstring for the fix: per-parcel outlier rejection, then a genuine
smoothing spline -- scipy.interpolate.UnivariateSpline -- that no longer
passes through every remaining point exactly). The curve is still only
ever evaluated within a given parcel's own observed date range -- no
extrapolation past its own first or last valid date.

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
from scipy.interpolate import PchipInterpolator, UnivariateSpline
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


def load_series(table_name="evi2_zonal_stats_subset_clipped", bad_dates=None, value_column="evi2_mean"):
    """bad_dates defaults to BAD_DATES (the subset's known-bad Sentinel-2
    date) -- pass a different set (or empty) for a different table, since a
    bad date found for one AOI/date-set isn't necessarily meaningful for
    another. Check the new table's own date-by-date trajectory rather than
    assuming this default applies.

    value_column names the underlying vegetation-index column to read
    (e.g. "evi2_mean" or "ndvi_mean", see zonal_stats.compute_zonal_stats'
    `prefix` argument) -- aliased back to "ndvi_mean" in the returned
    DataFrame regardless, since every downstream function in this module
    (fit_splines, extract_features, ...) is agnostic to which index it's
    actually operating on and keeps that column name for continuity. Real
    validation against USDA CDL 2021 ground truth found EVI2 clearly
    outperforms NDVI for this specific classification task (87.6% vs 80.1%
    cross-validated accuracy -- see validate_against_cdl.py), which is why
    it's the default here now; NDVI stays the default for visualize.py's
    separate anomaly-detection pipeline, which this finding doesn't bear on."""
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
            f"SELECT pin, date, {value_column} AS ndvi_mean FROM {table_name} "
            f"WHERE {value_column} IS NOT NULL AND date NOT IN :bad_dates ORDER BY pin, date"
        ).bindparams(bindparam("bad_dates", expanding=True))
        params = {"bad_dates": list(bad_dates)}
    else:
        query = text(f"SELECT pin, date, {value_column} AS ndvi_mean FROM {table_name} "
                     f"WHERE {value_column} IS NOT NULL ORDER BY pin, date")
        params = {}
    df = pd.read_sql(query, engine, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df


MIN_VALID_DATES = 6
MIN_SEASON_SPAN_DAYS = 60
OUTLIER_RESIDUAL_THRESHOLD = 0.15
SMOOTHING_NDVI_NOISE_STD = 0.02


def _reject_outliers(doy, values, threshold=OUTLIER_RESIDUAL_THRESHOLD):
    """Rejects single-point spikes/troughs that don't fit their immediate
    neighbors -- a residual cloud-shadow or haze pixel that slips past the
    per-pixel SCL/Fmask mask on just one date, for just this one parcel,
    shows up as a sharp jump inconsistent with real phenology (which moves
    gradually day to day). This is different in kind from BAD_DATES above:
    that was one whole-scene bad Sentinel-2 pass affecting every parcel in
    the subset on the same date, found and excluded by hand. A per-pixel/
    per-parcel artifact like this can't be hand-curated one date at a time
    at county scale, since it isn't tied to one date at all -- it has to be
    a per-parcel statistical test.

    For each interior point, compares its value against where a straight
    line between its immediate left and right neighbors (at their actual,
    irregularly-spaced doy) would put it, and flags it as an outlier if the
    residual exceeds `threshold` (a fixed NDVI amount, not a relative/MAD-
    based one).

    A first version used a Hampel-filter-style test instead (comparing each
    point to the median of a small window of rank-order neighbors) and
    rejected ~16% of all points on this data -- checked, and it wasn't
    conservatively catching only real artifacts: it collapsed the
    subset's corn-like cluster from 45 to 23 parcels, because comparing a
    point to nearby *rank-order* neighbors doesn't account for how far
    apart they actually are in time, and corn's genuine signal -- a fast,
    sustained green-up -- looks exactly like an "outlier" relative to a
    small index-window after a real gap in the data, however smooth the
    true curve is at the actual sampling times. A straight line through a
    point's real time-adjacent neighbors doesn't have that problem: a fast
    but *sustained* real change still lands close to that line (the point
    sits between two neighbors moving the same direction), while a genuine
    isolated spike or dip -- one where the neighbors on both sides broadly
    agree with each other but not with it -- shows up as a real residual
    from it. threshold=0.15 is picked to be well above ordinary sensor/
    atmospheric noise but below the size of jump a real, sustained
    transition can produce between typically-spaced (roughly weekly)
    observations.

    Only catches isolated single-point outliers -- two consecutive bad
    points would use each other as a "neighbor" and can cancel out in this
    test. Applied once, not iteratively.
    """
    if len(values) < 3:
        return doy, values
    keep = np.ones(len(values), dtype=bool)
    for i in range(1, len(values) - 1):
        expected = np.interp(doy[i], [doy[i - 1], doy[i + 1]], [values[i - 1], values[i + 1]])
        if abs(values[i] - expected) > threshold:
            keep[i] = False
    return doy[keep], values[keep]


def fit_splines(df, min_valid_dates=MIN_VALID_DATES, min_season_span_days=MIN_SEASON_SPAN_DAYS,
                 outlier_threshold=OUTLIER_RESIDUAL_THRESHOLD):
    """One shape-preserving cubic (PCHIP) fit per parcel, over day-of-year.
    This is the *analysis* curve -- what extract_features reads peak_doy,
    green_up_rate, etc. off of. popup_curve_data below builds a separate,
    more heavily smoothed curve purely for the map popup's display line;
    the two are intentionally different fits (see popup_curve_data's
    docstring for why a smoothing spline here would be the wrong choice).

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

    So each parcel is fit on its own valid dates instead of a shared pivot.
    Outliers are rejected first (_reject_outliers, per-parcel, see its
    docstring), then a parcel is excluded entirely (not fit at all) rather
    than guessed at when its remaining data can't support a real curve:
    - min_valid_dates: below this, there just aren't enough points left to
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

    A real smoothing fit (scipy.interpolate.UnivariateSpline, s > 0) was
    tried here too, after the county-wide interactive map surfaced sharp,
    single-point spikes/troughs in the popup curve that survived per-pixel
    cloud masking -- but even a small smoothing budget turned out to erode
    the sharp, narrow peak that identifies a corn-like curve: on the
    subset, adding smoothing on top of outlier rejection collapsed the
    corn-like cluster from 63 parcels down to 18, because the squared-
    residual budget a smoothing spline spends is cheaper to spend flattening
    a real, narrow peak than tracking noise between the many flatter points
    around it -- exactly backwards for a method whose entire signal *is*
    peak shape. PCHIP (exact interpolation, unchanged from before) is kept
    for this analysis curve; only the display curve in popup_curve_data
    below applies real smoothing, where flattening a peak's exact shape
    slightly is a purely cosmetic cost, not a correctness one. Still
    evaluated only within each parcel's own observed range (see
    extract_features below) -- no extrapolation past its own first or last
    valid date.
    """
    long = df.dropna(subset=["ndvi_mean"]).copy()
    long["doy"] = long["date"].dt.dayofyear

    splines, kept_series = {}, {}
    excluded_count, excluded_short_span, rejected_points = 0, 0, 0
    for pin, group in long.groupby("pin"):
        group = group.drop_duplicates(subset="doy").sort_values("doy")
        doy = group["doy"].to_numpy()
        values = group["ndvi_mean"].to_numpy(dtype=float)

        clean_doy, clean_values = _reject_outliers(doy, values, threshold=outlier_threshold)
        rejected_points += len(doy) - len(clean_doy)

        if len(clean_doy) < min_valid_dates:
            excluded_count += 1
            continue
        span = clean_doy[-1] - clean_doy[0]
        if span < min_season_span_days:
            excluded_count += 1
            excluded_short_span += 1
            continue

        splines[pin] = PchipInterpolator(clean_doy, clean_values, extrapolate=False)
        kept_series[pin] = (clean_doy, clean_values)

    print(f"  fit_splines: {len(splines)} parcels kept, {excluded_count} excluded "
          f"({excluded_short_span} for season span, "
          f"{excluded_count - excluded_short_span} for date count), "
          f"{rejected_points} outlier points rejected", flush=True)

    return splines, kept_series


def extract_features(splines, kept_series, oversample_days=1):
    """Each parcel's dense grid is built from its own domain (kept_series,
    from fit_splines -- its own outlier-rejected doy values), not a shared
    grid: since fit_splines fits each parcel on its own valid dates (see
    its docstring), a shared grid spanning the full season would evaluate
    parcels with a narrower own range outside their domain."""
    rows = {}
    for pin, cs in splines.items():
        own_doy, _ = kept_series[pin]
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


# This went through three real, evidence-driven iterations, each one
# found by actually deploying the previous version rather than trusting
# validation numbers alone -- see validate_against_cdl.py for the
# methodology (past-season imagery capped at the current season's
# day-of-year reach, validated against that same season's real USDA CDL
# labels, so there's no crop-rotation ambiguity and no lookahead).
#
# v1 -- absolute peak_doy threshold. Cross-validated against the 163-parcel
# SUBSET's CDL labels: 186 won every fold, 87.6% accuracy. Applying it
# unchanged to 2026 found only 2 of 150 corn-like parcels (vs. 40-60
# expected): peak_doy is frequently just the date of whichever satellite
# pass caught a parcel's true peak, and that date shifts every season, so
# a fixed calendar day doesn't transfer.
#
# v2 -- percentile of peak_doy, not an absolute day (31.7th percentile,
# what 186 corresponded to on the subset). Fixed the season-to-season
# transfer problem, but exposed a bigger one: validated on the full
# COUNTY's real CDL labels (8,202 parcels, not the subset's 163), peak_doy
# alone only reaches 63.8% cross-validated accuracy, with the actual
# optimal cutoff at the 66.6th percentile, not 31.7 -- nearly inverted.
# The subset's 87.6% was never representative: its own true CDL corn
# fraction (41.6%) doesn't even match the county's real acreage split
# (~52%), because a small rural cluster chosen for fetch cost, not
# representativeness, just happens to have a different local rotation mix.
# Deploying a percentile calibrated there was always going to bias the
# whole county's aggregate split, no matter how well peak_doy's *rank*
# supposedly transferred -- confirmed directly: 2026's county-wide split
# came out 31.7% corn against a real ~52%.
#
# v3 -- a proper multi-feature model, fit on the representative
# county-wide sample. peak_doy's weakness at county scale traces to the
# same quantization problem documented throughout this module (huge blocks
# of parcels tied to one calendar date), but two other features turned out
# to separate corn from soybean far more cleanly there: peak_ndvi (mean
# 0.700 corn vs. 0.797 soybean -- corn's canopy structure caps out lower)
# and green_up_rate (already established: corn rises faster). A logistic
# regression on all 5 features, cross-validated on the same 8,202-parcel
# county sample, reaches 80.5% (+/- 0.7%) -- beating peak_ndvi alone
# (77.0%) and far beating peak_doy alone (63.8%). This *reverses* the
# subset-scale finding that more features hurt (a 5-feature model
# underperformed a single threshold there) -- expected, not a
# contradiction: 161 samples isn't enough to reliably fit 5 coefficients,
# 8,202 is. Fit on raw (absolute) feature values, predicted corn fraction
# on that same 2021 training data was 47.9%, closely tracking the true
# 48.6%.
#
# v4 (current) -- same model, refit on each feature *standardized against
# its own season's row-crop population* (z-score: (value - that season's
# mean) / that season's std) rather than raw absolute values. Needed
# because v3, deployed on 2026, only produced 37.6% corn against the
# expected ~52% -- checked directly rather than assumed, and found a real
# season-to-season shift: 2026's row-crop peak_ndvi averages 0.793 vs.
# 2021's 0.737 (roughly half a standard deviation higher), and peak_ndvi
# carries the largest weight in the model, so that shift alone biased
# every 2026 prediction toward soybean. This is the same lesson as
# CORN_SOYBEAN_PEAK_DOY_PERCENTILE's abandoned absolute-day threshold,
# generalized: a raw feature value isn't comparable across seasons with
# different overall imagery/atmospheric conditions, but a value's position
# *relative to that season's own distribution* is more likely to be.
# Standardizing preserved cross-validated accuracy on 2021 (80.65% vs.
# 80.54%, no real change) while fixing the 2026 deployment: corn share
# moved from 37.6% to 56.1%, much closer to the real ~52% acreage split.
# Standardization at inference time uses each run's own row-crop
# population mean/std (see cluster() below), computed fresh every run --
# there's no fixed, portable "mean EVI2 value" to hardcode any more than
# there was a fixed peak_doy.
CORN_SOYBEAN_FEATURES = ["early_ndvi", "peak_ndvi", "peak_doy", "green_up_rate", "decline_rate"]
CORN_SOYBEAN_COEF = {
    "early_ndvi": -0.2223,
    "peak_ndvi": -2.0819,
    "peak_doy": -1.3480,
    "green_up_rate": 0.0775,
    "decline_rate": -0.3817,
}
CORN_SOYBEAN_INTERCEPT = 0.4298

# NDVI's 0.3 non-row-crop cutoff doesn't transfer to EVI2's naturally lower
# scale (EVI2 reads systematically lower than NDVI for the same
# vegetation -- median early-season value ~0.11 vs NDVI's ~0.18 on this
# subset). 0.19 is percentile-matched to reproduce NDVI's split size
# rather than independently validated: CDL 2021 only had 2 true "Other"
# parcels in this subset, too few to cross-validate a threshold against
# the way the corn/soybean split was -- but both of those 2 parcels'
# early_evi2 values (0.285, 0.371) do fall above 0.19, weak corroborating
# evidence rather than a proper validation. Kept as an absolute EVI2 value,
# not a percentile like the corn/soybean split above: it's a level (how
# green is this parcel in early spring), not a calendar day, so it isn't
# exposed to the same observation-date-sensitivity problem.
NON_ROW_CROP_EARLY_NDVI = 0.19

# cluster() always assigns these three ids by construction (0 = the
# deterministic early_ndvi split, 1/2 = the deterministic peak_doy-
# percentile split within row-crop, lower half first) -- not derived from
# each cluster's own feature means the way an earlier KMeans-based version
# needed to, since there's no arbitrary cluster numbering to resolve here.
CLUSTER_LABELS = {
    0: "Non-row-crop (already green in April)",
    1: "Corn-like (early peak, fast decline)",
    2: "Soybean-like (later peak, slower decline)",
}


def cluster(feats, non_row_crop_early_ndvi=NON_ROW_CROP_EARLY_NDVI):
    """Non-row-crop vs. row-crop on early_ndvi (a deterministic threshold),
    then corn-like vs. soybean-like within row-crop via the logistic
    regression fit described at CORN_SOYBEAN_COEF -- not a KMeans fit, and
    not a single-feature threshold either (see that comment for why both
    were tried and replaced).

    Confidence for the non-row-crop decision is a margin-based mechanism:
    how far a parcel's own early_ndvi sits from the threshold relative to
    the population's spread, clipped to [0.5, 1.0]. Confidence for
    corn/soybean is the logistic model's own predicted probability for
    whichever label it assigned (max(p, 1-p)) -- genuinely continuous by
    construction, unlike the single-feature threshold versions this
    replaced, where a margin computed from heavily-quantized peak_doy
    alone collapsed to only 3-5 distinct values across thousands of
    parcels (found by a direct report that clicking ~30 parcels on the
    live map only ever showed 3 confidence numbers).
    """
    def threshold_confidence(series, threshold):
        std = series.std()
        margin = (series - threshold).abs() / (std if std > 0 else 1)
        return np.clip(0.5 + margin, 0.5, 1.0)

    feats = feats.copy()
    is_non_row_crop = feats["early_ndvi"] > non_row_crop_early_ndvi

    non_row_crop = feats[is_non_row_crop].copy()
    non_row_crop["cluster"] = 0
    non_row_crop["confidence"] = threshold_confidence(non_row_crop["early_ndvi"], non_row_crop_early_ndvi)

    row_crop = feats[~is_non_row_crop].copy()
    # Standardized against THIS run's own row-crop population (not a fixed
    # mean/std baked into the model) -- see CORN_SOYBEAN_COEF's comment:
    # raw feature values shift season to season (2026's peak_ndvi ran
    # ~0.056 higher than the 2021 season the model was fit on), but each
    # value's position relative to its own season's distribution held up.
    standardized = (row_crop[CORN_SOYBEAN_FEATURES] - row_crop[CORN_SOYBEAN_FEATURES].mean()) \
        / row_crop[CORN_SOYBEAN_FEATURES].std()
    score = CORN_SOYBEAN_INTERCEPT + sum(
        CORN_SOYBEAN_COEF[col] * standardized[col] for col in CORN_SOYBEAN_FEATURES
    )
    p_corn = 1 / (1 + np.exp(-score))
    is_corn = p_corn >= 0.5
    row_crop["cluster"] = np.where(is_corn, 1, 2)
    row_crop["confidence"] = np.where(is_corn, p_corn, 1 - p_corn)

    print(f"Non-row-crop split (early_ndvi > {non_row_crop_early_ndvi}): "
          f"{len(non_row_crop)} non-row-crop, {len(row_crop)} row-crop "
          f"-> corn/soybean split (logistic model): "
          f"{int(is_corn.sum())} corn-like ({100 * is_corn.mean():.1f}%), "
          f"{int((~is_corn).sum())} soybean-like", flush=True)

    result = pd.concat([non_row_crop, row_crop])
    return result.loc[feats.index]


def popup_curve_data(splines, kept_series, display_noise_std=SMOOTHING_NDVI_NOISE_STD):
    """Per-parcel dense curve, as a GeoJSON-ready JSON string, plus how many
    valid (outlier-rejected) dates it's based on -- shared by
    visualize_subset.py and visualize_county_crops.py's interactive maps.
    Built per pin over that pin's own observed range (kept_series, from
    fit_splines) rather than one shared range for every parcel: since
    fit_splines fits each parcel on its own valid dates (parcels no longer
    all share one global date set -- see its docstring), a shared range
    would show fabricated values on dates a given parcel never cleared.

    Deliberately fits its own separate UnivariateSpline here rather than
    reusing `splines` (the PCHIP analysis fit from fit_splines): a real
    smoothing spline was tried for the analysis fit itself and rejected --
    even a small smoothing budget eroded the sharp peak that identifies a
    corn-like curve (see fit_splines' docstring for the measured effect,
    63->18 parcels). That failure mode is specific to using the smoothed
    curve for *feature extraction*; using it only for what's drawn in a
    popup has no such cost, and a real smoothing fit is exactly what makes
    the displayed line not visually kink at ordinary sensor noise the
    outlier filter didn't catch (which only targets clear single-point
    spikes/dips, not general small jitter).

    Individual raw points are deliberately not plotted alongside this
    curve: with the display curve now a real smoothing fit (not exact
    interpolation), a dot at each raw date would show the actual
    (slightly) noisy measurement sitting off the smoothed line, which
    reads as the fit being wrong rather than as the point being ordinary
    noise it's smoothing past. The date count (n_dates) is surfaced as
    plain text in the popup instead.
    """
    rows = {}
    for pin, cs in splines.items():
        own_doy, own_values = kept_series[pin]
        dense_doy = np.arange(int(own_doy.min()), int(own_doy.max()) + 1)
        if len(own_doy) >= 4:
            s = display_noise_std ** 2 * len(own_doy)
            display_cs = UnivariateSpline(own_doy, own_values, k=3, s=s, ext=3)
            dense_values = display_cs(dense_doy)
        else:
            dense_values = cs(dense_doy)
        rows[pin] = {
            "ndvi_dense_start_doy": int(dense_doy[0]),
            "ndvi_dense": json.dumps([round(v, 3) for v in dense_values]),
            "n_dates": int(len(own_doy)),
        }
    return pd.DataFrame.from_dict(rows, orient="index")




def plot_spline_sample(splines, kept_series, out_path, n=12, seed=0):
    """Raw points + fitted spline for a sample of parcels -- a visual check
    that the spline is tracking the real shape (smoothing past noise, not
    overshooting/oscillating in the wide gaps -- worst case, 45 days, May 9
    -> June 23) and that outlier rejection is dropping the right points."""
    import random

    import matplotlib.pyplot as plt

    pins = random.Random(seed).sample(list(splines.keys()), min(n, len(splines)))

    fig, ax = plt.subplots(figsize=(9, 6))
    for pin in pins:
        cs = splines[pin]
        own_doy, own_values = kept_series[pin]
        dense_doy = np.linspace(own_doy.min(), own_doy.max(), 300)
        line, = ax.plot(dense_doy, cs(dense_doy), linewidth=1, alpha=0.7)
        # The real (kept, post-outlier-rejection) measured values -- with
        # UnivariateSpline as a genuine smoothing fit, cs(own_doy) would show
        # the smoothed curve's value at those days, not the raw measurement.
        ax.scatter(own_doy, own_values, color=line.get_color(), s=15, zorder=3)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Vegetation index")
    ax.set_title(f"Smoothing spline fit, {len(pins)} sample parcels (dots = kept observed dates)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}", flush=True)


def plot_clusters(splines, kept_series, feats, out_path):
    import matplotlib.pyplot as plt

    all_doy = np.concatenate([d for d, _ in kept_series.values()])
    dense_doy = np.linspace(all_doy.min(), all_doy.max(), 300)
    colors = plt.cm.Set1.colors

    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster_id, group in feats.groupby("cluster"):
        color = colors[cluster_id % len(colors)]
        curves = []
        for pin in group.index:
            own_doy, _ = kept_series[pin]
            curve = splines[pin](dense_doy)
            # UnivariateSpline's ext=3 (fit_splines) returns a flat boundary
            # value outside a parcel's own domain rather than NaN -- masked
            # here instead, so the plotted gap is an honest "we don't know,"
            # not a manufactured flat continuation.
            curve = np.where((dense_doy >= own_doy.min()) & (dense_doy <= own_doy.max()), curve, np.nan)
            ax.plot(dense_doy, curve, color=color, alpha=0.15, linewidth=1)
            curves.append(curve)
        mean_curve = np.nanmean(curves, axis=0)
        ax.plot(dense_doy, mean_curve, color=color, linewidth=3,
                 label=f"Group {cluster_id} (n={len(group)})")

    ax.set_xlabel("Day of year")
    ax.set_ylabel("Vegetation index")
    ax.set_title("Season curves by behavioral cluster (spline-smoothed)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    df = load_series()
    splines, kept_series = fit_splines(df)
    all_doy = np.concatenate([d for d, _ in kept_series.values()])
    print(f"{len(splines)} parcels kept, day-of-year range {all_doy.min()}-{all_doy.max()} across them", flush=True)

    plot_spline_sample(splines, kept_series, REPORTS_DIR / "subset_spline_fit.png")

    feats = extract_features(splines, kept_series)
    feats = cluster(feats)
    print("\nCluster sizes:")
    print(feats["cluster"].value_counts().sort_index())
    print("\nCluster feature means:")
    print(feats.groupby("cluster").mean().round(3))

    plot_clusters(splines, kept_series, feats, REPORTS_DIR / "crop_clusters.png")

    feats.to_csv(REPORTS_DIR / "crop_clusters.csv")
    print(f"\nWrote {REPORTS_DIR / 'crop_clusters.csv'}", flush=True)
