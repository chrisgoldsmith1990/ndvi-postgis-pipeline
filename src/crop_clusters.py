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


# Validated against real USDA CDL 2021 ground truth for the subset area
# (validate_against_cdl.py): a straight peak_doy sweep against 161
# CDL-labeled Corn/Soybean parcels, 5-fold cross-validated, picked an
# absolute cutoff of 186 in *every* fold with zero variance -- 87.6%
# accuracy (EVI2), vs. 77.0% at the previous hand-picked 210.
#
# That absolute day-of-year cutoff does NOT transfer across seasons,
# though -- found out by actually deploying it, not just assumed: applying
# 186 unchanged to the 2026 season identified only 2 of 150 row-crop
# parcels as corn (vs. 43-67 expected), because peak_doy isn't a clean
# biological measurement independent of when the satellite happened to
# have a clear pass -- it's frequently the exact date of whichever
# observation caught each parcel's true peak, and that date shifts with
# each season's own cloud pattern. Confirmed directly: 2021's row-crop
# peak_doy distribution clusters at day 185 -- July 4, 2021, an actual
# fetched date -- while 2026's clusters at day 195 -- July 14, 2026,
# also an actual fetched date. A fixed absolute day threshold tuned on one
# year's specific observation calendar doesn't generalize to a different
# year's different calendar.
#
# Fix: use the *percentile* of the validation season's row-crop
# population that threshold 186 corresponded to (31.7%, not the 41.6%
# true CDL corn fraction -- accuracy-maximizing thresholds needn't
# preserve the marginal class split), and apply that percentile to each
# season's *own* peak_doy distribution at runtime instead of a fixed
# absolute day. This is the assumption that actually needs to hold for
# year-to-year transfer to work: not that corn peaks on the same calendar
# day every year, but that corn's peak-timing *rank* within that season's
# row-crop population is stable -- consistent with corn's real agronomic
# earlier-peak relationship to soybean, which is about relative timing,
# not an absolute date. Applying this to 2026 gives 43 of 150 corn-like --
# far more plausible than the absolute version's 2, and consistent with
# CDL 2021's real ~42% corn share.
CORN_SOYBEAN_PEAK_DOY_PERCENTILE = 31.7

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


def cluster(feats, non_row_crop_early_ndvi=NON_ROW_CROP_EARLY_NDVI,
            corn_soybean_peak_doy_percentile=CORN_SOYBEAN_PEAK_DOY_PERCENTILE):
    """Two deterministic threshold splits, not KMeans: non-row-crop vs.
    row-crop on early_ndvi (an absolute level), then corn-like vs.
    soybean-like on peak_doy within row-crop -- but the peak_doy split is
    a *percentile* of this call's own row-crop population, not a fixed
    calendar day. See CORN_SOYBEAN_PEAK_DOY_PERCENTILE's comment for why:
    an absolute day-of-year threshold, cross-validated against real CDL
    ground truth, still failed to transfer from its validation season to a
    different one, because which day captures each parcel's true peak
    shifts with each season's own cloud pattern. The percentile is what
    was actually validated to hold across a season change; the absolute
    day was not.

    This replaces an earlier KMeans-based version of the row-crop split,
    which was a different algorithm from what was actually validated in
    the first place: KMeans groups parcels by overall curve shape across 5
    features, then label_cluster() named the resulting *cluster means* --
    but a cluster's mean peak_doy landing on one side of a threshold says
    nothing about where each individual member's own peak_doy falls, so
    parcels could get bulk-labeled against the very rule that was
    cross-validated per-parcel. That KMeans stage was also the fix for a
    real, earlier bug (a single joint fit across the whole population
    picked k=2 and found zero corn-like parcels at county scale, since the
    early_ndvi outlier split dominated silhouette over the subtler
    peak-timing one) -- but a deterministic threshold sidesteps that
    failure mode too, while actually matching the validated method.

    Confidence for both decisions comes from the same margin-based
    mechanism: how far a parcel's own value sits from its threshold
    relative to the population's spread, clipped to [0.5, 1.0].
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
    # peak_doy is heavily quantized -- many parcels' PCHIP peak lands
    # exactly on whichever calendar date happened to catch their true
    # peak, so large blocks of parcels can share the exact same value (one
    # county-wide run found 1,957 of ~7,900 row-crop parcels sharing a
    # single peak_doy). That block alone spans ~24% of the population --
    # bigger than the gap between the achievable splits on either side of
    # it (~21% excluding it entirely vs. ~45% including it whole), so no
    # tie-handling rule based on peak_doy *alone* can land near the target
    # percentile when the target happens to fall inside a block that
    # large: rounding the whole tied block one way or the other is the
    # only two options peak_doy alone offers.
    #
    # Broken by green_up_rate as a secondary sort key within ties: corn's
    # real, validated signal is a faster green-up, not merely an earlier
    # peak (confirmed against CDL 2021 ground truth -- corn's mean
    # green_up_rate is 0.026 vs. soybean's 0.021), so ranking a tied block
    # by descending green_up_rate and taking however many of its members
    # are needed to hit the target percentile is a principled way to use
    # information the model already trusts elsewhere, not an arbitrary
    # tie-break. (The tie-breaking power specifically at the exact
    # boundary value couldn't be directly confirmed against CDL --  too
    # few ground-truth parcels fell in that narrow window -- so this
    # extends a validated *population-level* relationship into an
    # unvalidated but well-motivated regime, not a fully proven claim.)
    ordered = row_crop.sort_values(["peak_doy", "green_up_rate"], ascending=[True, False])
    n_corn = round(len(row_crop) * corn_soybean_peak_doy_percentile / 100)
    is_corn = row_crop.index.isin(ordered.index[:n_corn])
    row_crop["cluster"] = np.where(is_corn, 1, 2)

    # Confidence as distance from the decision boundary in *rank* space
    # (position in the same peak_doy/green_up_rate order that actually
    # decided the split), not a margin on raw peak_doy: peak_doy alone is
    # so heavily tied (large blocks of parcels sharing one exact value --
    # see above) that a peak_doy-based margin collapsed to only 3-5
    # distinct confidence values across thousands of row-crop parcels,
    # found by a direct report that clicking ~30 parcels only ever showed
    # 50%, 76%, or 100%. Confidence should read as "how far into its own
    # side of the split is this parcel," and rank position -- effectively
    # unique per parcel once green_up_rate (a continuous value) breaks
    # peak_doy ties -- actually varies parcel to parcel where the raw
    # value didn't.
    position = pd.Series(range(len(ordered)), index=ordered.index)
    boundary = n_corn - 0.5  # midpoint between the last corn rank and first soybean rank
    max_dist = max(boundary, len(row_crop) - 1 - boundary)
    rank_margin = (position.reindex(row_crop.index) - boundary).abs() / (max_dist if max_dist > 0 else 1)
    row_crop["confidence"] = np.clip(0.5 + 0.5 * rank_margin, 0.5, 1.0)

    peak_doy_threshold = row_crop.loc[is_corn, "peak_doy"].max() if is_corn.any() else row_crop["peak_doy"].min()
    print(f"Non-row-crop split (early_ndvi > {non_row_crop_early_ndvi}): "
          f"{len(non_row_crop)} non-row-crop, {len(row_crop)} row-crop "
          f"-> corn/soybean split (target {corn_soybean_peak_doy_percentile}th percentile, "
          f"realized boundary peak_doy~{peak_doy_threshold:.0f}): "
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
