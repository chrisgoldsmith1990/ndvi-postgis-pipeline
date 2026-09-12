# NDVI Field Monitoring — Python + PostGIS

A small, complete pipeline for detecting crop stress: pull Sentinel-2 imagery,
compute NDVI, intersect it with field boundaries, and flag fields whose NDVI
is anomalous relative to their spatial neighbors — the same shape of problem
as continental-scale agricultural monitoring, run here on one county so the
whole thing is reviewable in one sitting.

This is a batch pipeline run on demand, not a live service — there's no
backend, and the map on GitHub Pages is a static snapshot of one run's
output, not a live query. Making "pick any county on a map" work on demand
would mean solving nationwide parcel-data sourcing (every county runs its
own GIS portal, no unified free source) and standing up a real hosted
backend — a genuinely different, larger project than this one, which is
deliberately scoped to one county done well rather than a platform.

![McLean County field NDVI map, August 2026](reports/county_ndvi_map.png)

Every polygon is a real McLean County parcel, colored by its August mean
NDVI; the black-outlined ones are fields flagged as anomalous relative to
their spatial neighbors (see [Results](#results)). The blank area in the
middle is Bloomington-Normal — it drops out on its own because the >10-acre
parcel filter excludes town lots. An interactive version
(`reports/county_ndvi_map.html`, pan/zoom/click any parcel for its NDVI and
anomaly flag, satellite basemap toggle for ground-truthing) is generated
alongside this one by `src/visualize.py`.

Ground-truthing the flagged parcels against satellite imagery turns up
exactly what you'd expect from a >10-acre parcel filter with no land-use
attribute to screen on: golf courses, cemeteries, quarries, detention
ponds — and the Rivian plant in Normal (a multi-hundred-acre former auto
plant, now flat pavement and rooftop where the parcel data has no way to
know that isn't a field). None of these are pipeline bugs; they're the
correct output of "large parcel, doesn't green up like its neighbors,"
which just isn't the same claim as "cropland in distress." A production
version would cross-reference land-use/zoning data to exclude non-
agricultural parcels before flagging; this one surfaces them and leaves the
judgment call visible instead of hiding it.

![Persistently anomalous field vs. its neighbors](reports/anomaly_example.png)

The field above sits flat around -0.2 to -0.3 NDVI across three months while
its eight nearest neighbors follow the expected green-up curve
(0.50 → 0.66 → 0.67). That flat, never-greens-up shape is water or bare
ground, not a struggling crop — real crop stress still tracks the season,
just below its neighbors.

## Why Python *and* PostGIS

Most geospatial pipelines are Python end-to-end: read raster, load vector
into geopandas, do everything in memory. That's fine at small scale, but it
throws away a tool that's actually a better fit for part of the problem.

This project draws the line deliberately:

- **Python (`rasterio`) owns the raster side.** Reading satellite bands and
  computing NDVI is array math over pixel grids — `rasterio`/`numpy` are the
  right tool, and there's no reason to reach for anything else.
- **PostGIS owns the vector and spatial-analysis side.** Polygon storage,
  neighbor comparison, and the resulting per-field statistics are set-based
  operations over relational data with a natural index (`GIST`). At one
  county this barely matters; at the scale this pipeline is modeling
  (thousands of fields, daily imagery, multi-state) it's the difference
  between a spatial join that uses an index and one that doesn't. Pushing
  this logic into the database rather than pulling every geometry into
  `geopandas` and looping in Python is the same call made for the original
  20-state pipeline this project is a slice of.

The dividing line is: **arrays stay in Python, geometries stay in the
database.** Zonal statistics (step 4) sits right on that boundary — the
raster stays in Python because rasterizing polygons against a pixel grid
needs the array, but the *result* (one NDVI scalar per field per date) is a
tabular fact and gets written back to PostGIS immediately, because the next
step needs to join it against spatial neighbors. See
[`src/zonal_stats.py`](src/zonal_stats.py) for the full reasoning.

Step 5 (neighbor comparison) is where the PostGIS side does the most real
work: finding each field's nearest neighbors uses the idiomatic
GIST-accelerated KNN pattern (`LATERAL` join, `ORDER BY geometry <-> geometry
LIMIT k`) — confirmed via `EXPLAIN` to hit an Index Scan, not a sequential
scan, over 9,578 parcels. Doing the equivalent in geopandas means either the
full O(n²) pairwise distance matrix or hand-rolling a KD-tree; in PostGIS
it's the query planner's job and a five-line query.

## Pipeline

| Step | What | Where |
|---|---|---|
| 1 | Pull + composite Sentinel-2 imagery for the study area | [`src/fetch_imagery.py`](src/fetch_imagery.py) |
| 2 | Compute NDVI from red/NIR bands | [`src/compute_ndvi.py`](src/compute_ndvi.py) |
| 3 | Load parcel boundaries into PostGIS | [`src/load_boundaries.py`](src/load_boundaries.py) |
| 4 | Zonal statistics: mean NDVI per field per date | [`src/zonal_stats.py`](src/zonal_stats.py) |
| 5 | Neighbor comparison via GIST-accelerated KNN (`<->`) | [`src/neighbor_comparison.py`](src/neighbor_comparison.py) |
| 6 | Time series + anomaly flag | [`src/anomaly.py`](src/anomaly.py) |
| 7 | This README | — |
| — | County map (static + interactive), the payoff of 4-6 | [`src/visualize.py`](src/visualize.py) |
| — | Crop-type curve-shape clustering (subset area) | [`src/fetch_timeseries.py`](src/fetch_timeseries.py), [`src/fetch_hls.py`](src/fetch_hls.py), [`src/clip_parcels.py`](src/clip_parcels.py), [`src/crop_clusters.py`](src/crop_clusters.py), [`src/visualize_subset.py`](src/visualize_subset.py) — see [below](#crop-type-exploration-subset-area) |

## Study area

**McLean County, IL** (Bloomington-Normal) — corn/soybean country, chosen for
Sentinel-2 coverage and a public county parcel dataset. It also happens to
straddle four Sentinel-2 MGRS tiles, which turned into the most useful
accident of the project (see [Real-world gotchas](#real-world-gotchas)).

Field boundaries come from McLean County's public parcel layer (McLean
County GIS Consortium, CC BY 4.0), filtered to parcels over 10 acres —
~9,578 of the county's ~71,000 parcels — to approximate field-scale
agricultural land rather than every residential lot in Bloomington-Normal.
True USDA CLU (Common Land Unit) boundaries aren't publicly redistributable;
they require an FSA data request tied to a specific use case. Parcel data is
the practical public substitute, and this particular layer only carries a
parcel ID and acreage — no owner name or address — which keeps it
comfortably in open-data territory.

## Setup

```bash
# Python environment (conda-forge — geopandas/rasterio's GDAL deps are much
# less painful this way than plain pip on Windows)
conda env create -f environment.yml
conda activate ndvi-postgis

# PostGIS, via Docker
docker compose up -d
cp .env.example .env
```

Run the pipeline (each step takes an optional date label; `2026-06`,
`2026-07`, `2026-08` are the three months already loaded):

```bash
python -m src.fetch_imagery "2026-08-01/2026-08-31" 2026-08
python -m src.compute_ndvi 2026-08
python -m src.load_boundaries        # once — parcels don't change month to month
python -m src.zonal_stats 2026-08
python -m src.neighbor_comparison 2026-08
python -m src.anomaly
python -m src.visualize 2026-08   # writes reports/county_ndvi_map.{png,html}
```

## What NDVI measures, and why neighbor comparison matters

NDVI = (NIR − Red) / (NIR + Red). Healthy, chlorophyll-rich canopy absorbs
red light for photosynthesis and strongly reflects near-infrared, so dense
vegetation reads close to +1; bare soil, water, and stressed or absent
canopy read near 0 or negative.

An absolute NDVI threshold doesn't work as an anomaly signal because the
county's NDVI floor moves with the season and the weather — corn at
knee-high in June reads nothing like corn at full canopy in August, and a
regional dry spell suppresses NDVI everywhere at once. "This field is below
0.5" is ambiguous; it could mean stressed crop, or it could mean every field
in the county is at 0.5 that week. **Comparing a field to its immediate
spatial neighbors cancels out the shared regional signal** — weather, planting
date for the area, satellite viewing geometry — and leaves the field-specific
difference, which is the thing actually worth flagging.

## Results

Three monthly composites (June/July/August 2026) over 9,578 parcels:

| Month | Avg. field NDVI | Avg. neighbor-mean NDVI |
|---|---|---|
| 2026-06 | 0.628 | 0.627 |
| 2026-07 | 0.745 | 0.747 |
| 2026-08 | 0.860 | 0.862 |

That progression is the expected corn/soybean canopy-closure curve, and
field vs. neighbor-mean track within 0.002 of each other county-wide — the
neighbor baseline is doing its job as a shared reference, not just adding
noise.

**Anomaly flag:** z-score each field's deviation from its neighbor mean
(`ndvi_diff`) against the distribution of that same deviation across all
fields *on that date*, and flag `z < -2`. A single flagged date is treated
as a one-off (cloud contamination, a registration nudge between mosaic
tiles, a field cut that week); a field flagged on **every** date observed is
reported as a persistent anomaly — the stronger, still-simple signal.
**41 of 9,566 fields (0.4%)** are persistent anomalies across the three
months. The chart at the top of this README is the strongest one
(avg. z = −7.4).

## Real-world gotchas

Two things showed up building this against actual imagery that a
single-scene, single-tile toy version wouldn't have surfaced, and both
seemed more useful to fix properly and document than to route around:

- **A county doesn't fit in one tile.** McLean County straddles four
  Sentinel-2 MGRS tiles (16TBK/BL/CK/CL). Picking "the least-cloudy single
  scene" silently produced a mosaic covering less than half the county —
  the fix is mosaicking all four tiles for a date, not picking one scene.
- **A tile "existing" for a date doesn't mean it has full data over the
  AOI.** Individual Sentinel-2 passes can have large nodata gaps from swath
  edges — `s2:nodata_pixel_percentage` on one tile swung from 0% to 78%
  across a few weeks. Requiring a single date with all four tiles clean
  turned out to not exist for most of the growing season. The fix is a
  standard best-pixel composite: rank candidate acquisitions per tile by
  nodata%, then cloud%, and mosaic best-first so gaps in the top choice get
  filled from the next-best acquisition of that same tile.

Separately, the real county parcel data included 5 self-intersecting
geometries out of 9,578 — `ST_MakeValid` runs automatically after load so
`ST_Intersects`/KNN don't choke on them downstream.

## Crop-type exploration (subset area)

A follow-on question from the anomaly work: can NDVI curve *shape* — not
just level — distinguish what's actually growing in a field? Corn and
soybean have different phenology (corn greens up faster and peaks earlier;
soybean climbs more gradually and peaks later), which is the same signal
USDA's own Cropland Data Layer is built on. Scoped to a small rural subset
(122–163 parcels, depending on date coverage) rather than the full county,
since this needs every available date across the season, not monthly
composites — [`src/fetch_timeseries.py`](src/fetch_timeseries.py) and
[`src/fetch_hls.py`](src/fetch_hls.py) for imagery,
[`src/clip_parcels.py`](src/clip_parcels.py) for the road/waterway
correction below, [`src/crop_clusters.py`](src/crop_clusters.py) for the
clustering, [`src/visualize_subset.py`](src/visualize_subset.py) for the map.

**[Interactive map](https://chrisgoldsmith1990.github.io/ndvi-postgis-pipeline/reports/subset_crop_map.html)** —
tap any parcel for its NDVI curve, behavioral cluster, and an assignment
confidence.

Only 8 of 41 candidate Sentinel-2 passes were clear enough to use at first,
leaving a 45-day blind gap (May 9 → June 23) across almost the entire
green-up transition — enough to say *when* a field peaked, but not whether
it got there in a burst or a steady climb. Loosening the cloud threshold
recovered two dates inside that gap (and caught one bad one: a scene that
passed the pixel-level cloud check but showed whole-scene residual haze —
excluded explicitly, see `crop_clusters.BAD_DATES`). With real data inside
the gap, the shape difference is stark and quantitative: from an
almost-identical starting point in mid-May, the corn-like cluster reaches
97% of its season peak by June 23, while the soybean-like cluster has only
reached 67% of its (higher) peak by the same date and keeps climbing for
another two months. Clusters are unsupervised and reported as behavioral
groups, not validated crop labels — no ground truth exists for this
subset. (USDA's CDL would be the obvious cross-check, but it lags a full
year behind the crop it describes — the newest available CDL right now is
for last year, not this year — and corn/soybean rotation means a
year-old label can be wrong for the current season on any given field.
That's a real constraint on validating this, not an oversight.)

**Densifying with a second sensor** ([`src/fetch_hls.py`](src/fetch_hls.py)) —
Sentinel-2 alone only cleared 8 usable dates out of 41 candidate passes this
season. NASA's Harmonized Landsat Sentinel-2 (HLS) product is built
specifically to be numerically comparable to Sentinel-2 surface
reflectance and shares its MGRS tiling, so NDVI from either sensor slots
into the same time series without rescaling. Adding HLS-L30 (Landsat)
recovered 8 more usable dates — nearly doubling the season to 18 —
including one right inside what was previously a blind gap, and two
adjacent-day cross-sensor pairs that agree almost exactly (Aug 22
Sentinel-2 vs. Aug 23 Landsat: 0.895 vs. 0.894 mean NDVI).

Getting there required diagnosing a real, non-obvious bug: `rasterio`/GDAL's
generic HTTP streaming driver hangs indefinitely on Azure Blob Storage's
SAS-token-authenticated URLs (Planetary Computer's access method for HLS),
ignoring configured timeouts — confirmed by isolating each step (a plain
STAC search and a plain token-signing request both complete in under a
second on their own; only `rasterio.open()` on the signed URL itself
hangs). The fix: download each band fully via plain `requests` first, then
open the local file with `rasterio`, sidestepping GDAL's streaming path
entirely. Also worth noting: HLS surface reflectance occasionally produces
individual pixels with NDVI outside the theoretically valid [-1, 1] range
(atmospheric-correction artifacts at dark/shadow-edge pixels) — `compute_ndvi.py`
now clips to that range, though empirically it never changed a parcel's
zonal mean for this dataset; the artifact pixels are too few to move an
average of hundreds.

Honest result of the density increase: real curve texture is visible now
that wasn't with 8-10 sparse points (a shared dip across all three
clusters around day 165-170, likely an actual short-term weather effect,
not a fitting artifact) — the corn/soybean early-vs-late-peak story still
holds directionally. But the aggregate silhouette score actually *dropped*
slightly (0.385 → 0.338) rather than improving. More temporal resolution
made the picture more detailed, not more separable — combining two
sensors' worth of real day-to-day variation adds genuine texture that a
clean 3-cluster model doesn't perfectly capture. Reported as-is rather
than only reporting the version that looked best.

**Yield ranking** ([`src/yield_ranking.py`](src/yield_ranking.py)) —
ranks each parcel's season-integrated NDVI against others in its own
crop-type cluster, the same neighbor-relative logic the anomaly detector
uses, applied to a crop-type peer group instead of a spatial one. An
approximate bu/ac rate is also computed, anchored to McLean County's
actual 2025 NASS yield (243.1 bu/ac corn, 73.95 bu/ac soybean) and scaled
by real published models' coefficient of variation (Johnson et al. 2021,
*Remote Sens.* 13(21):4227 — Illinois-level accumulated-NDVI R²=0.91 for
corn, only R²=0.54 for soybean) — but the rate alone isn't the estimate
that matters, since parcels in this subset range from a dozen to well
over a hundred acres. The actual reported estimate is **total bushels**:
that rate multiplied by the parcel's *clipped* acreage (see below — the
crop-growing area with roads/waterways subtracted out, not the county's
raw deeded acreage). This is explicitly an approximation — it borrows a
cited relationship's *spread*, not a fitted equation specific to this
subset — and should be trusted much less for the soybean-like majority of
parcels than for the corn-like ones, per the source paper's own finding.

**Road/waterway clipping** ([`src/clip_parcels.py`](src/clip_parcels.py)) —
a parcel's tax-boundary polygon can include a road or stream running
through the field itself, and those pixels read as pavement or water, not
crop, dragging the zonal-stats NDVI mean along with them. Real, not
hypothetical: the subset area has 44 actual OpenStreetMap road/waterway
segments crossing its ~163 parcels. Fetches them from OSM's public
Overpass API, buffers each by an approximate right-of-way/channel width
(class-based assumptions — OSM doesn't carry real width for rural roads),
and subtracts the union from every parcel via PostGIS `ST_Difference` into
a `parcels_clipped` table — the same "PostGIS does the spatial-set work"
split used everywhere else here.

The effect is real and physically sensible, confirmed by comparing
clipped vs. unclipped NDVI directly: in spring (bare soil), clipping
*lowers* the mean slightly — removing already-green ditch-bank vegetation
from an otherwise bare field; in summer (mature canopy), clipping
*raises* it — removing low-NDVI pavement from a green field. One small
parcel where a road/stream eats 7.5% of its area shows up to a 0.049 NDVI
shift depending on date. But rerunning the full clustering pipeline on
clipped vs. unclipped data reassigns only 2 of 125 parcels and leaves
silhouette/confidence essentially unchanged — the correction matters for
per-parcel precision (and now feeds the yield estimate's acreage), not
for the aggregate corn/soybean story, which turns out to be robust to it.
`ndvi_zonal_stats_subset_clipped` is now the default the crop-type
pipeline reads from.

## Scaling the crop-type pipeline to all of McLean County

Everything above ran on a ~163-parcel subset to keep per-acquisition-date
fetching cheap. Running the same pipeline — imagery, clipping, clustering,
yield ranking — against all **9,571** county parcels (fetch scope: all
four Sentinel-2/HLS MGRS tiles covering the county, not one) surfaced
three real bugs that the subset was too small and too uniform to expose.

**[Interactive map](https://chrisgoldsmith1990.github.io/ndvi-postgis-pipeline/reports/county_crop_map.html)** —
same per-parcel click popup as the subset map (NDVI curve, cluster label,
assignment confidence, yield estimate), scaled to all 9,571 county
parcels. An earlier version of this section assumed that would balloon to
several hundred MB and shipped a static PNG instead; measured directly,
it's ~34MB — large, but this codebase already had its own precedent that
size range works fine (`visualize.py`'s county-wide anomaly map embeds a
full per-parcel NDVI series for the same ~9,571 parcels at ~7MB), so the
estimate was revised rather than trusted unchecked.
`src/visualize_county_crops.py` still also renders the plain static
choropleth below, as a lightweight fallback for the README itself.

**Bug 1 — whole-AOI cloud rejection discarded far more than it should have.**
`fetch_timeseries.py`'s per-date cloud check averaged bad-pixel fraction
across the *entire* bounding box before deciding whether to keep that
date at all. At subset scale (an 8km box) that's a reasonable proxy — a
cloud covering over a third of it plausibly means the whole small area is
compromised. At county scale (68×53km) it's wrong: a cloud sitting over
one third of the county could push the whole-AOI average over threshold
and discard *every* parcel's data for that date, including parcels sitting
in perfectly clear sky on the other side of the county. Caught by a direct
question about exactly this ("are we throwing out the whole county when we
hit that threshold?") rather than found by inspection. Fixed by raising
the whole-AOI threshold to 0.9 (reject only near-total cloud cover) for
county-scale runs and letting the existing per-pixel masking do the real
filtering — `rasterstats`' NaN-aware zonal averaging already excludes only
the pixels actually under cloud, per parcel. Result: 43 of 72 candidate
dates recovered, versus 11 of 72 before the fix.

**Bug 2 — a strict shared-date requirement no longer made sense once
rejection became per-pixel.** With the fix above, individual parcels
genuinely do end up with different numbers of usable dates now (per-date
valid coverage across the county ranged from ~2,000 to ~9,565 of 9,571
parcels, depending on where that day's clouds happened to sit) —
`crop_clusters.fit_splines()` used to require every parcel share the exact
same global date list (`.dropna()` on a complete-case pivot), which was
fine when a whole-AOI check meant every date either cleared for everyone
or nobody. Fixed by fitting each parcel's PCHIP curve on its own valid
dates instead, excluding a parcel outright only if it has fewer than 6
valid dates or less than 60 days of season span — rather than requiring
one shared date set for all 9,571 parcels, which would mean dropping
almost every parcel or almost every date. (Checked first that this
wasn't hiding a coverage gap: every kept parcel's own date range actually
spans April–September, not just some narrow mid-season window — the
60-day-span floor isn't doing the real work here, per-parcel valid-date
count is.)

**Bug 3 — one joint clustering pass couldn't find both real splits at
county scale.** The first county-wide run produced two clusters and
identified *zero* parcels as corn-like — implausible for a county that's
close to half corn by actual harvested acreage (318,000 corn vs. 294,000
soybean acres, 2025 NASS). The cause: silhouette-based k-selection over
the whole population picked k=2, and the strong, high-variance
already-green-in-April (non-row-crop) split dominated that choice over
the comparatively subtler corn/soybean peak-timing split, lumping every
row-crop parcel into one undifferentiated group. Confirmed by re-running
KMeans on just the row-crop parcels in isolation, which cleanly recovered
two peak-timing clusters (~day 199 and ~day 232 mean peak — consistent
with corn peaking first). Fixed by splitting the clustering into two
explicit stages: a deterministic `early_ndvi` threshold pulls out
non-row-crop first (the same rule the cluster-labeling step already used
to *name* clusters after the fact — now it also performs the split, not
just the naming), then only the remaining row-crop parcels go through
KMeans to find the corn/soybean split. An earlier attempt used a forced
k=2 KMeans fit to *find* the non-row-crop split instead of a fixed
threshold; that fixed the county run but split on an unrelated axis
entirely at the smaller subset scale, leaving almost nothing for the
second stage — the deterministic threshold is what actually generalizes
across both scales.

**Result:** 2,500 corn-like, 5,386 soybean-like, 1,372 non-row-crop, out of
9,258 clustered parcels. (This is the CDL-validated EVI2 + percentile
classifier's split, documented further down — see "Validating against
real ground truth" below. It superseded an earlier NDVI-based figure of
3,128/4,734/1,396 reported when this section was first written; the
county's real corn/soybean acreage is close to even, but the validated
classifier's own split need not exactly match acreage, since it's
optimized for per-parcel accuracy against real labels, not for matching
the county-wide total.)

![McLean County crop-type clusters, full county](reports/county_crop_map.png)

### Cleaning up the interactive map: outliers, urban parcels, and curve smoothing

Once the interactive version of this map was live (matching
visualize_subset.py's per-parcel popup, at full county scale), three
things stood out that the static PNG hadn't made visible: sharp,
single-date spikes/troughs in some popup curves, real farmland-only
clustering being computed over parcels that are obviously not
farmland (inside Bloomington/Normal), and a general request for a
smoother-looking popup curve.

**Per-parcel outlier rejection.** Some popup curves showed a sharp jump up
or down on a single date that immediately reverted on the next — a
residual cloud-shadow or haze pixel that survived the per-pixel SCL/Fmask
mask on just that date, for just that one parcel. This is a different
problem from `BAD_DATES` (crop_clusters.py): that was one whole-scene bad
Sentinel-2 pass affecting every subset parcel on the same date, found and
excluded by hand. A per-parcel artifact like this can't be hand-curated
one date at a time at county scale, so `crop_clusters._reject_outliers`
now checks each parcel's own series for points that don't fit a straight
line drawn between their immediate (real, irregularly-spaced) neighbors,
and drops the ones that don't. A first version used a Hampel-filter-style
test instead (comparing each point to a small window of *rank-order*
neighbors) — checked against the actual effect on cluster sizes rather
than assumed safe, and it wasn't: it rejected ~16% of all points and
collapsed the subset's corn-like cluster from 45 parcels down to 23,
because comparing a point to nearby rank-order neighbors doesn't account
for how far apart they actually are in time, and corn's genuine signal —
a fast, sustained green-up — looks exactly like an "outlier" relative to
a small index-window after a real gap in the data. The local-straight-line
version doesn't have that problem (a real, fast, sustained change still
lands close to a line through its actual time-adjacent neighbors) and
rejects a much more plausible ~1-6% of points depending on scale.

**Two different curves, on purpose.** A genuine smoothing fit
(`scipy.interpolate.UnivariateSpline`, `s > 0`) was tried next, to make
the popup line itself look smoother rather than just removing outliers —
and even a small smoothing budget turned out to erode the sharp, narrow
peak that identifies a corn-like curve: on the subset, adding smoothing on
top of outlier rejection alone collapsed corn-like from 63 parcels down to
18, because a smoothing spline's squared-residual budget is cheaper to
spend flattening a real narrow peak than tracking noise among the many
flatter points around it — exactly backwards for a method whose entire
signal *is* peak shape. The fix: two separate fits from the same
outlier-cleaned points. `crop_clusters.fit_splines` still uses PCHIP
(exact interpolation) for the *analysis* curve that `extract_features`
reads peak timing/height and green-up/decline rates off of;
`popup_curve_data` fits its own separate, real smoothing spline purely for
the *display* line in the map popup, where softening a peak's exact shape
is a cosmetic cost, not a correctness one. Individual raw-date dots are no
longer drawn on that display curve either — with a genuine smoothing fit,
a dot at each raw date would show the real (slightly noisy) measurement
sitting visibly off the smoothed line, which reads as the fit being wrong
rather than as the point being the ordinary noise it's smoothing past. The
popup now shows the count of valid dates (`N = ...`) as plain text instead.

**Excluding Bloomington/Normal.** The >10-acre parcel filter and
NDVI-based clustering have no notion of land use or zoning, so a handful
of parcels inside the two incorporated cities (parks, cemeteries,
industrial/commercial tracts) were passing every geometric/spectral test a
real field would and showing up colored on the map — the same
non-agricultural-parcel limitation already called out for the anomaly map
at the top of this README, now visibly a problem for the crop-type map
too. `src/exclude_urban.py` fetches Bloomington and Normal's real
municipal boundary polygons from OpenStreetMap via Nominatim's
`polygon_geojson` search (reassembling an administrative-boundary relation
from raw Overpass output by hand is a known headache; Nominatim already
does that assembly and hands back a clean polygon for a named place), and
excludes any parcel whose *centroid* falls inside either boundary — not
`ST_Intersects`, which would also exclude legitimate rural parcels merely
sharing a boundary edge with the city limit — from the clustering input
entirely, not just hidden from the rendered map afterward. 307 of 9,571
county parcels were inside Bloomington/Normal and are excluded this way.

**Not yet done:** county-wide NASA HLS (Landsat) densification.
`src/fetch_hls.py` supports a `county` mode (built and used to densify the
subset's Sentinel-2 series from 10 to 18 dates), but it hasn't been run at
county scale — the county-wide curves above are Sentinel-2 only (43
dates). Denser per-parcel coverage would likely reduce reliance on the
outlier filter above rather than replace the need for it, since the
underlying artifact (a residual cloud/shadow pixel slipping past masking
on one date) isn't sensor-specific.

### Validating against real ground truth: USDA's Cropland Data Layer

Every result up to this point was validated only by proxy: cluster
silhouette, KMeans assignment confidence, whether the corn/soybean split
matched the county's real *acreage* — never checked against actual known
crop labels for actual fields. USDA's Cropland Data Layer (CDL) is exactly
that: a real, per-pixel (30m) classified crop-type product covering the
whole country. `src/validate_against_cdl.py` uses it to properly validate
(and tune) the classification, and to finally settle the NDVI-vs-EVI2
question this project explored earlier with real accuracy numbers instead
of just proxy metrics.

**Why not just compare this season's map against the newest CDL?** CDL
for a season isn't published until after harvest, so the most recent one
available describes a *past* year — and Illinois corn/soybean farming is
dominated by annual rotation. A field that was corn last year is *expected*
to be soybean this year, not corn again, so a same-label match against
last year's CDL would mostly measure rotation, not classification
accuracy — a low match rate could mean the classifier is *right*, not
wrong. The fix: validate a past season's classification against that
*same* season's CDL — same year, same imagery, real ground truth, no
rotation ambiguity.

**Why day-of-year-capped.** The validation season's imagery is fetched
only up to the same day-of-year (246, Sept 3) the current 2026 season's
data actually reaches, so the validation never uses late-season
information the live pipeline wouldn't actually have partway through a
real season — otherwise it would be an unrealistically easy test.

**Why 2021.** Checked three free sources directly rather than assuming:
Planetary Computer's `usda-cdl` mirror's own declared metadata caps its
temporal extent at 2021-12-31 (not a query problem — genuinely not
mirrored past that year); USDA/GMU's own CropScape REST API
(nassgeodata.gmu.edu) has an expired SSL certificate as of this writing;
and USDA's direct-download portal returns a real `403 Forbidden` to
scripted requests. The only path to more recent years is Google Earth
Engine's CDL mirror, which needs a Google Cloud account/credentials this
project doesn't have. 2021 is what's actually available for free without
new credentials — and it's enough: corn/soybean growth *physics* don't
change year to year, only which specific field grows which crop (driven
by rotation), so a method validated on 2021 should generalize.

**Result.** Fetched 2021 Sentinel-2 + HLS for the subset area (16 dates,
capped at DOY 246) and CDL 2021 (67 Corn, 94 Soybean, 2 Other, via a
per-parcel majority-vote zonal join — the categorical equivalent of
`zonal_stats.py`'s mean-based reduction). Swept the `peak_doy`
classification threshold against those 161 CDL-labeled Corn/Soybean
parcels, 5-fold cross-validated so the reported accuracy isn't inflated by
picking the best of many thresholds against the same data being scored:

| | NDVI | EVI2 |
|---|---|---|
| Best threshold (every fold agreed) | 186 | 186 |
| Cross-validated accuracy | 80.1% ± 7.3% | **87.6% ± 5.1%** |
| Accuracy at the old hardcoded threshold (210) | 75.8% | 77.0% |

Both indices land on the *same* optimal threshold (186, not the
previously hardcoded 210, picked without any validation) — a genuine
phenological signal, not an artifact of one index's scale. EVI2 wins on
accuracy at that threshold regardless, consistent with the earlier
proxy-metric comparison (better silhouette, better KMeans confidence,
fewer detected per-parcel outliers) but now backed by real labels instead
of just those proxies. A 5-feature logistic regression (adding
`early_ndvi`, `peak_ndvi`, `green_up_rate`, `decline_rate` alongside
`peak_doy`) *underperformed* the single-threshold rule for both indices —
validation that this project's existing simple-rule architecture is the
right amount of complexity, not an oversimplification.

**A cross-validated threshold still failed to transfer — found by actually
deploying it.** Applying the validated absolute cutoff (186) unchanged to
the 2026 season identified only 2 of 150 row-crop parcels as corn-like,
against an expected 40-60. Root cause: `peak_doy` isn't a clean biological
measurement independent of when the satellite happened to have a clear
pass — it's frequently the exact date of whichever observation caught a
parcel's true peak, and that date shifts with each season's own cloud
pattern. Confirmed directly: 2021's row-crop `peak_doy` values cluster at
day 185 — July 4, 2021, an actual fetched date — while 2026's cluster at
day 195 — July 14, 2026, also an actual fetched date. A day-of-year
threshold tuned on one season's specific observation calendar doesn't
generalize to a different season's different one, no matter how carefully
it was cross-validated *within* that season.

**Fix: a percentile, not a day.** Threshold 186 corresponded to the 31.7th
percentile of 2021's own row-crop `peak_doy` distribution (not the CDL
ground truth's true ~42% corn share — an accuracy-maximizing threshold
needn't preserve the marginal class split). Using that *percentile*
against each season's own distribution, rather than the fixed day,
requires only that corn's peak-timing *rank* relative to soybean is
stable year to year — a weaker, more defensible assumption than "corn
peaks on the same calendar day every year," and one consistent with
corn's real agronomic earlier-peak relationship to soybean. Within one
season's own random cross-validation folds the percentile version scores
lower than the absolute-day version (75.1% vs. 87.6% for EVI2) —
expected, since a random split of one season has no calendar shift to
correct for, so the fixed day has a home-field advantage there that
doesn't exist across a real season change. The relevant test is the
cross-*season* one, where the absolute version failed outright.

**A second tie-handling problem, found the same way — by actually
deploying it at county scale.** The first county-wide run of the
percentile version produced only 20.7% corn against a 31.7% target.
Cause: `peak_doy`'s quantization (above) means large blocks of parcels can
share the *exact same* value — one run found 1,957 of ~7,900 county
row-crop parcels sharing a single peak_doy, a block spanning ~24% of the
population by itself, bigger than the gap between the two splits actually
achievable around it (~21% excluding the block entirely, ~45% including
it whole). `numpy.percentile` interpolates a boundary value and a strict
`<` comparison then dumps an entire tied block onto one side regardless of
size; no tie-handling rule based on peak_doy *alone* can land near a
target percentile that happens to fall inside a block that large. Fixed
by sorting ties within a shared peak_doy value by `green_up_rate`
(descending) as a secondary key: corn's validated signal is a faster
green-up, not merely an earlier peak (mean `green_up_rate` 0.026 for CDL
2021's real corn vs. 0.021 for soybean), so this extends an
already-validated population-level relationship to break ties, rather
than an arbitrary rule — though its power specifically at the exact
boundary value couldn't be directly confirmed against CDL, since too few
ground-truth parcels fell in that narrow window. This lands the split
within rounding of the target every time (32.0% on the subset, 31.7% on
the county, against a 31.7% target) rather than being at the mercy of
wherever the largest tied block happens to sit.

**A third quantization problem, reported directly by clicking around the
live map:** confidence values on the interactive map only ever showed 3
distinct numbers across dozens of clicked parcels. Same root cause as the
two problems above, showing up a third way: the popup's confidence was a
margin computed from raw `peak_doy` against the split threshold, and since
`peak_doy` collapses to a handful of distinct values across thousands of
parcels, so did that margin — checked directly, corn-like parcels had
only 5 distinct confidence values across 2,500 of them, soybean-like only
3 across 5,386 (non-row-crop, whose confidence comes from the continuous
`early_ndvi` level rather than `peak_doy`, correctly showed hundreds).
Fixed by computing confidence from *rank position* in the same
peak_doy/green_up_rate order that actually decides the split, rather than
raw peak_doy: distance from the boundary rank, normalized to [0.5, 1.0].
Since green_up_rate is continuous, this gives an effectively unique value
per parcel — every one of the 2,500 corn-like and 5,386 soybean-like
county parcels now has its own distinct confidence, instead of 5 and 3.

**Adopted:** `crop_clusters.py` now defaults to EVI2
(`evi2_zonal_stats_*` tables) with `CORN_SOYBEAN_PEAK_DOY_PERCENTILE =
31.7`, applied to each run's own row-crop `peak_doy` distribution (ties
broken by `green_up_rate`), for the crop-type pipeline specifically.
`visualize.py`'s separate anomaly-detection map is unaffected — this
finding is about distinguishing corn from soybean by curve shape, not
about field-health monitoring, and NDVI remains the right, more
field-tested choice there. The corn/soybean split no longer uses KMeans
at all (an earlier version did, then named the resulting cluster means)
— it's two deterministic threshold rules now, matching exactly what was
actually validated rather than approximating it through an unsupervised
intermediate step whose cluster boundaries weren't guaranteed to land on
the validated cutoff. The non-row-crop threshold (`NON_ROW_CROP_EARLY_NDVI`,
now 0.19 for EVI2's scale) stays an absolute value, not a percentile: it
measures an EVI2 *level* (how green a parcel is in early spring), not a calendar day, so
it isn't exposed to the same observation-date-sensitivity problem — but
it's also calibrated by percentile-matching to the old NDVI threshold's
split size rather than independently cross-validated the same way, since
CDL 2021 only had 2 true "Other" parcels in this subset, too few to
validate a threshold against (though both of those 2 parcels' values do
fall on the correct side of 0.19, weak corroboration rather than proof).
