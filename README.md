# NDVI Field Monitoring — Python + PostGIS

A small, complete pipeline for detecting crop stress: pull Sentinel-2 imagery,
compute NDVI, intersect it with field boundaries, and flag fields whose NDVI
is anomalous relative to their spatial neighbors — the same shape of problem
as continental-scale agricultural monitoring, run here on one county so the
whole thing is reviewable in one sitting.

![Persistently anomalous field vs. its neighbors](reports/anomaly_example.png)

The field above sits flat around -0.2 to -0.3 NDVI across three months while
its eight nearest neighbors follow the expected green-up curve
(0.50 → 0.66 → 0.67). That flat, never-greens-up shape is water or bare
ground, not a struggling crop — real crop stress still tracks the season,
just below its neighbors. This is one output of `src/anomaly.py`; see
[Results](#results) for the rest.

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
