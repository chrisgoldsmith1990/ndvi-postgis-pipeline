# NDVI Field Monitoring — Python + PostGIS

A small, complete pipeline for detecting crop stress: pull Sentinel-2 imagery,
compute NDVI, intersect it with field boundaries, and flag fields whose NDVI
is anomalous relative to their spatial neighbors — the same shape of problem
as continental-scale agricultural monitoring, run here on one county so the
whole thing is reviewable in one sitting.

## Why Python *and* PostGIS

Most geospatial pipelines are Python end-to-end: read raster, load vector
into geopandas, do everything in memory. That's fine at small scale, but it
throws away a tool that's actually a better fit for part of the problem.

This project draws the line deliberately:

- **Python (`rasterio`) owns the raster side.** Reading satellite bands and
  computing NDVI is array math over pixel grids — `rasterio`/`numpy` are the
  right tool, and there's no reason to reach for anything else.
- **PostGIS owns the vector and spatial-analysis side.** Polygon
  intersection, zonal statistics, and neighbor comparison are set-based
  spatial operations over relational data with a natural index
  (`GIST`). At one county this barely matters; at the scale this pipeline
  is modeling (thousands of fields, daily imagery, multi-state) it's the
  difference between a spatial join that uses an index and one that doesn't.
  Pushing this logic into the database rather than pulling every geometry
  into `geopandas` and looping in Python is the same call made for the
  original 20-state pipeline this project is a slice of.

The dividing line is: **arrays stay in Python, geometries stay in the
database.** Zonal statistics sits right on that boundary (see
[`src/zonal_stats.py`](src/zonal_stats.py)) — the README there explains
which side it landed on and why.

## Pipeline

| Step | What | Where |
|---|---|---|
| 1 | Pull Sentinel-2 imagery for the study area | [`src/fetch_imagery.py`](src/fetch_imagery.py) |
| 2 | Compute NDVI from red/NIR bands | [`src/compute_ndvi.py`](src/compute_ndvi.py) |
| 3 | Load field/parcel boundaries into PostGIS | [`src/load_boundaries.py`](src/load_boundaries.py) |
| 4 | Zonal statistics: mean NDVI per field per date | [`src/zonal_stats.py`](src/zonal_stats.py) |
| 5 | Neighbor comparison via `ST_Intersects` / KNN | [`src/neighbor_comparison.py`](src/neighbor_comparison.py) |
| 6 | Time series + anomaly flag | [`src/anomaly.py`](src/anomaly.py) |
| 7 | This README | — |

## Study area

TBD — one county, picked for Sentinel-2 coverage and public parcel/CLU
data availability.

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

## What NDVI measures, and why neighbor comparison matters

TBD once the pipeline is running against real imagery — this section is
where the domain reasoning goes: what the anomaly threshold means, why
comparing a field to its spatial neighbors controls for regional weather
rather than crop-specific stress, and what the resulting map/chart shows.
