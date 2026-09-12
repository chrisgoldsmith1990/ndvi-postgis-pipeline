"""USDA Cropland Data Layer (CDL) -- real, per-pixel classified crop-type
labels, used as ground truth to validate the NDVI/EVI2 curve-shape
classifier against actual known corn/soybean fields (not just plausibility
arguments like acreage matching or cluster silhouette).

USDA/GMU's own CropScape REST API (nassgeodata.gmu.edu) has a genuinely
expired SSL certificate as of this writing (verified directly, not
assumed) -- routed around entirely rather than disabling verification,
since Microsoft's Planetary Computer already mirrors CDL as a proper STAC
collection (`usda-cdl`), using the exact same SAS-token-signing +
download-then-open pattern already proven in fetch_hls.py for the same
reason (GDAL's streaming driver hangs on Azure SAS URLs).

That mirror is stale -- it stops at 2021, with nothing for more recent
years. Doesn't matter for this project's purpose: validating the
curve-shape classification *method* against real labels doesn't need the
most recent year, since corn/soybean growth physics don't change year to
year, only which specific field grows which crop (driven by rotation).
2021 is simply the most recent year this mirror actually has.
"""

import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from pystac_client import Client

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "usda-cdl"
SAS_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}"

DOWNLOAD_CACHE = Path(__file__).resolve().parent.parent / "data" / "cdl_cache"

# Standard CDL codes (USDA NASS) for the two crops this project classifies.
CDL_CORN = 1
CDL_SOYBEAN = 5


def sign_href(href):
    """Same pattern as fetch_hls.py's sign_href -- SAS-sign a Planetary
    Computer blob URL via a plain GET to its token endpoint."""
    parsed = urlparse(href)
    account = parsed.netloc.split(".")[0]
    container = parsed.path.lstrip("/").split("/")[0]
    resp = requests.get(SAS_TOKEN_URL.format(account=account, container=container), timeout=15)
    resp.raise_for_status()
    return f"{href}?{resp.json()['token']}"


def _download_tile(item, request_timeout):
    out_path = DOWNLOAD_CACHE / f"{item.id}_cropland.tif"
    if out_path.exists():
        return out_path
    signed = sign_href(item.assets["cropland"].href)
    resp = requests.get(signed, timeout=request_timeout, stream=True)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    print(f"Downloaded CDL cropland tile: {out_path}", flush=True)
    return out_path


def fetch_cdl_raster(bbox, year, request_timeout=60):
    """Downloads the classified cropland raster (not the 'cultivated' or
    'frequency' layers -- see the usda-cdl collection's item types) for the
    given bbox/year, full download via requests (not GDAL streaming --
    same Azure SAS hang as fetch_hls.py), and returns the local path.

    Single-tile only -- use fetch_cdl_mosaic for a bbox (e.g. the whole
    county) that spans more than one CDL tile."""
    catalog = Client.open(STAC_URL)
    search = catalog.search(collections=[COLLECTION], bbox=bbox, datetime=f"{year}-01-01/{year}-12-31")
    items = [i for i in search.items() if i.properties.get("usda_cdl:type") == "cropland"]
    if not items:
        raise ValueError(f"No CDL cropland item found for {year} at bbox {bbox}")
    return _download_tile(items[0], request_timeout)


def fetch_cdl_mosaic(bbox, year, request_timeout=60):
    """Like fetch_cdl_raster, but downloads and mosaics *every* CDL tile
    intersecting bbox -- McLean County itself straddles a CDL tile
    boundary (checked directly: 2 tiles cover the county bbox, not 1), so
    a single-tile fetch would silently miss part of the county."""
    import rasterio
    from rasterio.merge import merge

    catalog = Client.open(STAC_URL)
    search = catalog.search(collections=[COLLECTION], bbox=bbox, datetime=f"{year}-01-01/{year}-12-31")
    items = [i for i in search.items() if i.properties.get("usda_cdl:type") == "cropland"]
    if not items:
        raise ValueError(f"No CDL cropland item found for {year} at bbox {bbox}")

    tile_paths = [_download_tile(item, request_timeout) for item in items]
    out_path = DOWNLOAD_CACHE / f"mosaic_{year}_{len(tile_paths)}tiles.tif"
    if out_path.exists():
        return out_path

    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = merge(srcs)
        profile = srcs[0].profile.copy()
    finally:
        for src in srcs:
            src.close()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)
    print(f"Wrote CDL mosaic ({len(tile_paths)} tiles): {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    import sys

    year = sys.argv[1] if len(sys.argv) > 1 else "2021"
    bbox = (-88.65, 40.65, -88.55, 40.72)  # SUBSET_BBOX, from fetch_timeseries.py
    path = fetch_cdl_raster(bbox, year)
    print(f"CDL raster at {path}", flush=True)
