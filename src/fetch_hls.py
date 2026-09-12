"""Fetch NASA's Harmonized Landsat Sentinel-2 (HLS-L30) imagery as an
additional temporal source, to densify the season sequence beyond
Sentinel-2 alone. HLS is explicitly built to be numerically comparable to
Sentinel-2 surface reflectance (that's what "harmonized" means), gridded
onto the same MGRS tiling -- so this subset needs the same multi-tile
mosaicking fetch_timeseries.py does for Sentinel-2, and the resulting
red.tif/nir.tif slot into compute_ndvi.py and the rest of the pipeline
completely unchanged.

GDAL's generic /vsicurl/ streaming driver hangs indefinitely on Azure Blob
Storage's SAS-token-authenticated URLs -- confirmed via isolated testing to
not be a network issue, not the planetary_computer package, and not
background vs. foreground execution (a plain STAC search and a plain SAS
token request both complete in under a second on their own; only
rasterio.open() on the signed blob URL itself hangs, ignoring
GDAL_HTTP_TIMEOUT). Works around it by downloading each band fully via
plain requests first, then opening the local file -- proven reliable
end to end against a real HLS asset.
"""

import socket
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import rasterio
import rasterio.warp
import requests
from pystac_client import Client
from rasterio.merge import merge

from src.fetch_timeseries import MAX_BAD_FRACTION, SUBSET_BBOX

socket.setdefaulttimeout(30)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "hls2-l30"
SAS_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}"

# Fmask bit flags (HLS v2.0): bit1=cloud, bit2=adjacent to cloud/shadow,
# bit3=cloud shadow. Snow (bit4) and water (bit5) are left unmasked --
# fetch_timeseries.py's SCL-based mask makes the same call for Sentinel-2.
FMASK_BAD_BITS = 0b00001110
HLS_FILL_VALUE = -9999

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DOWNLOAD_CACHE = Path(__file__).resolve().parent.parent / "data" / "hls_cache"


def sign_href(href):
    """SAS-sign a Planetary Computer blob URL via a plain GET to its token
    endpoint -- avoids the planetary_computer package entirely, which isn't
    needed for this (the earlier hang was in rasterio/GDAL, not signing)."""
    parsed = urlparse(href)
    account = parsed.netloc.split(".")[0]
    container = parsed.path.lstrip("/").split("/")[0]
    resp = requests.get(SAS_TOKEN_URL.format(account=account, container=container), timeout=15)
    resp.raise_for_status()
    return f"{href}?{resp.json()['token']}"


def download(href, out_path):
    """Full download via requests, not GDAL streaming -- see module
    docstring. Cached: HLS tiles get reused across dates that share a
    tile-band pair, and reruns of this script skip already-fetched files."""
    if out_path.exists():
        return out_path
    signed = sign_href(href)
    resp = requests.get(signed, timeout=60, stream=True)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    return out_path


def find_season_items(bbox=SUBSET_BBOX, datetime_range="2026-04-01/2026-09-11"):
    print("Searching HLS-L30 catalog for the season...", flush=True)
    catalog = Client.open(STAC_URL)
    search = catalog.search(collections=[COLLECTION], bbox=bbox, datetime=datetime_range)
    items = list(search.items())
    print(f"Found {len(items)} candidate HLS-L30 scenes", flush=True)

    by_date = defaultdict(list)
    for item in items:
        by_date[item.datetime.date()].append(item)
    return dict(sorted(by_date.items()))


def _mosaic_band(items, band_key, bbox):
    paths = [download(item.assets[band_key].href, DOWNLOAD_CACHE / f"{item.id}_{band_key}.tif")
             for item in items]
    srcs = [rasterio.open(p) for p in paths]
    try:
        dst_bounds = rasterio.warp.transform_bounds("EPSG:4326", srcs[0].crs, *bbox)
        mosaic, transform = merge(srcs, bounds=dst_bounds)
        crs = srcs[0].crs
    finally:
        for src in srcs:
            src.close()
    return mosaic[0], transform, crs


def build_date(items, date, bbox=SUBSET_BBOX):
    """Mosaic red/NIR/Fmask for one date, mask cloud/shadow via Fmask, and
    write red.tif/nir.tif -- same nodata-zeroing convention
    fetch_timeseries.py uses, so compute_ndvi.py needs no changes."""
    red, transform, crs = _mosaic_band(items, "B04", bbox)
    nir, _, _ = _mosaic_band(items, "B05", bbox)
    fmask, _, _ = _mosaic_band(items, "Fmask", bbox)

    bad_mask = (fmask.astype(np.uint8) & FMASK_BAD_BITS) != 0
    bad_fraction = bad_mask.mean()
    if bad_fraction > MAX_BAD_FRACTION:
        print(f"  {date}: {bad_fraction:.0%} cloud/shadow (Fmask) -- skipped", flush=True)
        return None

    red = red.astype("int32").copy()
    nir = nir.astype("int32").copy()
    combined_bad = bad_mask | (red == HLS_FILL_VALUE) | (nir == HLS_FILL_VALUE)
    red[combined_bad] = 0
    nir[combined_bad] = 0

    out_dir = RAW_DIR / str(date)
    if out_dir.exists() and any(out_dir.iterdir()):
        # Landsat's 16-day cycle is offset from Sentinel-2's, so a same-date
        # collision is unlikely in practice -- but silently overwriting a
        # different sensor's data under a shared date label would be a real
        # correctness bug if it ever happened, so refuse rather than guess.
        print(f"  {date}: raw dir already exists (from Sentinel-2) -- skipping to avoid overwrite", flush=True)
        return None

    profile = {"driver": "GTiff", "dtype": "int16", "count": 1,
               "height": red.shape[0], "width": red.shape[1], "crs": crs, "transform": transform}
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_dir / "red.tif", "w", **profile) as dst:
        dst.write(red.astype("int16"), 1)
    with rasterio.open(out_dir / "nir.tif", "w", **profile) as dst:
        dst.write(nir.astype("int16"), 1)
    print(f"  {date}: {1 - bad_fraction:.0%} clear -- wrote {out_dir}", flush=True)
    return out_dir


if __name__ == "__main__":
    by_date = find_season_items()
    written = []
    for date, items in by_date.items():
        result = build_date(items, date)
        if result is not None:
            written.append(str(date))
    print(f"\nWrote {len(written)}/{len(by_date)} candidate dates: {written}", flush=True)
