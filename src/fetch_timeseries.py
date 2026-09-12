"""Crop-type prep: pull EVERY usable Sentinel-2 date across the growing
season for a small subset area, for building real per-parcel NDVI curves
(phenology) rather than the monthly best-pixel composites fetch_imagery.py
builds for the county-wide anomaly map.

Monthly composites are the wrong input here on purpose: compositing across
weeks smooths away the week-to-week shape, which is the entire signal that
distinguishes corn from soybean (peak timing/height, green-up rate,
senescence rate). This instead pulls each individual acquisition date,
masks per-pixel cloud/shadow with the L2A Scene Classification Layer, and
keeps a date only if the AOI itself came back mostly clear -- decided from
what's actually in our small clipped window, not the whole tile's
cloud/nodata percentage (a poor proxy once the AOI is this much smaller
than a tile).

Scoped to a small rural subset (~160 parcels, ~12,600 acres, no known
non-cropland anomalies) rather than the full county: pulling a full
season at daily-ish cadence for all 9,578 parcels would multiply
fetch_imagery.py's already-multi-minute per-month runtime by ~15-20x.
"""

import socket
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
from pystac_client import Client
from rasterio.merge import merge

socket.setdefaulttimeout(30)

GDAL_ENV_OPTS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_TIMEOUT": "30",
    "GDAL_HTTP_MAX_RETRY": "2",
}

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# Rural cluster in northeast McLean County: 159 parcels >10 acres, ~12,600
# acres, no fields already known (from the county-wide anomaly run) to be
# non-cropland. Picked to keep this run fast and the resulting curves clean.
SUBSET_BBOX = (-88.65, 40.65, -88.55, 40.72)

# L2A Scene Classification Layer codes that are NOT usable crop signal:
# 0 no-data, 1 saturated/defective, 3 cloud shadow, 8/9 cloud (med/high
# probability), 10 thin cirrus. Keeping the rest (vegetation, bare soil,
# water, unclassified, snow) is intentionally permissive -- the point is
# excluding cloud/shadow contamination, not pre-judging land cover.
SCL_BAD_VALUES = {0, 1, 3, 8, 9, 10}
# Loosened from 0.2 to 0.35: the entire green-up transition for most parcels
# falls in the 45-day gap between the two nearest clean dates either side of
# it (May 9 -> June 23), and the two next-best candidates inside that gap
# (May 14 at 34% bad, June 15 at 28% bad) sit just above the stricter cutoff
# -- worth the extra pixel noise on those two dates to get any signal at all
# inside the window that actually determines ramp shape.
MAX_BAD_FRACTION = 0.35

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def find_season_items(bbox=SUBSET_BBOX, datetime_range="2026-04-01/2026-09-11"):
    """All candidate scenes for the season -- no whole-tile cloud filter here.
    A tile's overall eo:cloud_cover is a poor proxy once the AOI is this much
    smaller than the tile: a scene can be heavily clouded on one side of a
    110km tile while our 8km subset sits clear underneath it. (Filtering at
    <70% whole-tile cloud here cut 41 real candidate dates down to 26 before
    this was caught -- the per-AOI SCL check below is the only filter that
    should apply at this AOI size.)"""
    print("Searching STAC catalog for the season...", flush=True)
    catalog = Client.open(STAC_URL)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=datetime_range,
    )
    items = list(search.items())
    print(f"Found {len(items)} candidate scenes", flush=True)

    by_date = defaultdict(list)
    for item in items:
        by_date[item.datetime.date()].append(item)
    return dict(sorted(by_date.items()))


def _read_clipped(items, band_key, bbox):
    with rasterio.Env(**GDAL_ENV_OPTS):
        srcs = [rasterio.open(item.assets[band_key].href) for item in items]
        try:
            dst_bounds = rasterio.warp.transform_bounds("EPSG:4326", srcs[0].crs, *bbox)
            mosaic, transform = merge(srcs, bounds=dst_bounds)
            crs = srcs[0].crs
        finally:
            for src in srcs:
                src.close()
    return mosaic[0], transform, crs


def build_date(items, date, bbox=SUBSET_BBOX):
    """Clip+mosaic red/NIR/SCL for one acquisition date, apply the cloud
    mask, and write red.tif/nir.tif with masked pixels zeroed out (the same
    nodata convention raw Sentinel-2 already uses, so compute_ndvi.py's
    existing (nir+red)==0 -> NaN logic picks it up with no changes needed).
    Returns None (writes nothing) if too much of the AOI is masked out.
    """
    red, transform, crs = _read_clipped(items, "red", bbox)
    nir, _, _ = _read_clipped(items, "nir", bbox)
    scl, scl_transform, _ = _read_clipped(items, "scl", bbox)

    # SCL is 20m native resolution vs. 10m for red/NIR -- resample to match.
    if scl.shape != red.shape:
        from rasterio.enums import Resampling
        from rasterio.warp import reproject

        scl_resampled = np.empty(red.shape, dtype=scl.dtype)
        reproject(
            source=scl, destination=scl_resampled,
            src_transform=scl_transform, src_crs=crs,
            dst_transform=transform, dst_crs=crs,
            resampling=Resampling.nearest,
        )
        scl = scl_resampled

    bad_mask = np.isin(scl, list(SCL_BAD_VALUES))
    bad_fraction = bad_mask.mean()
    if bad_fraction > MAX_BAD_FRACTION:
        print(f"  {date}: {bad_fraction:.0%} cloud/shadow/nodata -- skipped", flush=True)
        return None

    red = red.copy()
    nir = nir.copy()
    red[bad_mask] = 0
    nir[bad_mask] = 0

    profile = {
        "driver": "GTiff", "dtype": red.dtype, "count": 1,
        "height": red.shape[0], "width": red.shape[1],
        "crs": crs, "transform": transform,
    }
    out_dir = RAW_DIR / str(date)
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_dir / "red.tif", "w", **profile) as dst:
        dst.write(red, 1)
    with rasterio.open(out_dir / "nir.tif", "w", **profile) as dst:
        dst.write(nir, 1)
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
