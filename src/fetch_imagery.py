"""Step 1: pull Sentinel-2 imagery for the study area from AWS Open Data via the
Element84 Earth Search STAC API.

McLean County straddles four Sentinel-2 MGRS tiles (16TBK/BL/CK/CL), so a
single scene never covers the whole county — this mosaics same-date tiles
and clips to the county bbox in one step via rasterio.merge.
"""

import socket
from collections import defaultdict
from pathlib import Path

import rasterio
import rasterio.warp
from pystac_client import Client
from rasterio.merge import merge

# Fail fast instead of hanging indefinitely on a stalled connection.
socket.setdefaulttimeout(30)

GDAL_ENV_OPTS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_TIMEOUT": "30",
    "GDAL_HTTP_MAX_RETRY": "2",
}

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# McLean County, IL (Bloomington-Normal) — corn/soybean county. Source of the
# bbox: McLean County GIS Consortium (mcgis.org). WGS84 lon/lat.
STUDY_AREA_BBOX = (-89.2966, 40.2673, -88.43, 40.7711)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def find_best_date_items(bbox=STUDY_AREA_BBOX, datetime_range="2026-06-01/2026-08-31", max_cloud=20):
    """Find the acquisition date whose tiles give the fullest, cleanest coverage of the AOI."""
    print("Searching STAC catalog...", flush=True)
    catalog = Client.open(STAC_URL)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    print(f"Found {len(items)} candidate scenes", flush=True)

    by_date = defaultdict(list)
    for item in items:
        by_date[item.datetime.date()].append(item)

    def tile_id(item):
        return item.properties.get("s2:mgrs_tile") or item.properties.get("grid:code")

    max_tiles = max(len({tile_id(it) for it in its}) for its in by_date.values())

    def score(its):
        tile_count = len({tile_id(it) for it in its})
        mean_cloud = sum(it.properties["eo:cloud_cover"] for it in its) / len(its)
        return (tile_count < max_tiles, mean_cloud)  # full coverage first, then lowest cloud

    best_date = min(by_date, key=lambda d: score(by_date[d]))
    chosen = by_date[best_date]
    print(f"Selected {best_date}: {len(chosen)} tiles, "
          f"mean cloud {sum(it.properties['eo:cloud_cover'] for it in chosen) / len(chosen):.2f}%", flush=True)
    return chosen


def download_mosaic_clipped(items, band_key, out_path, bbox=STUDY_AREA_BBOX):
    """Mosaic same-date tiles for one band and clip to bbox in a single rasterio.merge call."""
    print(f"Mosaicking {band_key} from {len(items)} tiles...", flush=True)
    with rasterio.Env(**GDAL_ENV_OPTS):
        srcs = [rasterio.open(item.assets[band_key].href) for item in items]
        try:
            dst_bounds = rasterio.warp.transform_bounds("EPSG:4326", srcs[0].crs, *bbox)
            mosaic, transform = merge(srcs, bounds=dst_bounds)
            profile = srcs[0].profile.copy()
        finally:
            for src in srcs:
                src.close()

    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform, count=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)
    print(f"Wrote {out_path} ({mosaic.shape[2]}x{mosaic.shape[1]})", flush=True)


if __name__ == "__main__":
    items = find_best_date_items()
    date_str = items[0].datetime.date().isoformat()

    out_dir = RAW_DIR / date_str
    download_mosaic_clipped(items, "red", out_dir / "red.tif")
    download_mosaic_clipped(items, "nir", out_dir / "nir.tif")
    print(f"Done. red.tif and nir.tif written to {out_dir}", flush=True)
