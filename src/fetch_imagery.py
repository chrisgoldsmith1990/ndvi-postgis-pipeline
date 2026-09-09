"""Step 1: pull Sentinel-2 imagery for the study area from AWS Open Data via the
Element84 Earth Search STAC API.

Two real-world gotchas showed up building this against actual imagery
rather than a single hand-picked scene, and both are worth keeping visible
rather than papering over:

1. McLean County straddles four Sentinel-2 MGRS tiles (16TBK/BL/CK/CL), so
   a single scene never covers the whole county.
2. A tile "existing" for a given date doesn't mean it has full data over
   the AOI — individual passes can have large nodata gaps from swath edges
   (STAC exposes this as `s2:nodata_pixel_percentage`, and it's a tile-wide
   number that can be 0% for one date and 78% for the same tile a few weeks
   later). Requiring one single date with all four tiles clean turned out
   to not exist for most of the growing season.

So this builds a best-pixel composite instead of picking one scene: for
each of the four tile positions independently, rank candidate acquisitions
by nodata% then cloud%, and hand rasterio.merge the ranked list so it fills
each tile's position from the best available date, and falls back to the
next-best acquisition of that same tile only where the first still has
gaps. This is the standard cloud-free-composite approach, not a workaround
specific to this AOI.
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


def _tile_id(item):
    return item.properties.get("s2:mgrs_tile") or item.properties.get("grid:code")


def select_best_tiles(bbox=STUDY_AREA_BBOX, datetime_range="2026-06-01/2026-08-31", max_cloud=30):
    """For each MGRS tile covering the AOI, rank candidate acquisitions by nodata% then cloud%."""
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

    by_tile = defaultdict(list)
    for item in items:
        by_tile[_tile_id(item)].append(item)

    for tile, candidates in by_tile.items():
        candidates.sort(key=lambda it: (
            it.properties.get("s2:nodata_pixel_percentage", 0.0),
            it.properties["eo:cloud_cover"],
        ))

    ordered = [it for candidates in by_tile.values() for it in candidates]
    primary = [candidates[0] for candidates in by_tile.values()]
    for it in primary:
        print(f"  primary {_tile_id(it)}: {it.id} "
              f"(nodata={it.properties.get('s2:nodata_pixel_percentage', 0):.1f}%, "
              f"cloud={it.properties['eo:cloud_cover']:.1f}%)", flush=True)
    return ordered


def download_mosaic_clipped(items, band_key, out_path, bbox=STUDY_AREA_BBOX):
    """Mosaic ranked per-tile candidates for one band, clipped to bbox, best-pixel-first."""
    print(f"Mosaicking {band_key} from {len(items)} candidate rasters...", flush=True)
    with rasterio.Env(**GDAL_ENV_OPTS):
        srcs = [rasterio.open(item.assets[band_key].href) for item in items]
        try:
            dst_bounds = rasterio.warp.transform_bounds("EPSG:4326", srcs[0].crs, *bbox)
            mosaic, transform = merge(srcs, bounds=dst_bounds, nodata=0)
            profile = srcs[0].profile.copy()
        finally:
            for src in srcs:
                src.close()

    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform, count=1, nodata=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)

    remaining_gap = (mosaic[0] == 0).mean()
    print(f"Wrote {out_path} ({mosaic.shape[2]}x{mosaic.shape[1]}), "
          f"{remaining_gap:.2%} nodata remaining after compositing", flush=True)


if __name__ == "__main__":
    import sys

    datetime_range = sys.argv[1] if len(sys.argv) > 1 else "2026-06-01/2026-08-31"
    label = sys.argv[2] if len(sys.argv) > 2 else datetime_range.split("/")[0][:7]  # YYYY-MM

    items = select_best_tiles(datetime_range=datetime_range)
    out_dir = RAW_DIR / label
    download_mosaic_clipped(items, "red", out_dir / "red.tif")
    download_mosaic_clipped(items, "nir", out_dir / "nir.tif")
    print(f"Done. red.tif and nir.tif written to {out_dir}", flush=True)
