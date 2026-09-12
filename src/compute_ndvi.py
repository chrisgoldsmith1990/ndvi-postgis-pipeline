"""Step 2: compute NDVI from Sentinel-2 red/NIR bands with rasterio.

NDVI = (NIR - Red) / (NIR + Red), the standard measure of vegetation
"greenness": healthy chlorophyll-rich canopy absorbs red light for
photosynthesis and strongly reflects NIR, so dense healthy vegetation reads
close to +1, bare soil/water/stressed crop reads near 0 or negative.
"""

from pathlib import Path

import numpy as np
import rasterio

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def compute_ndvi(red_path, nir_path, out_path):
    with rasterio.open(red_path) as red_src, rasterio.open(nir_path) as nir_src:
        red = red_src.read(1).astype("float32")
        nir = nir_src.read(1).astype("float32")
        profile = red_src.profile.copy()

    denom = nir + red
    ndvi = np.where(denom == 0, np.nan, (nir - red) / denom)
    # NDVI is bounded [-1, 1] by definition; a value outside that range is
    # necessarily an artifact (e.g. atmospheric-correction algorithms can
    # produce small negative reflectance at individual dark/shadow-edge
    # pixels, which the ratio can amplify past the theoretical bound), not
    # a real vegetation signal. Seen in practice on HLS Landsat scenes
    # (max NDVI up to 1.18 pre-clip) -- doesn't change any parcel's zonal
    # mean for this dataset (the offending pixels are too few and get
    # averaged out), but clipping is the principled fix regardless.
    ndvi = np.clip(ndvi, -1.0, 1.0)

    profile.update(dtype="float32", count=1, nodata=np.nan)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(ndvi.astype("float32"), 1)

    valid = ndvi[~np.isnan(ndvi)]
    print(f"Wrote {out_path} — NDVI min/max/mean: "
          f"{valid.min():.3f}/{valid.max():.3f}/{valid.mean():.3f}", flush=True)
    return ndvi


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        scene_dir = DATA_DIR / "raw" / sys.argv[1]
    else:
        scene_dir = sorted((DATA_DIR / "raw").iterdir())[-1]
    out_path = DATA_DIR / "processed" / scene_dir.name / "ndvi.tif"
    compute_ndvi(scene_dir / "red.tif", scene_dir / "nir.tif", out_path)
