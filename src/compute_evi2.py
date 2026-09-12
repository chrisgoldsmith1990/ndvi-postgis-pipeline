"""EVI2 (2-band Enhanced Vegetation Index) from the same red/NIR bands
NDVI already uses -- an experimental alternative, not (yet) wired into the
main pipeline.

EVI2 = 2.5 * (NIR - Red) / (NIR + 2.4*Red + 1), all in true reflectance
[0, 1]. NASA's own MODIS land-surface-phenology product (MCD12Q2) uses
EVI2 instead of NDVI specifically because it saturates less at high
canopy closure and is less sensitive to atmospheric/soil-background
noise -- both real NDVI weaknesses this project's yield_ranking.py already
calls out (corn and soybean's peak_ndvi values compress to nearly the same
range, ~0.88-0.93 for both clusters, right where NDVI saturates).

Unlike NDVI's formula, EVI2's "+1" term is NOT scale-invariant, so it
matters what units Red/NIR are actually stored in here. Checked directly:
this project's stored red.tif/nir.tif are raw uint16 digital numbers
(typical nonzero mean ~1300 red, ~2300 NIR), the standard ESA/USGS
convention of true reflectance x10000 for both Sentinel-2 L2A (Earth
Search) and HLS L30 (Planetary Computer) surface reflectance products --
confirmed, not assumed, since getting this wrong would silently produce a
meaningless index. Rather than converting to true [0,1] reflectance first,
the "+1" is scaled to "+10000" to match, which is algebraically identical
(dividing the whole expression through by 10000 recovers the textbook
formula exactly) and avoids an extra division pass over every pixel.
"""

from pathlib import Path

import numpy as np
import rasterio

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Raw-DN equivalent of EVI2's "+1" term, since Red/NIR here are stored as
# reflectance x10000 (see module docstring) rather than true [0, 1] values.
REFLECTANCE_SCALE = 10000


def compute_evi2(red_path, nir_path, out_path):
    with rasterio.open(red_path) as red_src, rasterio.open(nir_path) as nir_src:
        red = red_src.read(1).astype("float32")
        nir = nir_src.read(1).astype("float32")
        profile = red_src.profile.copy()

    denom = nir + 2.4 * red + REFLECTANCE_SCALE
    zero_pixel = (nir == 0) & (red == 0)  # this project's masked/nodata convention (fetch_timeseries.py etc.)
    evi2 = np.where(zero_pixel | (denom == 0), np.nan, 2.5 * (nir - red) / denom)
    # Same rationale as compute_ndvi.py's clip: bounded in practice, and an
    # artifact pixel (e.g. residual atmospheric-correction noise) shouldn't
    # be allowed to swing the zonal mean past what's physically plausible.
    evi2 = np.clip(evi2, -1.0, 1.0)

    profile.update(dtype="float32", count=1, nodata=np.nan)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(evi2.astype("float32"), 1)

    valid = evi2[~np.isnan(evi2)]
    print(f"Wrote {out_path} — EVI2 min/max/mean: "
          f"{valid.min():.3f}/{valid.max():.3f}/{valid.mean():.3f}", flush=True)
    return evi2


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        scene_dir = DATA_DIR / "raw" / sys.argv[1]
    else:
        scene_dir = sorted((DATA_DIR / "raw").iterdir())[-1]
    out_path = DATA_DIR / "processed_evi2" / scene_dir.name / "evi2.tif"
    compute_evi2(scene_dir / "red.tif", scene_dir / "nir.tif", out_path)
