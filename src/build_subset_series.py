"""Crop-type prep, continued: run compute_ndvi + zonal_stats for every date
fetch_timeseries.py wrote, into a table of its own.

Kept separate from ndvi_zonal_stats (table: ndvi_zonal_stats_subset) rather
than appended to it -- these are full acquisition-date labels like
"2026-09-03", not the "2026-08" monthly labels the county-wide map's
"latest date" logic expects, and they only cover the ~159-parcel subset.
Mixed into the same table, a bare `max(date)` would string-sort a subset
date above the real monthly ones and silently break the finished map's
default view. Two tables, two purposes.
"""

from pathlib import Path

from src.compute_ndvi import compute_ndvi
from src.zonal_stats import compute_zonal_stats, load_zonal_stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUBSET_TABLE = "ndvi_zonal_stats_subset"

if __name__ == "__main__":
    # data/raw/ also holds the old monthly composites ("2026-06", 7 chars)
    # from fetch_imagery.py -- only process fetch_timeseries.py's full-date
    # ("2026-04-09", 10 chars) directories here.
    raw_dirs = sorted(d for d in (DATA_DIR / "raw").iterdir() if len(d.name) == 10)
    for scene_dir in raw_dirs:
        date = scene_dir.name
        red, nir = scene_dir / "red.tif", scene_dir / "nir.tif"
        if not (red.exists() and nir.exists()):
            continue

        ndvi_path = DATA_DIR / "processed" / date / "ndvi.tif"
        compute_ndvi(red, nir, ndvi_path)

        result = compute_zonal_stats(ndvi_path)
        valid = result["ndvi_mean"].notna().sum()
        print(f"{date}: {valid} parcels with valid NDVI coverage", flush=True)
        load_zonal_stats(result, date, table_name=SUBSET_TABLE)
