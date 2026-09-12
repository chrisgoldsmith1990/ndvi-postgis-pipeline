"""Crop-type prep, continued: run compute_ndvi + zonal_stats for every date
fetch_timeseries.py/fetch_hls.py wrote, into a table of its own.

Kept separate from ndvi_zonal_stats (table: ndvi_zonal_stats_subset*) rather
than appended to it -- these are full acquisition-date labels like
"2026-09-03", not the "2026-08" monthly labels the county-wide map's
"latest date" logic expects, and they only cover the ~159-parcel subset.
Mixed into the same table, a bare `max(date)` would string-sort a subset
date above the real monthly ones and silently break the finished map's
default view. Two tables, two purposes.

Defaults to parcels_clipped / ndvi_zonal_stats_subset_clipped, matching
crop_clusters.py's default (see clip_parcels.py) -- clipped is the
standard everywhere else in this pipeline now, not just an alternative.
"""

from pathlib import Path

from src.compute_ndvi import compute_ndvi
from src.zonal_stats import compute_zonal_stats, load_zonal_stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUBSET_TABLE = "ndvi_zonal_stats_subset_clipped"


def run(parcels_table="parcels_clipped", zonal_table=SUBSET_TABLE, recompute_ndvi=True,
        raw_dir=None, processed_dir=None):
    """NDVI + zonal stats for every fetch_timeseries.py/fetch_hls.py date,
    against whichever parcels table is given. raw_dir/processed_dir default
    to the subset paths (data/raw, data/processed); pass data/raw/county
    and data/processed/county for the county-wide run, so its NDVI outputs
    never collide with a subset date that happens to share a calendar date."""
    raw_dir = raw_dir or (DATA_DIR / "raw")
    processed_dir = processed_dir or (DATA_DIR / "processed")

    # raw_dir also holds the old monthly composites ("2026-06", 7 chars)
    # from fetch_imagery.py, and (subset raw_dir only) the 2021
    # CDL-validation season's dates (validate_against_cdl.py) -- also
    # 10-char date strings, so length alone isn't enough anymore. Only
    # process this (2026) season's full-date directories here.
    raw_dirs = sorted(d for d in raw_dir.iterdir() if len(d.name) == 10 and d.name.startswith("2026-"))
    for scene_dir in raw_dirs:
        date = scene_dir.name
        red, nir = scene_dir / "red.tif", scene_dir / "nir.tif"
        if not (red.exists() and nir.exists()):
            continue

        ndvi_path = processed_dir / date / "ndvi.tif"
        if recompute_ndvi or not ndvi_path.exists():
            compute_ndvi(red, nir, ndvi_path)

        result = compute_zonal_stats(ndvi_path, parcels_table=parcels_table)
        valid = result["ndvi_mean"].notna().sum()
        print(f"{date}: {valid} parcels with valid NDVI coverage ({parcels_table})", flush=True)
        load_zonal_stats(result, date, table_name=zonal_table)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "county":
        run(parcels_table="parcels_clipped_county", zonal_table="ndvi_zonal_stats_county_clipped",
            raw_dir=DATA_DIR / "raw" / "county", processed_dir=DATA_DIR / "processed" / "county")
    else:
        run()
