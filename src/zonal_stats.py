"""Step 4: mean NDVI per polygon per date — the raster-vector intersection step,
and the point in the pipeline where the Python/PostGIS split actually gets
decided rather than just followed.

The raster (NDVI, a per-pixel array) stays in Python — that's the rule set
in load_boundaries.py. Computing "mean of the pixels under this polygon" is
therefore a Python-side operation too: it needs the array, not just the
geometry, and PostGIS's raster support exists but is the less-traveled path
compared to its vector tooling. rasterstats does this by rasterizing each
polygon against the NDVI grid and reducing the pixels underneath it — the
same operation zonal statistics always is, just named differently depending
on which library owns it.

What changes hands at the PostGIS boundary is the *result*, not the raster:
one mean-NDVI scalar per parcel per date is a tabular fact, and tabular
facts that need to be joined against other tabular facts (here: spatial
neighbors, in neighbor_comparison.py) belong in the database, not held in a
Python DataFrame for the rest of the pipeline's life.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import rasterstats
from sqlalchemy import text

from src.db import get_engine

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def compute_zonal_stats(ndvi_path, parcels_table="parcels"):
    engine = get_engine()
    parcels = gpd.read_postgis(f"SELECT pin, geometry FROM {parcels_table}", engine, geom_col="geometry")

    with rasterio.open(ndvi_path) as src:
        raster_crs = src.crs
    parcels = parcels.to_crs(raster_crs)

    # A parcel fully consumed by the roads/waterways clip (clip_parcels.py)
    # has a null or empty geometry -- no area, so no pixels to average, but
    # rasterstats' shape() crashes on None rather than returning NaN for it.
    # Skip those here and NaN-fill them back in, rather than let one non-crop
    # sliver from ST_Difference kill the whole date's zonal-stats run.
    has_geom = parcels.geometry.notna() & ~parcels.geometry.is_empty
    empty_pins = parcels.loc[~has_geom, "pin"]
    parcels = parcels.loc[has_geom]

    stats = rasterstats.zonal_stats(
        parcels.geometry,
        ndvi_path,
        stats=["mean", "min", "max", "std", "count"],
        nodata=float("nan"),
        geojson_out=False,
    )
    result = pd.DataFrame(stats).add_prefix("ndvi_")
    result["pin"] = parcels["pin"].values

    if len(empty_pins):
        empty_rows = pd.DataFrame({"pin": empty_pins.values})
        result = pd.concat([result, empty_rows], ignore_index=True)
    return result


def load_zonal_stats(result, date, table_name="ndvi_zonal_stats"):
    result = result.copy()
    result["date"] = date
    engine = get_engine()
    with engine.begin() as conn:
        exists = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = :t)"
        ), {"t": table_name}).scalar()
        if exists:
            conn.execute(text(f"DELETE FROM {table_name} WHERE date = :date"), {"date": date})
    result.to_sql(table_name, engine, if_exists="append", index=False)
    with engine.begin() as conn:
        conn.execute(text(f'CREATE INDEX IF NOT EXISTS {table_name}_pin_date '
                           f'ON {table_name} (pin, date)'))
    print(f"Loaded {len(result)} zonal stats rows for {date} into '{table_name}'", flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        ndvi_dir = DATA_DIR / "processed" / sys.argv[1]
    else:
        ndvi_dir = sorted((DATA_DIR / "processed").iterdir())[-1]
    date = ndvi_dir.name
    result = compute_zonal_stats(ndvi_dir / "ndvi.tif")
    valid = result["ndvi_mean"].notna().sum()
    print(f"Computed zonal stats for {len(result)} parcels ({valid} with valid NDVI coverage)", flush=True)
    load_zonal_stats(result, date)
