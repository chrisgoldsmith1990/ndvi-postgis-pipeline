"""Per-parcel ground-truth crop label from a USDA CDL raster (fetch_cdl.py)
-- majority CDL class within each parcel's clipped geometry, the
categorical equivalent of zonal_stats.py's mean-based reduction for
continuous NDVI/EVI2 rasters.
"""

import geopandas as gpd
import pandas as pd
import rasterio
import rasterstats
from sqlalchemy import text

from src.db import get_engine

# Standard USDA NASS CDL class codes.
CDL_LABELS = {1: "Corn", 5: "Soybean"}


def load_cdl_ground_truth(cdl_path, parcels_table="parcels_clipped", pins=None):
    engine = get_engine()
    query = f"SELECT pin, geometry FROM {parcels_table}"
    params = {}
    if pins is not None:
        query += " WHERE pin = ANY(:pins)"
        params["pins"] = list(pins)
    parcels = gpd.read_postgis(text(query), engine, params=params, geom_col="geometry")

    with rasterio.open(cdl_path) as src:
        raster_crs = src.crs
    parcels = parcels.to_crs(raster_crs)

    # Same empty-geometry issue as zonal_stats.py: a parcel fully consumed
    # by the roads/waterways clip has a null/empty geometry, which crashes
    # rasterstats' shape() rather than returning NaN for it.
    has_geom = parcels.geometry.notna() & ~parcels.geometry.is_empty
    empty_pins = parcels.loc[~has_geom, "pin"]
    parcels = parcels.loc[has_geom]

    stats = rasterstats.zonal_stats(parcels.geometry, cdl_path, categorical=True, nodata=0)
    rows = [{"pin": pin, "cdl_code": None, "cdl_label": None, "cdl_purity": None} for pin in empty_pins]
    for pin, s in zip(parcels["pin"], stats):
        if not s:
            rows.append({"pin": pin, "cdl_code": None, "cdl_label": None, "cdl_purity": None})
            continue
        total = sum(s.values())
        majority_code = max(s, key=s.get)
        rows.append({
            "pin": pin,
            "cdl_code": majority_code,
            "cdl_label": CDL_LABELS.get(majority_code, "Other"),
            # fraction of the parcel's pixels agreeing with the majority class --
            # a parcel straddling a class boundary (e.g. clipped-out edge
            # pixels, or genuinely mixed land use) gets a low purity, useful
            # for optionally excluding ambiguous ground-truth parcels rather
            # than trusting a bare-majority call on a near-even split.
            "cdl_purity": s[majority_code] / total,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    from src.fetch_cdl import fetch_cdl_raster

    year = sys.argv[1] if len(sys.argv) > 1 else "2021"
    bbox = (-88.65, 40.65, -88.55, 40.72)
    cdl_path = fetch_cdl_raster(bbox, year)
    gt = load_cdl_ground_truth(cdl_path)
    print(gt["cdl_label"].value_counts(dropna=False))
    print("\nPurity distribution for Corn/Soybean parcels:")
    print(gt[gt["cdl_label"].isin(["Corn", "Soybean"])]["cdl_purity"].describe())
