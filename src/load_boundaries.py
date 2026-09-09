"""Step 3: load field/parcel boundaries into PostGIS with geopandas.

True USDA CLU (Common Land Unit) boundaries aren't publicly redistributable —
they require an FSA data request tied to a specific use case. County parcel
data is the public substitute: same "polygon per piece of land" shape, and
McLean County's parcel layer only carries a parcel ID + acreage (no owner
name/address), which keeps this comfortably in public-open-data territory.

Parcels are filtered to >10 acres to approximate field-scale agricultural
land rather than every residential lot in Bloomington-Normal — ~9.6k of the
county's ~71k parcels clear that bar, which is the actual point of pushing
this into PostGIS: geopandas-in-memory is fine at that scale, but it's also
exactly the scale where an un-indexed spatial join stops being fine.
"""

from pathlib import Path

import geopandas as gpd
import requests
from sqlalchemy import text

from src.db import get_engine

PARCELS_QUERY_URL = "https://www.mcgisweb.org/mcgc/rest/services/OpenData/OpenData/MapServer/2/query"
MIN_ACRES = 10
PAGE_SIZE = 2000


def fetch_parcels(min_acres=MIN_ACRES):
    """Page through the ArcGIS FeatureServer query endpoint and return one GeoDataFrame."""
    where = f"COMPUTED_AC > {min_acres}"
    features = []
    offset = 0
    while True:
        resp = requests.get(
            PARCELS_QUERY_URL,
            params={
                "where": where,
                "outFields": "OBJECTID,PIN,DEED_AC,COMPUTED_AC",
                "outSR": 4326,
                "f": "geojson",
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
            },
            timeout=30,
        )
        resp.raise_for_status()
        page = resp.json()
        page_features = page.get("features", [])
        features.extend(page_features)
        print(f"Fetched {len(features)} parcels...", flush=True)
        if len(page_features) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf.columns = [c.lower() for c in gdf.columns]
    return gdf


def load_to_postgis(gdf, table_name="parcels"):
    engine = get_engine()
    gdf.to_postgis(table_name, engine, if_exists="replace", index=False)
    with engine.begin() as conn:
        # Real county parcel data reliably includes a handful of self-intersecting
        # rings; ST_MakeValid before anything downstream tries ST_Intersects on them.
        conn.execute(text(f'UPDATE {table_name} SET geometry = ST_MakeValid(geometry) '
                           f'WHERE NOT ST_IsValid(geometry)'))
        conn.execute(text(f'CREATE INDEX IF NOT EXISTS {table_name}_geom_gist '
                           f'ON {table_name} USING GIST (geometry)'))
    print(f"Loaded {len(gdf)} parcels into '{table_name}' with a GIST index", flush=True)


if __name__ == "__main__":
    gdf = fetch_parcels()
    load_to_postgis(gdf)
