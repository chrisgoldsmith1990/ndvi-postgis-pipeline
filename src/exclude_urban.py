"""Exclude urban parcels (Bloomington/Normal) from the crop-type pipeline.

The >10-acre parcel filter (load_boundaries.py) and NDVI-based clustering
have no notion of land use or zoning -- a golf course, cemetery, or the
Rivian plant's flat pavement can pass every geometric/spectral test a real
field would, the same limitation the county-wide anomaly map's README
section already calls out. For the crop-type map specifically, this shows
up as parcels inside Bloomington/Normal city limits getting labeled
corn-like/soybean-like/non-row-crop despite obviously not being farmland
(confirmed directly: real parcels inside the twin cities showing up
colored on the interactive crop map).

Fetches Bloomington and Normal's real municipal boundary polygons from
OpenStreetMap via Nominatim's polygon_geojson search -- reassembling an
administrative-boundary relation from raw Overpass output into a single
clean polygon by hand is a known headache (multi-way outer/inner ring
stitching); Nominatim already does exactly that lookup and hands back a
ready GeoJSON polygon for a named place, so there's no reason to
reimplement it. Loads the two boundaries into a small PostGIS table and
excludes any parcel whose *centroid* falls within either one -- not
ST_Intersects, which would also exclude legitimate rural parcels that
merely share a boundary edge with the city limit -- from the crop-type
pipeline's input entirely, before clustering, not just hidden from the
map afterward. That matters: a handful of obviously non-agricultural
parcels sitting in the input could otherwise skew the corn/soybean split
itself, not just look wrong on the rendered map.
"""

import time

import requests
from shapely.geometry import shape
from sqlalchemy import text

from src.db import get_engine

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "ndvi-postgis-pipeline (portfolio project)"

# McLean County's two incorporated urban areas -- the only ones large
# enough to plausibly contain parcels the >10-acre filter still lets
# through (city parks, cemeteries, industrial/commercial tracts, the
# Rivian plant).
URBAN_PLACES = [
    "Bloomington, McLean County, Illinois, USA",
    "Normal, McLean County, Illinois, USA",
]


def fetch_urban_boundary(place, request_timeout=15):
    """Real municipal boundary polygon for a named place, via Nominatim's
    polygon_geojson search -- returns a shapely (Multi)Polygon in EPSG:4326."""
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": place, "format": "jsonv2", "polygon_geojson": 1, "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=request_timeout,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"Nominatim returned no results for {place!r}")
    return shape(results[0]["geojson"])


def load_urban_boundaries(geoms, table_name="urban_areas"):
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.execute(text(f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                geometry geometry(MultiPolygon, 4326)
            )
        """))
        for geom in geoms:
            conn.execute(
                text(f"INSERT INTO {table_name} (geometry) VALUES (ST_Multi(ST_GeomFromText(:wkt, 4326)))"),
                {"wkt": geom.wkt},
            )
        conn.execute(text(f"CREATE INDEX {table_name}_geom_gist ON {table_name} USING GIST (geometry)"))
    print(f"Loaded {len(geoms)} urban boundaries into '{table_name}'", flush=True)


def rural_pins(parcels_table="parcels_clipped_county", urban_table="urban_areas"):
    """Pins whose centroid does NOT fall inside any urban boundary."""
    engine = get_engine()
    with engine.begin() as conn:
        return [row[0] for row in conn.execute(text(f"""
            SELECT p.pin FROM {parcels_table} p
            WHERE p.geometry IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM {urban_table} u WHERE ST_Within(ST_Centroid(p.geometry), u.geometry)
            )
        """))]


def filter_rural(df, parcels_table="parcels_clipped_county", urban_table="urban_areas"):
    """Drop rows for parcels inside Bloomington/Normal from a pin-keyed
    DataFrame (e.g. crop_clusters.load_series()'s output) before fitting
    curves or clustering."""
    pins = set(rural_pins(parcels_table, urban_table))
    return df[df["pin"].isin(pins)]


if __name__ == "__main__":
    geoms = []
    for place in URBAN_PLACES:
        geoms.append(fetch_urban_boundary(place))
        time.sleep(1)  # Nominatim's public-instance usage policy: max 1 req/sec
    load_urban_boundaries(geoms)

    engine = get_engine()
    with engine.begin() as conn:
        total = conn.execute(text("SELECT count(*) FROM parcels_clipped_county")).scalar()
    pins = rural_pins()
    print(f"{len(pins)} of {total} county parcels are rural "
          f"(outside Bloomington/Normal city limits) -- {total - len(pins)} excluded", flush=True)
