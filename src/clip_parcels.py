"""Clip roads and waterways out of parcel polygons before zonal stats.

A parcel's tax-boundary polygon can include or abut a road right-of-way or
a stream/ditch channel that cuts through the field itself -- those pixels
read as pavement or water, not crop, and get averaged into the parcel's
NDVI mean along with everything else. This isn't a hypothetical: the
subset area has 44 real OSM road/waterway segments running through or
along its ~163 parcels (mostly rural county roads and farm streams, per
an Overpass query -- no major highways).

Fetches roads/waterways from OpenStreetMap's public Overpass API, buffers
each by an approximate right-of-way/channel width (real widths aren't in
OSM for rural roads, so these are reasonable class-based assumptions, not
measurements), unions them, and subtracts the result from each parcel via
PostGIS ST_Difference -- the same "PostGIS does the vector/spatial-set
work" split used everywhere else in this project. Output goes to a new
parcels_clipped table, not a modification of parcels, so the unclipped
zonal stats already computed stay comparable against this.
"""

import json
from pathlib import Path

import requests
from shapely.geometry import LineString
from sqlalchemy import text

from src.db import get_engine
from src.fetch_timeseries import COUNTY_BBOX, SUBSET_BBOX

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Half-width buffer in meters, by OSM tag class -- approximate right-of-way/
# channel width, not a measurement. Rural roads/farm streams in this subset
# don't carry width tags in OSM, so these are reasonable assumptions.
HIGHWAY_BUFFER_M = {
    "motorway": 15, "trunk": 15, "primary": 10, "secondary": 8,
    "tertiary": 7, "residential": 6, "unclassified": 6,
    "service": 4, "track": 3, "path": 2,
}
WATERWAY_BUFFER_M = {"river": 8, "stream": 4, "ditch": 2, "drain": 2}
DEFAULT_BUFFER_M = 5


def fetch_osm_obstructions(bbox=SUBSET_BBOX, overpass_timeout=25, request_timeout=30):
    """Roads + waterways in the AOI from OpenStreetMap, each tagged with an
    approximate buffer half-width by class."""
    west, south, east, north = bbox
    query = (
        f'[out:json][timeout:{overpass_timeout}];'
        f'(way["highway"]({south},{west},{north},{east});'
        f'way["waterway"]({south},{west},{north},{east}););'
        f'out body;>;out skel qt;'
    )
    resp = requests.post(
        OVERPASS_URL, data={"data": query}, timeout=request_timeout,
        headers={"User-Agent": "ndvi-postgis-pipeline (portfolio project)"},
    )
    resp.raise_for_status()
    data = resp.json()

    nodes = {el["id"]: (el["lon"], el["lat"]) for el in data["elements"] if el["type"] == "node"}
    features = []
    for el in data["elements"]:
        if el["type"] != "way" or len(el.get("nodes", [])) < 2:
            continue
        tags = el.get("tags", {})
        coords = [nodes[n] for n in el["nodes"] if n in nodes]
        if len(coords) < 2:
            continue
        if "highway" in tags:
            buffer_m = HIGHWAY_BUFFER_M.get(tags["highway"], DEFAULT_BUFFER_M)
        elif "waterway" in tags:
            buffer_m = WATERWAY_BUFFER_M.get(tags["waterway"], DEFAULT_BUFFER_M)
        else:
            continue
        features.append((LineString(coords), buffer_m))
    print(f"Fetched {len(features)} road/waterway segments from OSM", flush=True)
    return features


def load_obstructions(features, table_name="osm_obstructions"):
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.execute(text(f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                buffer_m double precision,
                geometry geometry(LineString, 4326)
            )
        """))
        for geom, buffer_m in features:
            conn.execute(
                text(f"INSERT INTO {table_name} (buffer_m, geometry) "
                     f"VALUES (:buffer_m, ST_GeomFromText(:wkt, 4326))"),
                {"buffer_m": buffer_m, "wkt": geom.wkt},
            )
    print(f"Loaded {len(features)} obstructions into '{table_name}'", flush=True)


def clip_parcels(pins, obstructions_table="osm_obstructions",
                  parcels_table="parcels", out_table="parcels_clipped"):
    """ST_Difference each parcel against the unioned, buffered obstructions.
    Buffering happens in geography (meters) then casts back to geometry for
    the difference, since the source data is WGS84 lon/lat.

    The county's own parcel data has a handful (4 of 9,578) of duplicate
    PINs -- degenerate near-zero-area sliver records, and in one case what
    looks like a real parcel split that kept the same PIN on both pieces.
    Never surfaced before because no earlier step enforced one-row-per-pin.
    Deduped here by keeping the largest-area record per PIN (DISTINCT ON),
    which naturally drops the zero-area duplicates and, for the genuine
    split, keeps the larger piece -- an acceptable simplification at this
    scale (4 of 9,578 parcels)."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {out_table}"))
        conn.execute(text(f"""
            CREATE TABLE {out_table} AS
            WITH obstruction_union AS (
                SELECT ST_Union(ST_Buffer(geometry::geography, buffer_m)::geometry) AS geom
                FROM {obstructions_table}
            ),
            deduped_parcels AS (
                SELECT DISTINCT ON (pin) pin, geometry, computed_ac
                FROM {parcels_table}
                WHERE pin = ANY(:pins)
                ORDER BY pin, ST_Area(geometry) DESC
            )
            SELECT p.pin,
                   ST_Difference(p.geometry, obstruction_union.geom) AS geometry,
                   ST_Area(ST_Difference(p.geometry, obstruction_union.geom)::geography) / 4046.8564224 AS clipped_acres,
                   p.computed_ac AS original_acres
            FROM deduped_parcels p
            CROSS JOIN obstruction_union
        """), {"pins": list(pins)})
        conn.execute(text(f"CREATE INDEX {out_table}_geom_gist ON {out_table} USING GIST (geometry)"))
        conn.execute(text(f"ALTER TABLE {out_table} ADD PRIMARY KEY (pin)"))

    with engine.begin() as conn:
        result = conn.execute(text(
            f"SELECT round(avg(original_acres - clipped_acres)::numeric, 3) AS avg_removed, "
            f"round(sum(original_acres - clipped_acres)::numeric, 1) AS total_removed FROM {out_table}"
        )).one()
    print(f"Clipped {len(pins)} parcels into '{out_table}': "
          f"avg {result.avg_removed} acres removed/parcel, {result.total_removed} total", flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "county":
        engine = get_engine()
        with engine.begin() as conn:
            pins = [row[0] for row in conn.execute(text("SELECT pin FROM parcels"))]
        print(f"County mode: {len(pins)} parcels", flush=True)
        features = fetch_osm_obstructions(bbox=COUNTY_BBOX, overpass_timeout=90, request_timeout=120)
        load_obstructions(features, table_name="osm_obstructions_county")
        clip_parcels(pins, obstructions_table="osm_obstructions_county",
                      parcels_table="parcels", out_table="parcels_clipped_county")
    else:
        from src.crop_clusters import load_series

        df = load_series()
        pins = df["pin"].unique().tolist()
        features = fetch_osm_obstructions()
        load_obstructions(features)
        clip_parcels(pins)
