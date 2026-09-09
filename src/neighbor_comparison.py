"""Step 5: compare each field against spatially adjacent fields.

This is the step the PostGIS argument is built for. Two options for
"neighbor": strict topological adjacency (ST_Touches/ST_Intersects) fails
here because parcel polygons stop at the property line — most fields have a
road or ditch between them, so true field neighbors usually don't share a
boundary at all. K-nearest-neighbor by centroid distance is the right model
of "adjacent field" for this data, and it's also the operation PostGIS is
built to accelerate: the `<->` distance operator, ordered and LIMIT-ed
inside a LATERAL join, is index-accelerated by the GIST index created in
load_boundaries.py — Postgres walks the index for the K closest geometries
instead of computing every pairwise distance and sorting. Doing this in
geopandas means the full O(n^2) distance matrix, or hand-rolling a KD-tree;
in PostGIS it's the query planner's job and a five-line query.

Neighbor comparison — rather than an absolute NDVI threshold — is what
controls for regional weather: a drought year suppresses NDVI county-wide,
so "this field is 20% below its healthy peak" is a weak signal on its own,
but "this field is 20% below its immediate neighbors, planted in the same
week under the same rain" is a much stronger one.
"""

from sqlalchemy import text

from src.db import get_engine

K_NEAREST = 8
MAX_NEIGHBOR_DISTANCE_M = 3000  # cap so an isolated parcel doesn't get "neighbors" a mile away


def build_neighbor_table(k=K_NEAREST, max_distance_m=MAX_NEIGHBOR_DISTANCE_M):
    """Materialize each parcel's K nearest other parcels by centroid distance."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS parcel_neighbors"))
        conn.execute(text(f"""
            CREATE TABLE parcel_neighbors AS
            SELECT p.pin,
                   n.pin AS neighbor_pin,
                   ST_Distance(p.geometry::geography, n.geometry::geography) AS distance_m
            FROM parcels p
            CROSS JOIN LATERAL (
                SELECT n.pin, n.geometry
                FROM parcels n
                WHERE n.pin != p.pin
                ORDER BY n.geometry <-> p.geometry
                LIMIT {k}
            ) n
            WHERE ST_DWithin(p.geometry::geography, n.geometry::geography, {max_distance_m})
        """))
        conn.execute(text("CREATE INDEX parcel_neighbors_pin ON parcel_neighbors (pin)"))
    print("Built parcel_neighbors (KNN via GIST-accelerated <-> operator)", flush=True)


def compare_to_neighbors(date, table_name="neighbor_comparison"):
    """For a given date, compute each field's NDVI deviation from its neighbor mean."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                pin text, date text, field_ndvi double precision,
                neighbor_mean_ndvi double precision, ndvi_diff double precision, neighbor_count int
            )
        """))
        if _table_exists(conn, table_name):
            conn.execute(text(f"DELETE FROM {table_name} WHERE date = :date"), {"date": date})
        conn.execute(text(f"""
            INSERT INTO {table_name} (pin, date, field_ndvi, neighbor_mean_ndvi, ndvi_diff, neighbor_count)
            SELECT z.pin,
                   z.date,
                   z.ndvi_mean AS field_ndvi,
                   avg(zn.ndvi_mean) AS neighbor_mean_ndvi,
                   z.ndvi_mean - avg(zn.ndvi_mean) AS ndvi_diff,
                   count(zn.ndvi_mean) AS neighbor_count
            FROM ndvi_zonal_stats z
            JOIN parcel_neighbors pn ON pn.pin = z.pin
            JOIN ndvi_zonal_stats zn ON zn.pin = pn.neighbor_pin AND zn.date = z.date
            WHERE z.date = :date AND z.ndvi_mean IS NOT NULL AND zn.ndvi_mean IS NOT NULL
            GROUP BY z.pin, z.date, z.ndvi_mean
        """), {"date": date})
    print(f"Computed neighbor comparison for {date}", flush=True)


def _table_exists(conn, table_name):
    return conn.execute(text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = :t)"
    ), {"t": table_name}).scalar()


if __name__ == "__main__":
    import sys

    date = sys.argv[1] if len(sys.argv) > 1 else None
    if date is None:
        engine = get_engine()
        with engine.begin() as conn:
            date = conn.execute(text("SELECT max(date) FROM ndvi_zonal_stats")).scalar()

    build_neighbor_table()
    compare_to_neighbors(date)
