"""County-wide map output: the visual the rest of the pipeline builds toward.

Not one of the original 7 pipeline steps, but the natural payoff of them —
zonal_stats.py and anomaly.py produce per-parcel numbers; this is where
those numbers become a map of the county, which is what actually makes the
pipeline's output legible at a glance.

Produces two things from the same query:
- reports/county_ndvi_map.png — static choropleth for the README/GitHub.
- reports/county_ndvi_map.html — interactive (pan/zoom/click) version, built
  on folium via geopandas.explore(), for actually looking around the county.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from sqlalchemy import text

from src.db import get_engine

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def load_map_data(date):
    engine = get_engine()
    gdf = gpd.read_postgis(
        text("""
            SELECT p.pin, p.geometry, z.ndvi_mean,
                   COALESCE(a.is_anomaly, false) AS is_anomaly
            FROM parcels p
            JOIN ndvi_zonal_stats z ON z.pin = p.pin AND z.date = :date
            LEFT JOIN field_anomalies a ON a.pin = p.pin AND a.date = :date
            WHERE z.ndvi_mean IS NOT NULL
        """),
        engine,
        params={"date": date},
        geom_col="geometry",
    )
    return gdf


def plot_static_map(gdf, date, out_path):
    fig, ax = plt.subplots(figsize=(10, 9))
    gdf.plot(column="ndvi_mean", cmap="RdYlGn", vmin=0, vmax=1, legend=True,
              legend_kwds={"label": "Mean NDVI", "shrink": 0.7}, ax=ax)
    anomalies = gdf[gdf["is_anomaly"]]
    if len(anomalies):
        anomalies.boundary.plot(ax=ax, color="black", linewidth=1.2)
        anomalies.representative_point().plot(ax=ax, color="black", markersize=8, marker="x")
    ax.set_title(f"McLean County field NDVI — {date}\n"
                 f"({len(anomalies)} fields flagged as anomalous, outlined in black)")
    ax.set_axis_off()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}", flush=True)


def plot_interactive_map(gdf, out_path):
    # Simplify boundaries before embedding in HTML/JS — full parcel-survey
    # precision isn't visible at county zoom and it's most of the file size.
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].simplify(0.00003)
    m = gdf.explore(
        column="ndvi_mean",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        tooltip=["pin", "ndvi_mean", "is_anomaly"],
        style_kwds={"weight": 0.5},
        name="Parcels (NDVI)",
    )

    # Satellite layer for ground-truthing flagged parcels against what's
    # actually there (e.g. a plant or quarry, not a crop under stress).
    import folium

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)
    # bottomright, not topright: the branca colormap legend from .explore()
    # also docks topright and was covering the layer control there.
    folium.LayerControl(collapsed=False, position="bottomright").add_to(m)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    import sys

    date = sys.argv[1] if len(sys.argv) > 1 else "2026-08"
    gdf = load_map_data(date)
    print(f"Loaded {len(gdf)} parcels for {date} ({gdf['is_anomaly'].sum()} anomalous)", flush=True)
    plot_static_map(gdf, date, REPORTS_DIR / "county_ndvi_map.png")
    plot_interactive_map(gdf, REPORTS_DIR / "county_ndvi_map.html")
