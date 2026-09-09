"""County-wide map output: the visual the rest of the pipeline builds toward.

Not one of the original 7 pipeline steps, but the natural payoff of them —
zonal_stats.py and anomaly.py produce per-parcel numbers; this is where
those numbers become a map of the county, which is what actually makes the
pipeline's output legible at a glance.

Produces two things from the same query:
- reports/county_ndvi_map.png — static choropleth for the README/GitHub.
- reports/county_ndvi_map.html — interactive (pan/zoom/click) version, built
  on folium via geopandas.explore(), for actually looking around the county.
  Hovering a parcel draws its full-season NDVI sparkline (all dates loaded
  in ndvi_zonal_stats, not just the one being mapped) client-side in JS —
  the season history is embedded as plain numbers per parcel, and the SVG
  is built in the browser on hover rather than pre-rendered per parcel,
  which would balloon the file for no benefit.
"""

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import text

from src.db import get_engine

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def latest_date(engine):
    with engine.begin() as conn:
        return conn.execute(text("SELECT max(date) FROM ndvi_zonal_stats")).scalar()


def load_map_data(date=None):
    engine = get_engine()
    if date is None:
        date = latest_date(engine)

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

    # Full season-to-date series per parcel, for the hover sparkline —
    # independent of `date`, which only picks what colors the choropleth.
    series = pd.read_sql(
        text("SELECT pin, date, ndvi_mean FROM ndvi_zonal_stats "
             "WHERE ndvi_mean IS NOT NULL ORDER BY pin, date"),
        engine,
    )
    # json.dumps, not raw lists: geopandas' GeoJSON writer str()-ifies list-typed
    # columns using Python repr (single-quoted strings), which isn't valid JSON
    # and breaks JSON.parse() in the browser. A pre-serialized JSON string
    # survives that str() round-trip unchanged since it's already a str.
    by_pin = series.groupby("pin")
    gdf["ndvi_dates"] = gdf["pin"].map(
        lambda p: json.dumps(by_pin.get_group(p)["date"].tolist()) if p in by_pin.groups else "[]"
    )
    gdf["ndvi_series"] = gdf["pin"].map(
        lambda p: json.dumps([round(v, 3) for v in by_pin.get_group(p)["ndvi_mean"]]) if p in by_pin.groups else "[]"
    )
    return gdf, date


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


# Draws the sparkline itself in the browser on hover, from each feature's
# ndvi_dates/ndvi_series properties. Kept dependency-free (no Chart.js/
# Plotly) since this is a single self-contained HTML file.
SPARKLINE_JS = """
function ndviSparklineSvg(dates, values) {
    if (!values || values.length === 0) { return '<em>no data</em>'; }
    var w = 140, h = 40, pad = 4;
    var n = values.length;
    var x = function(i) { return n === 1 ? w / 2 : pad + i * (w - 2 * pad) / (n - 1); };
    var y = function(v) { return h - pad - v * (h - 2 * pad); };  // NDVI 0-1 fixed scale
    var pts = values.map(function(v, i) { return x(i) + ',' + y(v); }).join(' ');
    var last = values[values.length - 1];
    var color = last < 0.3 ? '#c0392b' : (last < 0.5 ? '#e67e22' : '#2c7a3f');
    var dots = values.map(function(v, i) {
        return '<circle cx="' + x(i) + '" cy="' + y(v) + '" r="2.5" fill="' + color + '"></circle>';
    }).join('');
    var labels = dates.map(function(d, i) {
        return '<text x="' + x(i) + '" y="' + (h - 1) + '" font-size="8" text-anchor="middle" fill="#555">' + d.slice(5) + '</text>';
    }).join('');
    return '<svg width="' + w + '" height="' + (h + 10) + '">' +
           '<polyline points="' + pts + '" fill="none" stroke="' + color + '" stroke-width="1.5"></polyline>' +
           dots + labels + '</svg>';
}
function bindNdviSparklines(map) {
    map.eachLayer(function(layer) {
        if (layer.feature && layer.feature.properties && 'ndvi_series' in layer.feature.properties) {
            var props = layer.feature.properties;
            var dates = JSON.parse(props.ndvi_dates);
            var series = JSON.parse(props.ndvi_series);
            var html = '<b>Parcel ' + props.pin + '</b><br>' +
                        'Latest NDVI: ' + props.ndvi_mean.toFixed(3) +
                        (props.is_anomaly ? ' &mdash; <span style="color:#c0392b">flagged anomalous</span>' : '') +
                        '<br>' + ndviSparklineSvg(dates, series);
            layer.bindTooltip(html, {sticky: true});
        }
    });
}
"""


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
        tooltip=False,  # replaced by the custom sparkline tooltip below
        popup=False,
        style_kwds={"weight": 0.5},
        name="Parcels (NDVI)",
    )

    import folium

    # Satellite layer for ground-truthing flagged parcels against what's
    # actually there (e.g. a plant or quarry, not a crop under stress).
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

    m.get_root().script.add_child(folium.Element(SPARKLINE_JS))
    m.get_root().script.add_child(folium.Element(f"bindNdviSparklines({m.get_name()});"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    import sys

    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    gdf, date = load_map_data(date_arg)
    print(f"Loaded {len(gdf)} parcels for {date} ({gdf['is_anomaly'].sum()} anomalous)", flush=True)
    plot_static_map(gdf, date, REPORTS_DIR / "county_ndvi_map.png")
    plot_interactive_map(gdf, REPORTS_DIR / "county_ndvi_map.html")
