"""Interactive map for the crop-type subset area: each parcel colored by
its behavioral cluster from crop_clusters.py, with its actual measured
NDVI season curve and a plain-language cluster description on both hover
and click.

Separate from visualize.py's county-wide map -- different area (163
parcels vs 9,578), different question (crop-type clustering vs anomaly
detection), categorical coloring instead of continuous NDVI.

Cluster IDs from KMeans are arbitrary and can change on a rerun, so labels
are derived from each cluster's own feature means (early_ndvi, peak_doy)
rather than hardcoded by ID. These are still behavioral labels, not
validated crop identifications -- no ground truth exists for this subset.
"""

import json
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from sqlalchemy import text

from src.crop_clusters import cluster, extract_features, fit_splines, load_series
from src.db import get_engine

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def label_cluster(row):
    """Behavioral label from a cluster's own feature means -- not a fixed
    ID mapping, since KMeans cluster numbering is arbitrary per run."""
    if row["early_ndvi"] > 0.3:
        return "Non-row-crop (already green in April)"
    if row["peak_doy"] < 210:
        return "Corn-like (early peak, fast decline)"
    return "Soybean-like (later peak, slower decline)"


def build_dataset():
    df = load_series()
    splines, doy, pivot = fit_splines(df)
    feats = extract_features(splines, doy, pivot)
    feats, best_k = cluster(feats)

    cluster_means = feats.groupby("cluster").mean()
    label_by_id = {cid: label_cluster(row) for cid, row in cluster_means.iterrows()}
    feats["cluster_label"] = feats["cluster"].map(label_by_id)

    engine = get_engine()
    pins = feats.index.tolist()
    gdf = gpd.read_postgis(
        text("SELECT pin, geometry FROM parcels WHERE pin = ANY(:pins)"),
        engine, params={"pins": pins}, geom_col="geometry",
    )
    gdf = gdf.merge(feats.reset_index().rename(columns={"index": "pin"}), on="pin")

    dates_list = [d.strftime("%Y-%m-%d") for d in pivot.columns]
    gdf["ndvi_dates"] = json.dumps(dates_list)
    gdf["ndvi_series"] = gdf["pin"].map(lambda p: json.dumps([round(v, 3) for v in pivot.loc[p].tolist()]))
    return gdf


# Same client-side sparkline approach as visualize.py's county map, extended
# to show the cluster label. Kept dependency-free (no Chart.js/Plotly).
SPARKLINE_JS = """
function ndviSparklineSvg(dates, values) {
    if (!values || values.length === 0) { return '<em>no data</em>'; }
    var w = 160, h = 45, pad = 4;
    var n = values.length;
    var x = function(i) { return n === 1 ? w / 2 : pad + i * (w - 2 * pad) / (n - 1); };
    var y = function(v) { return h - pad - v * (h - 2 * pad); };
    var pts = values.map(function(v, i) { return x(i) + ',' + y(v); }).join(' ');
    var dots = values.map(function(v, i) {
        return '<circle cx="' + x(i) + '" cy="' + y(v) + '" r="2.5" fill="#333"></circle>';
    }).join('');
    var labels = dates.map(function(d, i) {
        return '<text x="' + x(i) + '" y="' + (h - 1) + '" font-size="7.5" text-anchor="middle" fill="#555">' + d.slice(5) + '</text>';
    }).join('');
    return '<svg width="' + w + '" height="' + (h + 10) + '">' +
           '<polyline points="' + pts + '" fill="none" stroke="#333" stroke-width="1.5"></polyline>' +
           dots + labels + '</svg>';
}
function bindSubsetTooltips(map) {
    map.eachLayer(function(layer) {
        if (layer.feature && layer.feature.properties && 'ndvi_series' in layer.feature.properties) {
            var props = layer.feature.properties;
            var dates = JSON.parse(props.ndvi_dates);
            var series = JSON.parse(props.ndvi_series);
            var html = '<b>Parcel ' + props.pin + '</b><br>' +
                        '<b>' + props.cluster_label + '</b><br>' +
                        ndviSparklineSvg(dates, series);
            layer.bindTooltip(html, {sticky: true});
            layer.bindPopup(html);
        }
    });
}
"""


def plot_interactive_map(gdf, out_path):
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].simplify(0.00001)  # subset area is tiny; light simplify only

    # cmap as a name, not an explicit color list: geopandas assigns colors
    # to categories in its own sorted order, and a hand-built list risks
    # silently mismatching category-to-color if that order isn't exactly
    # what's assumed.
    m = gdf.explore(
        column="cluster_label",
        categorical=True,
        cmap="Set1",
        tooltip=False,
        popup=False,
        style_kwds={"weight": 1},
        name="Parcels (crop-type cluster)",
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.LayerControl(collapsed=False, position="bottomright").add_to(m)

    m.get_root().script.add_child(folium.Element(SPARKLINE_JS))
    m.get_root().script.add_child(folium.Element(
        f"window.addEventListener('load', function() {{ bindSubsetTooltips({m.get_name()}); }});"
    ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    gdf = build_dataset()
    print(gdf["cluster_label"].value_counts(), flush=True)
    plot_interactive_map(gdf, REPORTS_DIR / "subset_crop_map.html")
