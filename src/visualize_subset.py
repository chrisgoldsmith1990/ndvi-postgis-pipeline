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
import numpy as np
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

    # Dense curve (one value per day, from the PCHIP fit) for a genuinely
    # smooth line, plus the actual 8 raw measurements so the popup can
    # still show what was measured vs. interpolated -- same distinction
    # crop_clusters.png draws (thin line = fit, dots = observed).
    dense_doy = np.arange(int(doy.min()), int(doy.max()) + 1)
    gdf["ndvi_dense_start_doy"] = int(dense_doy.min())
    gdf["ndvi_dense"] = gdf["pin"].map(lambda p: json.dumps([round(v, 3) for v in splines[p](dense_doy)]))
    gdf["ndvi_raw_doy"] = json.dumps([int(d) for d in doy])
    gdf["ndvi_raw_dates"] = json.dumps([d.strftime("%Y-%m-%d") for d in pivot.columns])
    gdf["ndvi_raw_values"] = gdf["pin"].map(lambda p: json.dumps([round(v, 3) for v in pivot.loc[p].tolist()]))
    return gdf


# Draws the dense PCHIP-fitted curve (one point per day -- genuinely smooth,
# not a straight line between 8 raw samples) with small dots marking the
# actual measured dates, so it's clear what was observed vs. interpolated.
# Bound only to click/tap (bindPopup), not also to hover (bindTooltip):
# binding both was firing twice on a single tap on touch devices, since a
# tap can synthesize both a hover and a click event.
SPARKLINE_JS = """
function ndviSmoothSparklineSvg(denseValues, denseStartDoy, rawDoy, rawValues) {
    if (!denseValues || denseValues.length === 0) { return '<em>no data</em>'; }
    var w = 180, h = 50, pad = 4;
    var minDoy = denseStartDoy, maxDoy = denseStartDoy + denseValues.length - 1;
    var x = function(doy) { return pad + (doy - minDoy) * (w - 2 * pad) / (maxDoy - minDoy); };
    var y = function(v) { return h - pad - v * (h - 2 * pad); };
    var linePts = denseValues.map(function(v, i) { return x(minDoy + i) + ',' + y(v); }).join(' ');
    var dots = rawDoy.map(function(d, i) {
        return '<circle cx="' + x(d) + '" cy="' + y(rawValues[i]) + '" r="2.5" fill="#333"></circle>';
    }).join('');
    return '<svg width="' + w + '" height="' + (h + 4) + '">' +
           '<polyline points="' + linePts + '" fill="none" stroke="#333" stroke-width="1.5"></polyline>' +
           dots + '</svg>';
}
function bindSubsetPopups(map) {
    map.eachLayer(function(layer) {
        if (layer.feature && layer.feature.properties && 'ndvi_dense' in layer.feature.properties) {
            var props = layer.feature.properties;
            var dense = JSON.parse(props.ndvi_dense);
            var rawDoy = JSON.parse(props.ndvi_raw_doy);
            var rawValues = JSON.parse(props.ndvi_raw_values);
            var html = '<b>Parcel ' + props.pin + '</b><br>' +
                        '<b>' + props.cluster_label + '</b><br>' +
                        ndviSmoothSparklineSvg(dense, props.ndvi_dense_start_doy, rawDoy, rawValues);
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
    # topright, not bottomright: unlike visualize.py's continuous colorbar
    # (a real Leaflet control, docks via the corner-stacking system), a
    # *categorical* legend from geopandas.explore() renders as a plain
    # `.maplegend` div with hardcoded `position: fixed; right: 10px;
    # bottom: 20px` -- it doesn't know about Leaflet controls at all, so it
    # can't stack with them, it just sits on top. Putting the layer control
    # in the one fixed corner (topright) the legend never touches.
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    m.get_root().script.add_child(folium.Element(SPARKLINE_JS))
    m.get_root().script.add_child(folium.Element(
        f"window.addEventListener('load', function() {{ bindSubsetPopups({m.get_name()}); }});"
    ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    gdf = build_dataset()
    print(gdf["cluster_label"].value_counts(), flush=True)
    plot_interactive_map(gdf, REPORTS_DIR / "subset_crop_map.html")
