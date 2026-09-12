"""County-wide crop-type map: the full-scale counterpart to
visualize_subset.py's interactive map, now built the same way -- each
parcel colored by its behavioral cluster, click for its actual NDVI curve,
cluster label, assignment confidence, and yield estimate -- scaled from
the 163-parcel subset to all ~9,571 county parcels.

An earlier version produced only a static PNG here, on the assumption
that embedding a full NDVI curve per parcel (as visualize_subset.py does)
would balloon the file to several hundred MB at this scale. That estimate
was never actually measured against this codebase's own precedent:
visualize.py's county-wide anomaly map already embeds a full per-parcel
NDVI time series (not just a single value) for all ~9,571 parcels and
comes in at ~7MB. Measured directly, the interactive version here (curve
data plus simplified clipped-parcel geometry) lands in the same
tens-of-MB range -- large for a single static HTML file, but well within
what a browser (and GitHub Pages) handles fine, and worth it for the same
reason the subset map was built this way: seeing the actual measured curve
behind a cluster assignment is more convincing than a color alone.

Both outputs are still produced from the same query: the static PNG stays
as the lightweight image embedded directly in the README, and the
interactive HTML is the one to actually click around in.
"""

from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
from sqlalchemy import text

from src.crop_clusters import (
    CLUSTER_LABELS,
    cluster,
    extract_features,
    fit_splines,
    load_series,
    popup_curve_data,
)
from src.db import get_engine
from src.exclude_urban import filter_rural
from src.yield_ranking import (
    estimate_total_bushels,
    estimate_yield_bu_ac,
    fetch_acreage,
    rank_within_cluster,
    seasonal_ndvi_integral,
)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

CLUSTER_COLORS = {
    "Corn-like (early peak, fast decline)": "#377eb8",
    "Soybean-like (later peak, slower decline)": "#4daf4a",
    "Non-row-crop (already green in April)": "#e41a1c",
}


def build_dataset(zonal_table="evi2_zonal_stats_county_clipped", value_column="evi2_mean",
                   parcels_table="parcels_clipped_county", bad_dates=frozenset(), exclude_urban=True):
    df = load_series(table_name=zonal_table, bad_dates=bad_dates, value_column=value_column)
    if exclude_urban:
        # Bloomington/Normal parcels excluded before clustering, not just
        # hidden on the map afterward -- see exclude_urban.py. Requires
        # `python -m src.exclude_urban` to have populated the urban_areas
        # table first.
        before = df["pin"].nunique()
        df = filter_rural(df, parcels_table=parcels_table)
        print(f"Urban exclusion: {before - df['pin'].nunique()} parcels dropped "
              f"(inside Bloomington/Normal city limits)", flush=True)
    splines, kept_series = fit_splines(df)
    feats = extract_features(splines, kept_series)
    feats = cluster(feats)

    feats["cluster_label"] = feats["cluster"].map(CLUSTER_LABELS)

    integrals = seasonal_ndvi_integral(splines, kept_series)
    feats["seasonal_ndvi_integral"] = feats.index.map(integrals)
    feats = rank_within_cluster(feats)
    feats = estimate_yield_bu_ac(feats)
    feats = estimate_total_bushels(feats, fetch_acreage(feats.index, parcels_table=parcels_table))

    engine = get_engine()
    pins = feats.index.tolist()
    gdf = gpd.read_postgis(
        text(f"SELECT pin, geometry FROM {parcels_table} WHERE pin = ANY(:pins)"),
        engine, params={"pins": pins}, geom_col="geometry",
    )
    gdf = gdf.merge(feats.reset_index().rename(columns={"index": "pin"}), on="pin")

    popup_data = popup_curve_data(splines, kept_series)
    gdf = gdf.merge(popup_data, left_on="pin", right_index=True)
    return gdf


def plot_static_map(gdf, out_path):
    fig, ax = plt.subplots(figsize=(12, 11))
    for label, color in CLUSTER_COLORS.items():
        group = gdf[gdf["cluster_label"] == label]
        if len(group):
            group.plot(ax=ax, color=color, linewidth=0, label=f"{label} (n={len(group)})")
    ax.set_title(f"McLean County crop-type clusters (from EVI2 curve shape), n={len(gdf)} parcels")
    ax.set_axis_off()
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}", flush=True)


# Identical to visualize_subset.py's popup JS -- same curve-data shape
# (popup_curve_data, crop_clusters.py), same fields, same design (click,
# not hover, to avoid the double-fire touch-device issue noted there; no
# per-date dots, since UnivariateSpline smooths past noise rather than
# passing through every point -- see popup_curve_data).
SPARKLINE_JS = """
function ndviSmoothSparklineSvg(denseValues, denseStartDoy) {
    if (!denseValues || denseValues.length === 0) { return '<em>no data</em>'; }
    var w = 180, h = 50, pad = 4;
    var minDoy = denseStartDoy, maxDoy = denseStartDoy + denseValues.length - 1;
    var x = function(doy) { return pad + (doy - minDoy) * (w - 2 * pad) / (maxDoy - minDoy); };
    var y = function(v) { return h - pad - v * (h - 2 * pad); };
    var linePts = denseValues.map(function(v, i) { return x(minDoy + i) + ',' + y(v); }).join(' ');
    return '<svg width="' + w + '" height="' + (h + 4) + '">' +
           '<polyline points="' + linePts + '" fill="none" stroke="#333" stroke-width="1.5"></polyline>' +
           '</svg>';
}
function bindCountyCropPopups(map) {
    map.eachLayer(function(layer) {
        if (layer.feature && layer.feature.properties && 'ndvi_dense' in layer.feature.properties) {
            var props = layer.feature.properties;
            var dense = JSON.parse(props.ndvi_dense);
            var confPct = Math.round(props.confidence * 100);
            var confColor = confPct >= 75 ? '#2c7a3f' : (confPct >= 60 ? '#e67e22' : '#c0392b');
            var confWord = confPct >= 75 ? 'high' : (confPct >= 60 ? 'moderate' : 'low');

            var yieldHtml = '';
            if (props.estimated_total_bushels !== null) {
                yieldHtml = '<br>Est. yield: <b>' + Math.round(props.estimated_total_bushels).toLocaleString() + ' bu</b>' +
                            ' (' + props.estimated_yield_bu_ac.toFixed(0) + ' bu/ac &times; ' + props.acres.toFixed(0) + ' ac)' +
                            '<br>' + Math.round(props.percentile_in_cluster) + 'th percentile in cluster' +
                            '<br><span style="font-size:10px;color:#777">approximate -- see README</span>';
            }

            var html = '<b>Parcel ' + props.pin + '</b><br>' +
                        '<b>' + props.cluster_label + '</b><br>' +
                        '<span style="color:' + confColor + '">' + confPct + '% confidence (' + confWord + ')</span>' +
                        ' vs. next-closest group' + yieldHtml + '<br>' +
                        ndviSmoothSparklineSvg(dense, props.ndvi_dense_start_doy) +
                        '<div style="font-size:10px;color:#777">N = ' + props.n_dates + ' valid dates</div>';
            layer.bindPopup(html);
        }
    });
}
"""


def plot_interactive_map(gdf, out_path):
    gdf = gdf.copy()
    # Heavier than the subset map's 0.00001 -- 9,571 parcels' worth of
    # ST_Difference-clipped boundaries (roads/waterways cut out, which adds
    # vertices right where they were subtracted) is the dominant contributor
    # to file size at this scale, and full survey precision isn't visible at
    # county zoom anyway. visualize.py's county-wide anomaly map uses this
    # same tolerance for the same reason.
    gdf["geometry"] = gdf["geometry"].simplify(0.00003)

    m = gdf.explore(
        column="cluster_label",
        categorical=True,
        cmap="Set1",
        tooltip=False,
        popup=False,
        style_kwds={"weight": 0.5},
        name="Parcels (crop-type cluster)",
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    m.get_root().script.add_child(folium.Element(SPARKLINE_JS))
    m.get_root().script.add_child(folium.Element(
        f"window.addEventListener('load', function() {{ bindCountyCropPopups({m.get_name()}); }});"
    ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    gdf = build_dataset()
    print(gdf["cluster_label"].value_counts(), flush=True)
    plot_static_map(gdf, REPORTS_DIR / "county_crop_map.png")
    plot_interactive_map(gdf, REPORTS_DIR / "county_crop_map.html")
