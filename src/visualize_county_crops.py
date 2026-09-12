"""County-wide crop-type map: static choropleth, the full-scale counterpart
to visualize_subset.py's interactive map.

Static, not interactive, on purpose: visualize_subset.py's per-parcel
hover popup embeds each parcel's full NDVI curve directly in the page,
which works fine at 163 parcels (a 6-9MB file) but would balloon to
several hundred MB at ~9,578 parcels -- the same scale problem
visualize.py's county-wide anomaly map already solved by going static, and
the same solution here.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from sqlalchemy import text

from src.crop_clusters import cluster, extract_features, fit_splines, label_cluster, load_series
from src.db import get_engine

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

CLUSTER_COLORS = {
    "Corn-like (early peak, fast decline)": "#377eb8",
    "Soybean-like (later peak, slower decline)": "#4daf4a",
    "Non-row-crop (already green in April)": "#e41a1c",
}


def build_dataset(zonal_table="ndvi_zonal_stats_county_clipped",
                   parcels_table="parcels_clipped_county", bad_dates=frozenset()):
    df = load_series(table_name=zonal_table, bad_dates=bad_dates)
    splines, doy, pivot = fit_splines(df)
    feats = extract_features(splines, doy, pivot)
    feats, best_k = cluster(feats)

    cluster_means = feats.groupby("cluster").mean()
    label_by_id = {cid: label_cluster(row) for cid, row in cluster_means.iterrows()}
    feats["cluster_label"] = feats["cluster"].map(label_by_id)

    engine = get_engine()
    pins = feats.index.tolist()
    gdf = gpd.read_postgis(
        text(f"SELECT pin, geometry FROM {parcels_table} WHERE pin = ANY(:pins)"),
        engine, params={"pins": pins}, geom_col="geometry",
    )
    gdf = gdf.merge(feats.reset_index().rename(columns={"index": "pin"}), on="pin")
    return gdf, best_k


def plot_static_map(gdf, out_path):
    fig, ax = plt.subplots(figsize=(12, 11))
    for label, color in CLUSTER_COLORS.items():
        group = gdf[gdf["cluster_label"] == label]
        if len(group):
            group.plot(ax=ax, color=color, linewidth=0, label=f"{label} (n={len(group)})")
    ax.set_title(f"McLean County crop-type clusters (from NDVI curve shape), n={len(gdf)} parcels")
    ax.set_axis_off()
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    gdf, best_k = build_dataset()
    print(gdf["cluster_label"].value_counts(), flush=True)
    plot_static_map(gdf, REPORTS_DIR / "county_crop_map.png")
