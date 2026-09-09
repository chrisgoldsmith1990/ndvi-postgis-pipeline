"""Step 6: time series + a simple anomaly flag.

The flag is relative, not absolute: z-score each field's NDVI deviation
from its neighbor mean (`ndvi_diff`, from neighbor_comparison.py) against
the distribution of that same deviation across all fields *on that date*.
An absolute NDVI cutoff (e.g. "flag anything under 0.5") would fire on
every field in a cool wet June and stay silent in a drought August, because
the whole county's NDVI floor moves with the season and the weather.
Z-scoring within a date cancels that out — what's left is "unusually far
from its immediate neighbors," which is the thing actually worth a look.

A single below-threshold date is flagged as a one-off — cloud contamination
at the pixel level, a registration nudge between tiles, a field that was
freshly cut that week. Persistent anomaly (flagged on every date observed)
is the stronger, still-simple signal this reports: a field that stays
anomalous across three separate acquisitions, three different composite
sources, and a growing season's worth of weather is much more likely to be
a genuine standing difference — non-cropland, drainage issues, or real
season-long crop stress — than noise.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import text

from src.db import get_engine

Z_THRESHOLD = -2.0  # ~2 standard deviations below the neighborhood mean deviation
# Unlike data/raw and data/processed (gitignored — regenerable from source
# imagery), reports/ is committed: it's the portfolio-facing output.
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def compute_anomalies(z_threshold=Z_THRESHOLD):
    engine = get_engine()
    df = pd.read_sql(text("SELECT pin, date, field_ndvi, neighbor_mean_ndvi, ndvi_diff FROM neighbor_comparison"), engine)
    df["z"] = df.groupby("date")["ndvi_diff"].transform(lambda s: (s - s.mean()) / s.std())
    df["is_anomaly"] = df["z"] < z_threshold

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS field_anomalies"))
    df.to_sql("field_anomalies", engine, if_exists="replace", index=False)

    n_dates = df["date"].nunique()
    persistent = (
        df[df["is_anomaly"]].groupby("pin").size()
        .loc[lambda s: s == n_dates]
        .index
    )
    print(f"{df['is_anomaly'].sum()} field-date rows flagged (z < {z_threshold}); "
          f"{len(persistent)} fields flagged on all {n_dates} dates", flush=True)
    return df, list(persistent)


def plot_field_vs_neighbors(df, pin, out_path):
    field = df[df["pin"] == pin].sort_values("date")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(field["date"], field["field_ndvi"], marker="o", label=f"Parcel {pin}", color="#c0392b")
    ax.plot(field["date"], field["neighbor_mean_ndvi"], marker="o", label="Neighbor mean (8 nearest)", color="#2c7a3f")
    ax.set_ylabel("Mean NDVI")
    ax.set_title(f"Parcel {pin} vs. its spatial neighbors")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    df, persistent = compute_anomalies()
    if persistent:
        # Most anomalous by average z-score across dates.
        worst = (
            df[df["pin"].isin(persistent)].groupby("pin")["z"].mean()
            .sort_values().index[0]
        )
        plot_field_vs_neighbors(df, worst, REPORTS_DIR / "anomaly_example.png")
