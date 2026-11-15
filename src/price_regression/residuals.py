"""Residuals (actual - predicted) grouped by mileage band and body type.

Mean residual near 0 in every group means the straight-line model has no
systematic blind spot. A big positive/negative mean means the line is wrong there.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .pipeline import LABEL

KM_BANDS = ((0, 50_000), (50_000, 100_000), (100_000, 150_000), (150_000, 200_001))


def band_label(low: int, high: int) -> str:
    return f"{low // 1000}-{min(high, 200_000) // 1000}k km"


def _summarize(pred: DataFrame, group_col: str) -> list[tuple[str, int, float]]:
    rows = (
        pred.withColumn("residual", F.col(LABEL) - F.col("prediction"))
        .groupBy(group_col)
        .agg(F.count("*").alias("n"), F.avg("residual").alias("mean_residual"))
        .orderBy(group_col)
        .collect()
    )
    return [(str(r[group_col]), int(r["n"]), float(r["mean_residual"])) for r in rows]


def residuals_by_km(pred: DataFrame) -> list[tuple[str, int, float]]:
    """``[(band label, row count, mean residual), ...]`` in mileage order."""
    band = F.lit(None).cast("string")
    for i, (low, high) in reversed(list(enumerate(KM_BANDS))):
        band = F.when((F.col("km") >= low) & (F.col("km") < high), F.lit(f"{i}|{band_label(low, high)}")).otherwise(band)
    rows = _summarize(pred.withColumn("km_band", band), "km_band")
    return [(label.split("|", 1)[1], n, mean) for label, n, mean in rows]


def residuals_by_body(pred: DataFrame) -> list[tuple[str, int, float]]:
    return _summarize(pred, "body")
