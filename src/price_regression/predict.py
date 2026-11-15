"""Save/load the fitted pipeline and predict the price of one new car."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from .data import AGE_RANGE, KM_RANGE, schema
from .pipeline import body_labels


class UnknownBodyError(ValueError):
    """The body type was never seen while training."""

    def __init__(self, body: str, known: list[str]):
        super().__init__(f"unknown body type {body!r}; known: {', '.join(known)}")
        self.body = body
        self.known = known


@dataclass
class Prediction:
    price: float
    warnings: list[str] = field(default_factory=list)  # message keys, translated by the CLI


def model_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "model"


def save_model(model: PipelineModel, out_dir: str | Path) -> Path:
    path = model_path(out_dir)
    model.write().overwrite().save(str(path))
    return path


def load_model(out_dir: str | Path) -> PipelineModel | None:
    path = model_path(out_dir)
    return PipelineModel.load(str(path)) if path.exists() else None


def predict_price(spark: SparkSession, model: PipelineModel, age: float, km: float, body: str) -> Prediction:
    """Predict one car. Raises :class:`UnknownBodyError` for a body type the model never saw."""
    known = body_labels(model)
    if body not in known:
        raise UnknownBodyError(body, known)
    row = spark.createDataFrame([(0, float(age), float(km), body, 0.0)], schema())
    price = float(model.transform(row).first()["prediction"])
    warnings = []
    if not AGE_RANGE[0] <= age <= AGE_RANGE[1]:
        warnings.append("warn_age_range")
    if not KM_RANGE[0] <= km <= KM_RANGE[1]:
        warnings.append("warn_km_range")
    if price < 0:
        warnings.append("warn_negative_price")
    return Prediction(price=price, warnings=warnings)
