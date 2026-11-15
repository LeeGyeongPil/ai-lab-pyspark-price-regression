"""Metrics: RMSE, MAE, R2 on held-out data, plus the "just guess the mean" baseline."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .pipeline import LABEL


@dataclass(frozen=True)
class Metrics:
    rmse: float
    mae: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_predictions(pred: DataFrame, label_col: str = LABEL) -> Metrics:
    """Score a DataFrame that has ``label_col`` and ``prediction`` columns."""
    def metric(name: str) -> float:
        evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName=name)
        return float(evaluator.evaluate(pred))

    return Metrics(rmse=metric("rmse"), mae=metric("mae"), r2=metric("r2"))


def evaluate_model(model: PipelineModel, data: DataFrame) -> Metrics:
    return evaluate_predictions(model.transform(data))


def baseline_metrics(train: DataFrame, test: DataFrame) -> Metrics:
    """Baseline model: always predict the *training* mean price."""
    mean_price = float(train.agg(F.avg(LABEL)).first()[0])
    return evaluate_predictions(test.withColumn("prediction", F.lit(mean_price)))


def train_summary_metrics(model: PipelineModel) -> Metrics:
    """Metrics Spark computed on the TRAINING data (``model.summary``). Not a test score."""
    summary = model.stages[-1].summary
    return Metrics(
        rmse=float(summary.rootMeanSquaredError),
        mae=float(summary.meanAbsoluteError),
        r2=float(summary.r2),
    )


def improvement(model_rmse: float, baseline_rmse: float) -> float:
    """Share of the baseline error that the model removed (0.92 = 92% smaller error)."""
    if baseline_rmse <= 0:
        return 0.0
    return 1.0 - model_rmse / baseline_rmse
