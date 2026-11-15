"""Experiments: overfitting + regularization, cross-validation, the dummy-variable trap."""
from __future__ import annotations

from dataclasses import dataclass, field

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession

from .data import NOISE_COLUMNS, make_cars
from .evaluate import evaluate_model
from .pipeline import LABEL, build_pipeline, fit_model, named_coefficients

OVERFIT_TRAIN_ROWS = 15
OVERFIT_VALID_ROWS = 1000
L2_REG_PARAM = 1.0
L1_REG_PARAM = 20.0  # picked with seed 42: zeroes 6 of the 10 noise columns
REG_GRID = (0.0, 0.1, 1.0)
ELASTIC_GRID = (0.0, 1.0)


@dataclass
class OverfitRow:
    name: str  # "none" | "l2" | "l1"
    reg_param: float
    elastic_net: float
    train_rmse: float
    valid_rmse: float
    zero_noise: int  # how many of the noise coefficients are exactly 0.0


@dataclass
class OverfitResult:
    rows: list[OverfitRow]
    n_noise: int

    def by_name(self, name: str) -> OverfitRow:
        return next(r for r in self.rows if r.name == name)


def overfit_experiment(spark: SparkSession, seed: int = 42, n_noise: int = 10) -> OverfitResult:
    """Tiny training set + useless noise columns: the model memorizes instead of learning."""
    noise_cols = NOISE_COLUMNS[:n_noise]
    train = make_cars(spark, OVERFIT_TRAIN_ROWS, seed, n_noise=n_noise)
    valid = make_cars(spark, OVERFIT_VALID_ROWS, seed + 100, n_noise=n_noise)
    settings = [("none", 0.0, 0.0), ("l2", L2_REG_PARAM, 0.0), ("l1", L1_REG_PARAM, 1.0)]
    rows = []
    for name, reg, elastic in settings:
        model = fit_model(train, extra_cols=noise_cols, reg_param=reg, elastic_net=elastic)
        coefs = named_coefficients(model, noise_cols)
        zero = sum(1 for c in noise_cols if coefs[c] == 0.0)
        rows.append(
            OverfitRow(
                name, reg, elastic,
                train_rmse=evaluate_model(model, train).rmse,
                valid_rmse=evaluate_model(model, valid).rmse,
                zero_noise=zero,
            )
        )
    return OverfitResult(rows=rows, n_noise=n_noise)


@dataclass
class TuneResult:
    params: list[dict[str, float]]  # one dict per grid cell
    avg_rmse: list[float]
    best_index: int
    folds: int

    @property
    def best_params(self) -> dict[str, float]:
        return self.params[self.best_index]


def tune_with_cv(train: DataFrame, folds: int = 3, seed: int = 42) -> TuneResult:
    """Try every regParam x elasticNetParam cell with k-fold cross-validation."""
    pipeline = build_pipeline(scale=True)
    lr = pipeline.getStages()[-1]
    grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, list(REG_GRID))
        .addGrid(lr.elasticNetParam, list(ELASTIC_GRID))
        .build()
    )
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=RegressionEvaluator(labelCol=LABEL, metricName="rmse"),
        numFolds=folds,
        seed=seed,
    )
    model = cv.fit(train)
    params = [
        {"regParam": float(pm[lr.regParam]), "elasticNetParam": float(pm[lr.elasticNetParam])} for pm in grid
    ]
    avg = [float(m) for m in model.avgMetrics]
    best = min(range(len(avg)), key=avg.__getitem__)  # RMSE: smaller is better
    return TuneResult(params=params, avg_rmse=avg, best_index=best, folds=folds)


@dataclass
class TrapResult:
    keep_coefficients: dict[str, float]
    keep_pvalues_error: str | None  # message when p-values are impossible, None if they worked
    keep_max_std_error: float | None  # absurdly large when p-values "work" on a singular system
    error_coefficients: dict[str, float]
    error_pvalue_count: int
    error_coefficient_count: int
    notes: list[str] = field(default_factory=list)


def dummy_trap(spark: SparkSession, train: DataFrame) -> TrapResult:
    """``handleInvalid="keep"`` + ``dropLast`` makes the body columns add up to 1 = collinear with the intercept."""
    sc = spark.sparkContext
    sc.setLogLevel("OFF")  # the failing solver step logs scary (harmless) errors
    try:
        # solver="auto": the singular system makes Spark fall back to L-BFGS, which has no p-values
        keep = fit_model(train, handle_invalid="keep")
        keep_error: str | None
        keep_se: float | None = None
        try:
            keep.stages[-1].summary.pValues
            keep_error = None
            # Numerically the singular system sometimes still "solves": the standard errors expose it.
            keep_se = float(max(keep.stages[-1].summary.coefficientStandardErrors))
        except Exception as exc:  # noqa: BLE001 - Spark raises a Java exception wrapper
            keep_error = str(exc).strip().splitlines()[0][:200] or type(exc).__name__
        strict = fit_model(train, handle_invalid="error", solver="normal")
        summary = strict.stages[-1].summary
        return TrapResult(
            keep_coefficients=named_coefficients(keep),
            keep_pvalues_error=keep_error,
            keep_max_std_error=keep_se,
            error_coefficients=named_coefficients(strict),
            error_pvalue_count=len(summary.pValues),
            error_coefficient_count=len(strict.stages[-1].coefficients),
        )
    finally:
        sc.setLogLevel("ERROR")
