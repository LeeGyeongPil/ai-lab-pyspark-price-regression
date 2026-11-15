"""StringIndexer -> OneHotEncoder -> VectorAssembler -> LinearRegression."""
from __future__ import annotations

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame

LABEL = "price"
NUMERIC = ("age", "km")
TRAIN_FRACTION = 0.8


def split_train_test(df: DataFrame, seed: int = 42) -> tuple[DataFrame, DataFrame]:
    """80/20 split. The same seed always gives the same rows on each side."""
    train, test = df.randomSplit([TRAIN_FRACTION, 1 - TRAIN_FRACTION], seed=seed)
    return train.cache(), test.cache()


def build_pipeline(
    handle_invalid: str = "error",
    scale: bool = False,
    extra_cols: tuple[str, ...] = (),
    reg_param: float = 0.0,
    elastic_net: float = 0.0,
    solver: str = "auto",
) -> Pipeline:
    """Build the regression pipeline.

    * ``alphabetDesc`` fixes the label order to [suv, sedan, hatch], so the
      reference (dropped) category is always ``hatch`` - not whatever is most
      frequent in this particular data.
    * ``scale=True`` adds StandardScaler (coefficients are then NOT in original units).
    """
    indexer = StringIndexer(
        inputCol="body", outputCol="body_idx",
        stringOrderType="alphabetDesc", handleInvalid=handle_invalid,
    )
    encoder = OneHotEncoder(inputCols=["body_idx"], outputCols=["body_ohe"], dropLast=True)
    assembled = "raw_features" if scale else "features"
    assembler = VectorAssembler(inputCols=[*NUMERIC, *extra_cols, "body_ohe"], outputCol=assembled)
    stages = [indexer, encoder, assembler]
    if scale:
        stages.append(StandardScaler(inputCol=assembled, outputCol="features", withMean=False, withStd=True))
    stages.append(
        LinearRegression(
            featuresCol="features", labelCol=LABEL,
            regParam=reg_param, elasticNetParam=elastic_net, solver=solver,
        )
    )
    return Pipeline(stages=stages)


def fit_model(train: DataFrame, **kwargs) -> PipelineModel:
    return build_pipeline(**kwargs).fit(train)


def body_labels(model: PipelineModel) -> list[str]:
    """Body types the indexer saw while fitting, in index order."""
    return list(model.stages[0].labels)


def feature_names(model: PipelineModel, extra_cols: tuple[str, ...] = ()) -> list[str]:
    """Names for each entry of ``coefficients`` (numeric columns first, then one-hot body columns)."""
    numeric = [*NUMERIC, *extra_cols]
    coefs = model.stages[-1].coefficients
    n_body = len(coefs) - len(numeric)
    return numeric + [f"body={label}" for label in body_labels(model)[:n_body]]


def named_coefficients(model: PipelineModel, extra_cols: tuple[str, ...] = ()) -> dict[str, float]:
    """``{"intercept": ..., "age": ..., ...}`` as plain Python floats (no ``np.float64`` in output)."""
    lr = model.stages[-1]
    out = {"intercept": float(lr.intercept)}
    for name, value in zip(feature_names(model, extra_cols), lr.coefficients):
        out[name] = float(value)
    return out
