"""Synthetic used-car data with a *known* price formula.

Because we know the formula, we can later check whether the model learned it.
Prices are in units of 10,000 KRW ("man won"): 3000 means 30,000,000 KRW.
"""
from __future__ import annotations

import random

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

BODIES = ("hatch", "sedan", "suv")
BODY_BONUS = {"hatch": -400.0, "sedan": 0.0, "suv": 800.0}

AGE_RANGE = (0, 15)
KM_RANGE = (0, 200_000)

# The hidden formula: price = 3000 - 150*age - 0.02*km + body bonus + noise
TRUE_FORMULA = {
    "intercept": 3000.0,
    "age": -150.0,
    "km": -0.02,
    "noise_sd": 120.0,
    "nonlinear_from_km": 150_000,
    "nonlinear_slope": -0.04,  # extra depreciation per km above 150,000 (--nonlinear)
}

NOISE_COLUMNS = tuple(f"noise_{i}" for i in range(10))
PARTITIONS = 4  # fixed so that randomSplit(seed) gives the same split everywhere


def true_coefficients() -> dict[str, float]:
    """The formula rewritten with ``hatch`` as the reference body type.

    The model uses hatch as the reference category, so every other body type is
    "how much more than a hatchback", and the intercept absorbs hatch's bonus.
    """
    ref = BODY_BONUS["hatch"]
    return {
        "intercept": TRUE_FORMULA["intercept"] + ref,  # 2600
        "age": TRUE_FORMULA["age"],
        "km": TRUE_FORMULA["km"],
        "body=suv": BODY_BONUS["suv"] - ref,  # 1200
        "body=sedan": BODY_BONUS["sedan"] - ref,  # 400
    }


def true_price(age: float, km: float, body: str, nonlinear: bool = False) -> float:
    """Noise-free price from the hidden formula."""
    price = TRUE_FORMULA["intercept"] + TRUE_FORMULA["age"] * age + TRUE_FORMULA["km"] * km
    price += BODY_BONUS[body]
    if nonlinear and km > TRUE_FORMULA["nonlinear_from_km"]:
        price += TRUE_FORMULA["nonlinear_slope"] * (km - TRUE_FORMULA["nonlinear_from_km"])
    return price


def schema(n_noise: int = 0) -> StructType:
    fields = [
        StructField("id", LongType(), False),
        StructField("age", DoubleType(), False),
        StructField("km", DoubleType(), False),
        StructField("body", StringType(), False),
    ]
    fields += [StructField(c, DoubleType(), False) for c in NOISE_COLUMNS[:n_noise]]
    fields.append(StructField("price", DoubleType(), False))
    return StructType(fields)


def make_rows(rows: int, seed: int, nonlinear: bool = False, n_noise: int = 0) -> list[tuple]:
    """Generate rows with plain Python ``random`` so results never depend on Spark."""
    if rows < 1:
        raise ValueError("rows must be at least 1")
    if not 0 <= n_noise <= len(NOISE_COLUMNS):
        raise ValueError(f"n_noise must be between 0 and {len(NOISE_COLUMNS)}")
    rng = random.Random(seed)
    noise_rng = random.Random(seed + 1)  # separate stream: noise features never change prices
    out = []
    for i in range(rows):
        age = float(rng.randint(*AGE_RANGE))
        km = float(rng.randint(*KM_RANGE))
        body = rng.choice(BODIES)
        price = true_price(age, km, body, nonlinear) + rng.gauss(0.0, TRUE_FORMULA["noise_sd"])
        noise = [noise_rng.gauss(0.0, 1.0) for _ in range(n_noise)]
        out.append((i, age, km, body, *noise, float(price)))
    return out


def make_cars(
    spark: SparkSession, rows: int = 5000, seed: int = 42, nonlinear: bool = False, n_noise: int = 0
) -> DataFrame:
    """Spark DataFrame ``id, age, km, body[, noise_0..], price`` (cached, 4 partitions)."""
    data = make_rows(rows, seed, nonlinear, n_noise)
    rdd = spark.sparkContext.parallelize(data, PARTITIONS)
    return spark.createDataFrame(rdd, schema(n_noise)).cache()
