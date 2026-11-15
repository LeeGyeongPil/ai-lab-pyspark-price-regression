"""Share ONE Spark session (and one small data set) across all tests: starting Spark takes seconds."""
from __future__ import annotations

import pytest

from price_regression.data import make_cars
from price_regression.pipeline import fit_model, split_train_test
from price_regression.session import get_spark, stop_spark


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    # tmp folder under pytest's tmp dir: works even where the default temp folder is not writable
    session = get_spark(master="local[2]", tmp_dir=tmp_path_factory.mktemp("spark"))
    yield session
    stop_spark()


@pytest.fixture(scope="session")
def cars(spark):
    return make_cars(spark, rows=1000, seed=42)


@pytest.fixture(scope="session")
def split(cars):
    return split_train_test(cars, seed=42)


@pytest.fixture(scope="session")
def model(split):
    return fit_model(split[0], solver="normal")
