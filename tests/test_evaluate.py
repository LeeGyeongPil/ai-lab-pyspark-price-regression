import math

import pytest

from price_regression.evaluate import (
    baseline_metrics, evaluate_predictions, improvement, train_summary_metrics,
)


def test_metrics_match_manual_and_baseline(split, model):
    """AC-3: Evaluator numbers equal a hand calculation; the baseline has R2 near 0."""
    train, test = split
    rows = model.transform(test).select("price", "prediction").collect()
    errors = [r["price"] - r["prediction"] for r in rows]
    mean_y = sum(r["price"] for r in rows) / len(rows)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    mae = sum(abs(e) for e in errors) / len(errors)
    r2 = 1 - sum(e * e for e in errors) / sum((r["price"] - mean_y) ** 2 for r in rows)

    got = evaluate_predictions(model.transform(test))
    assert got.rmse == pytest.approx(rmse, abs=1e-6)
    assert got.mae == pytest.approx(mae, abs=1e-6)
    assert got.r2 == pytest.approx(r2, abs=1e-6)

    base = baseline_metrics(train, test)
    assert abs(base.r2) < 0.02
    assert base.rmse > 5 * got.rmse


def test_training_summary_is_not_the_test_score(split, model):
    _, test = split
    train_m = train_summary_metrics(model)
    test_m = evaluate_predictions(model.transform(test))
    assert train_m.rmse != test_m.rmse  # two different data sets, two different numbers


def test_improvement_edge_cases():
    assert improvement(100, 1000) == pytest.approx(0.9)
    assert improvement(100, 0) == 0.0
