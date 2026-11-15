from price_regression.data import make_cars
from price_regression.pipeline import fit_model
from price_regression.residuals import residuals_by_body, residuals_by_km


def _band_means(spark, nonlinear):
    df = make_cars(spark, rows=1000, seed=42, nonlinear=nonlinear)
    pred = fit_model(df).transform(df)  # all 1000 rows, about 250 per mileage band
    return {label: mean for label, _, mean in residuals_by_km(pred)}, pred


def test_residuals_reveal_high_km_bias(spark):
    """AC-4: straight-line data -> residuals near 0; bent data -> big misses in the bent bands."""
    linear, _ = _band_means(spark, nonlinear=False)
    assert len(linear) == 4
    assert all(abs(m) < 40 for m in linear.values())

    bent, _ = _band_means(spark, nonlinear=True)
    assert bent["150-200k km"] < -150
    assert bent["100-150k km"] > 150


def test_residuals_by_body_cover_all_bodies(spark):
    _, pred = _band_means(spark, nonlinear=False)
    assert [b for b, _, _ in residuals_by_body(pred)] == ["hatch", "sedan", "suv"]
