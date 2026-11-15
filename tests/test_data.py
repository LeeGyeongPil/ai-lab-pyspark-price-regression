import pytest

from price_regression.data import AGE_RANGE, BODIES, KM_RANGE, make_rows, true_coefficients, true_price


def test_same_seed_same_rows():
    assert make_rows(100, seed=1) == make_rows(100, seed=1)
    assert make_rows(100, seed=1) != make_rows(100, seed=2)


def test_values_stay_in_range():
    for _id, age, km, body, price in make_rows(500, seed=3):
        assert AGE_RANGE[0] <= age <= AGE_RANGE[1]
        assert KM_RANGE[0] <= km <= KM_RANGE[1]
        assert body in BODIES


def test_noise_columns_do_not_change_prices():
    plain = make_rows(50, seed=5)
    noisy = make_rows(50, seed=5, n_noise=10)
    assert [r[-1] for r in plain] == [r[-1] for r in noisy]
    assert len(noisy[0]) == len(plain[0]) + 10


def test_true_coefficients_use_hatch_as_reference():
    coefs = true_coefficients()
    assert coefs == {"intercept": 2600.0, "age": -150.0, "km": -0.02, "body=suv": 1200.0, "body=sedan": 400.0}


def test_nonlinear_term_only_above_150k():
    assert true_price(5, 100_000, "sedan", nonlinear=True) == true_price(5, 100_000, "sedan")
    assert true_price(5, 200_000, "sedan", nonlinear=True) == pytest.approx(true_price(5, 200_000, "sedan") - 2000)


def test_bad_arguments_explain_themselves():
    with pytest.raises(ValueError, match="rows must be at least 1"):
        make_rows(0, seed=1)
    with pytest.raises(ValueError, match="n_noise"):
        make_rows(5, seed=1, n_noise=99)


def test_spark_frame_has_expected_shape(cars):
    assert cars.count() == 1000
    assert cars.columns == ["id", "age", "km", "body", "price"]
