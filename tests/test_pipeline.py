import pytest

from price_regression.pipeline import body_labels, feature_names, named_coefficients, split_train_test


def test_recovers_true_formula(model):
    """AC-2: the learned coefficients match the hidden formula (reference category: hatch)."""
    assert body_labels(model) == ["suv", "sedan", "hatch"]  # hatch is last -> dropped -> reference
    assert feature_names(model) == ["age", "km", "body=suv", "body=sedan"]
    c = named_coefficients(model)
    assert c["age"] == pytest.approx(-150, abs=5)
    assert c["km"] == pytest.approx(-0.02, abs=0.002)
    assert c["body=sedan"] == pytest.approx(400, abs=60)
    assert c["body=suv"] == pytest.approx(1200, abs=60)
    assert c["intercept"] == pytest.approx(2600, abs=80)


def test_coefficients_are_plain_floats(model):
    assert all(type(v) is float for v in named_coefficients(model).values())


def test_split_is_reproducible_and_disjoint(cars, split):
    """AC-3: same seed -> same rows; train and test never share a row; ratio about 80/20."""
    train, test = split
    again_train, _ = split_train_test(cars, seed=42)
    train_ids = {r["id"] for r in train.select("id").collect()}
    test_ids = {r["id"] for r in test.select("id").collect()}
    assert train_ids == {r["id"] for r in again_train.select("id").collect()}
    assert not train_ids & test_ids
    assert len(train_ids) + len(test_ids) == 1000
    assert len(train_ids) / 1000 == pytest.approx(0.8, abs=0.05)
    _, other_test = split_train_test(cars, seed=7)
    assert test_ids != {r["id"] for r in other_test.select("id").collect()}
