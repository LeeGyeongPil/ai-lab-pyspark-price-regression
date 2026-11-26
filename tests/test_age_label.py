import pytest

from price_regression.steps import age_label


@pytest.mark.parametrize(
    ("age", "text"),
    [(3.5, "3.5"), (4, "4"), (4.0, "4"), (2.5, "2.5"), (3.0, "3")],
)
def test_age_label_keeps_fraction_without_rounding(age: float, text: str) -> None:
    assert age_label(age) == text
