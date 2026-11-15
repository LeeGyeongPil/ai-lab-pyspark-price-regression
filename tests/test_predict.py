import pytest

from price_regression.predict import UnknownBodyError, load_model, predict_price, save_model
from price_regression.steps import EXIT_INPUT, EXIT_OK, Options, Run, step_predict


def test_save_load_predict_and_guards(spark, model, tmp_path, capsys):
    """AC-7: a saved model predicts the same after loading; bad inputs are handled kindly."""
    assert load_model(tmp_path) is None  # nothing saved yet
    save_model(model, tmp_path)
    loaded = load_model(tmp_path)
    assert loaded is not None

    suv = predict_price(spark, loaded, age=3, km=40_000, body="suv")
    assert suv.price == pytest.approx(2550, abs=60)
    assert suv.warnings == []
    assert predict_price(spark, model, 3, 40_000, "suv").price == pytest.approx(suv.price, abs=1e-6)

    with pytest.raises(UnknownBodyError) as info:
        predict_price(spark, loaded, 3, 40_000, "tractor")
    assert info.value.known == ["suv", "sedan", "hatch"]

    far = predict_price(spark, loaded, age=30, km=400_000, body="hatch")
    assert far.price < 0
    assert set(far.warnings) == {"warn_age_range", "warn_km_range", "warn_negative_price"}

    run = Run(opts=Options(lang="en", out=str(tmp_path)), spark=spark)
    assert step_predict(run, 3, 40_000, "tractor") == EXIT_INPUT
    assert "Unknown body type 'tractor'" in capsys.readouterr().out
    assert step_predict(run, 30, 400_000, "hatch") == EXIT_OK
    assert "negative" in capsys.readouterr().out
