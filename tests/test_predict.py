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


def test_option_mismatch_note(spark, model, tmp_path, capsys):
    """Differing / same / missing-metadata cases for the saved-model option note."""
    from price_regression.predict import write_meta
    save_model(model, tmp_path)
    write_meta(tmp_path, seed=42, rows=5000, nonlinear=False)
    for lang, word in (("en", "Using the saved model (seed 42, 5000 rows, linear data)"),
                       ("ko", "저장된 모델(시드 42, 5000행, 선형 데이터)")):
        run = Run(opts=Options(lang=lang, out=str(tmp_path), seed=7, nonlinear=True), spark=spark)
        assert step_predict(run, 3, 40_000, "suv") == EXIT_OK
        out = capsys.readouterr().out
        assert word in out and str(tmp_path / "model") in out
    same = Run(opts=Options(lang="en", out=str(tmp_path)), spark=spark)
    step_predict(same, 3, 40_000, "suv")
    assert "saved model" not in capsys.readouterr().out
    (tmp_path / "model_meta.json").unlink()
    old = Run(opts=Options(lang="en", out=str(tmp_path), seed=7), spark=spark)
    assert step_predict(old, 3, 40_000, "suv") == EXIT_OK
    assert "saved model" not in capsys.readouterr().out
