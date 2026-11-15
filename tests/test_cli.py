import json

import pytest

from price_regression import cli


@pytest.fixture()
def shared_spark(monkeypatch, spark):
    """Let main() use the shared session and not stop it at the end."""
    monkeypatch.setattr(cli, "get_spark", lambda: spark)
    monkeypatch.setattr(cli, "stop_spark", lambda: None)


def test_help_is_localized(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help", "--lang", "ko"])
    assert exc.value.code == 0
    assert "중고차" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["--help", "--lang", "en"])
    assert "used-car" in capsys.readouterr().out


def test_bad_rows_and_missing_predict_args(capsys):
    assert cli.main(["evaluate", "--rows", "5", "--lang", "en"]) == 3
    assert "--rows must be at least 50" in capsys.readouterr().err
    assert cli.main(["predict", "--age", "3", "--lang", "ko"]) == 3
    assert "--age" in capsys.readouterr().err


@pytest.mark.slow
def test_all_writes_run_json(shared_spark, tmp_path, capsys):
    assert cli.main(["all", "--lang", "ko", "--rows", "1000", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "RMSE" in out and "R²" in out and "그래서 무슨 뜻?" in out
    run = json.loads((tmp_path / "run.json").read_text())
    for key in ("test", "baseline", "improvement", "train_rows", "test_rows"):
        assert key in run
    assert {"rmse", "mae", "r2"} <= set(run["test"])
    assert run["test"]["r2"] >= 0.98
    assert run["baseline"]["rmse"] >= 5 * run["test"]["rmse"]
    assert (tmp_path / "model").exists()


def test_predict_command_trains_then_reuses_model(shared_spark, tmp_path, capsys):
    args = ["predict", "--age", "3", "--km", "40000", "--body", "suv", "--lang", "en",
            "--out", str(tmp_path), "--rows", "1000"]
    assert cli.main(args) == 0
    assert "training one first" in capsys.readouterr().out  # no model saved yet
    assert cli.main(args) == 0
    second = capsys.readouterr().out
    assert "Predicted price:" in second and "training one first" not in second
    unknown = ["predict", "--age", "3", "--km", "1", "--body", "tractor", "--out", str(tmp_path)]
    assert cli.main(unknown) == 3
