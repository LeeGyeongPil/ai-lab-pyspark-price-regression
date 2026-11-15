import pytest

from price_regression import cli, doctor


@pytest.mark.parametrize("text, major", [
    ('openjdk version "21.0.10" 2026-01-20', 21),
    ('java version "1.8.0_392"', 8),
    ('openjdk version "17" 2021-09-14', 17),
    ("not java", None),
])
def test_parse_java_major(text, major):
    assert doctor.parse_java_major(text) == major


def test_missing_java_exits_2_with_guide(monkeypatch, capsys):
    """AC-9: no Java 17+ -> install guide in the chosen language, exit 2, Spark never started."""
    monkeypatch.setattr(doctor, "find_java", lambda: None)

    def must_not_start(*args, **kwargs):
        raise AssertionError("Spark must not start without Java")

    monkeypatch.setattr(cli, "get_spark", must_not_start)
    for step in ("doctor", "evaluate", "all"):
        assert cli.main([step, "--lang", "en"]) == 2
        assert "Java 17 or newer was not found" in capsys.readouterr().out
    assert cli.main(["doctor", "--lang", "ko"]) == 2
    assert "Java 17 이상을 찾지 못했습니다" in capsys.readouterr().out


def test_old_java_is_skipped(monkeypatch, tmp_path):
    fake = tmp_path / "java"
    fake.write_text("")
    monkeypatch.setattr(doctor, "_candidates", lambda: [(str(fake), False)])
    monkeypatch.setattr(doctor, "_probe", lambda exe: (11, "11.0.2"))
    assert doctor.find_java() is None
    monkeypatch.setattr(doctor, "_probe", lambda exe: (21, "21.0.1"))
    assert doctor.find_java().major == 21


def test_doctor_ok_reports_versions(capsys):
    assert cli.main(["doctor", "--lang", "en"]) == 0
    out = capsys.readouterr().out
    assert "PySpark 4.3" in out and "Java" in out and "Temp folder is writable" in out


def test_tmp_writable_false_for_file_path(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert doctor.tmp_writable(blocker / "sub") is False
