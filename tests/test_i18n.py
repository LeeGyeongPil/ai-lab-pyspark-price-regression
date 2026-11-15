import string

import pytest

from price_regression.i18n import EN, KO, detect_lang, t


def placeholders(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_keys_match_and_language_precedence():
    """AC-8: both languages have the same keys; --lang > PRICE_REGRESSION_LANG > LC_ALL/LC_MESSAGES/LANG."""
    assert set(EN) == set(KO)
    for key in EN:
        assert placeholders(EN[key]) == placeholders(KO[key]), key  # same {placeholders} in both

    env = {"PRICE_REGRESSION_LANG": "ko", "LANG": "en_US.UTF-8"}
    assert detect_lang("en", env) == "en"  # option wins
    assert detect_lang(None, env) == "ko"  # then the tool variable
    assert detect_lang(None, {"LC_ALL": "ko_KR.UTF-8", "LANG": "en_US.UTF-8"}) == "ko"
    assert detect_lang(None, {"LC_MESSAGES": "ko_KR.UTF-8", "LANG": "en_US.UTF-8"}) == "ko"
    assert detect_lang(None, {"LANG": "ko_KR.UTF-8"}) == "ko"
    assert detect_lang(None, {"LANG": "fr_FR.UTF-8"}) == "en"  # not Korean -> English
    assert detect_lang(None, {}) == "en"
    assert detect_lang(None, {"PRICE_REGRESSION_LANG": "xx", "LANG": "ko_KR"}) == "ko"  # bad value ignored


def test_switching_language_changes_text():
    assert t("en", "doctor_all_good") != t("ko", "doctor_all_good")
    assert "만 원" in t("ko", "unit")
    with pytest.raises(KeyError):
        t("en", "no_such_key")
