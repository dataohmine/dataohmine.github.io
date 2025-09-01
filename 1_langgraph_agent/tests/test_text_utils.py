import pytest
from utils.text_utils import normalize_text, extract_first_last


def test_normalize_text_replaces_quotes_and_dashes_and_normalizes():
    raw = "Cafe\u0301 — “smart” quotes"
    expected = 'Café - "smart" quotes'
    assert normalize_text(raw) == expected


def test_extract_first_last_from_phrase():
    text = "hello, my name is jane smith"
    assert extract_first_last(text) == ("Jane", "Smith")


def test_extract_first_last_from_capitals():
    text = "John Doe went to the store"
    assert extract_first_last(text) == ("John", "Doe")


def test_extract_first_last_fallback_words():
    text = "john doe"
    assert extract_first_last(text) == ("John", "Doe")


def test_extract_first_last_single_word():
    text = "Madonna"
    assert extract_first_last(text) == ("Madonna", "Unknown")

