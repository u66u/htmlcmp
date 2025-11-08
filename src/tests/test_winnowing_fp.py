import pytest
from fingerprint.winnowing_fp import (
    WinnowConfig,
    kgram_hashes,
    winnow,
    winnow_from_hashes,
)


def test_empty_and_short_inputs():
    cfg = WinnowConfig(k_words=5, window_size=4, use_xxhash=False)
    assert kgram_hashes([], cfg) == []
    assert kgram_hashes(["a", "b", "c"], cfg) == []
    fp_set, picks = winnow([], cfg)
    assert fp_set == set()
    assert picks == []


def test_determinism_and_basic_props():
    tokens = "this is a simple test document with repeated words test document".split()
    cfg = WinnowConfig(k_words=3, window_size=4, use_xxhash=False, seed=123)
    fp1, picks1 = winnow(tokens, cfg)
    fp2, picks2 = winnow(tokens, cfg)
    assert fp1 == fp2
    assert picks1 == picks2

    hashes = kgram_hashes(tokens, cfg)
    assert fp1.issubset(set(hashes))


def test_window_size_controls_fingerprint_count():
    tokens = ("alpha beta gamma delta epsilon " * 20).strip().split()
    cfg_small = WinnowConfig(k_words=5, window_size=4, use_xxhash=False)
    cfg_large = WinnowConfig(k_words=5, window_size=12, use_xxhash=False)
    fp_small, _ = winnow(tokens, cfg_small)
    fp_large, _ = winnow(tokens, cfg_large)

    assert len(fp_large) <= len(fp_small)


def test_small_edit_keeps_high_overlap():
    base = ("w" + " x " * 4) * 80
    base_tokens = base.split()
    edited_tokens = base_tokens[:200] + ["zzz"] + base_tokens[200 + 3 :]
    cfg = WinnowConfig(k_words=5, window_size=10, use_xxhash=False, seed=7)
    fp_a, _ = winnow(base_tokens, cfg)
    fp_b, _ = winnow(edited_tokens, cfg)
    inter = len(fp_a & fp_b)
    jacc = inter / (len(fp_a | fp_b)) if fp_a or fp_b else 1.0
    assert jacc > 0.6


def test_rightmost_tie_break_behavior():
    hashes = [5, 4, 4, 7, 2, 2, 9]
    fp, picks = winnow_from_hashes(hashes, window_size=3)
    expected = [(4, 2), (2, 4), (2, 5)]
    filtered = [(v, i) for (v, i) in picks if v in {4, 2}]
    assert expected == filtered[: len(expected)]
