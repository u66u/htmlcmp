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


examples = [
    ("Plagiarism detection techniques typically use document fingerprinting methods", 5, 10),
    ("The quick brown fox jumps over the lazy dog multiple times", 4, 5),
    ("A rolling hash significantly speeds up substring search in large texts", 6, 12),
    ("Winnowing algorithm selects minimal hashes to produce document fingerprints", 5, 8),
    ("Document similarity detection can benefit from combined k-gram hashing approaches", 7, 10),
    ("Text preprocessing including tokenization stop-word removal affects fingerprinting", 5, 9),
    ("Hash collisions are rare but possible; they can introduce false positives in matches", 4, 7),
    ("Applications of Winnowing span plagiarism detection text deduplication and bioinformatics", 6, 11),
    ("Plagiarism detection systems require efficient and robust document comparison algorithms", 5, 10),
    ("Efficient local fingerprinting like Winnowing balances accuracy and computational cost", 5, 8),
]


@pytest.mark.parametrize("text,k,w", examples)
def test_winnowing_general(text, k, w):
    def tokenize(text):
        return text.lower().split()

    tokens = tokenize(text)
    cfg = WinnowConfig(k_words=k, window_size=w)
    fingerprints, seq = winnow(tokens, cfg)
    assert len(fingerprints) <= max(len(tokens) - k + 1, 0)

    for _, pos in seq:
        assert 0 <= pos <= len(tokens) - k

    assert len(fingerprints) == len(set(fp for fp, _ in seq))





