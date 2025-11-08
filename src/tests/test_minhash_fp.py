import random
import pytest
from fingerprint.minhash_fp import (
    MinHashConfig,
    hashed_k_gram_set,
    minhash_signature,
    estimate_jaccard,
    jaccard_of_sets,
)


def test_empty_and_short_inputs_signature():
    cfg = MinHashConfig(k_words=5, num_perm=64, use_xxhash=False)
    sig = minhash_signature(set(), cfg)
    assert len(sig) == cfg.num_perm
    assert all(x == 2**64 - 1 for x in sig)

    tokens = ["a", "b", "c"]
    sh = hashed_k_gram_set(tokens, cfg)
    assert sh == set()


def test_identical_docs_identical_signatures():
    tokens = "this is a minhash test this is only a test".split()
    cfg = MinHashConfig(k_words=3, num_perm=64, use_xxhash=False, seed=42)
    sh1 = hashed_k_gram_set(tokens, cfg)
    sh2 = hashed_k_gram_set(tokens[:], cfg)
    sig1 = minhash_signature(sh1, cfg)
    sig2 = minhash_signature(sh2, cfg)
    assert sig1 == sig2
    assert estimate_jaccard(sig1, sig2) == 1.0
    assert jaccard_of_sets(sh1, sh2) == 1.0


def test_jaccard_estimate_tracks_true_jaccard():
    base_tokens = [f"w{i}" for i in range(100)]
    edited_tokens = base_tokens[:]
    for idx in [20, 50, 70]:
        edited_tokens[idx] = f"x{idx}"

    cfg = MinHashConfig(k_words=3, num_perm=128, use_xxhash=False, seed=7)
    A = hashed_k_gram_set(base_tokens, cfg)
    B = hashed_k_gram_set(edited_tokens, cfg)
    sigA = minhash_signature(A, cfg)
    sigB = minhash_signature(B, cfg)

    true_j = jaccard_of_sets(A, B)
    est_j = estimate_jaccard(sigA, sigB)
    assert abs(est_j - true_j) < 0.15


def test_far_sets_low_estimated_jaccard():
    tokens_a = [f"a{i}" for i in range(200)]
    tokens_b = [f"b{i}" for i in range(200)]
    cfg = MinHashConfig(k_words=3, num_perm=128, use_xxhash=False, seed=1)
    A = hashed_k_gram_set(tokens_a, cfg)
    B = hashed_k_gram_set(tokens_b, cfg)
    sigA = minhash_signature(A, cfg)
    sigB = minhash_signature(B, cfg)
    est_j = estimate_jaccard(sigA, sigB)
    assert est_j < 0.1
