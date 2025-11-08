import pytest
from collections import Counter

import fingerprint.simhash_fp as sh


def test_determinism_same_tokens_same_signature():
    cfg = sh.SimHashConfig(n_bits=64, hash_name="blake2b", seed=123)
    tokens = "this is a test this is only a test".split()
    sig1 = sh.compute_simhash(tokens, cfg)
    sig2 = sh.compute_simhash(tokens, cfg)
    assert isinstance(sig1, int)
    assert sig1 == sig2


def test_order_invariance_for_unigrams():
    cfg = sh.SimHashConfig(n_bits=64, hash_name="blake2b", include_bigrams=False, seed=0)
    tokens = "alpha beta gamma delta epsilon".split()
    tokens_perm = list(reversed(tokens))
    sig1 = sh.compute_simhash(tokens, cfg)
    sig2 = sh.compute_simhash(tokens_perm, cfg)
    assert sig1 == sig2


def test_bigrams_make_order_matter():
    cfg = sh.SimHashConfig(n_bits=64, hash_name="blake2b", include_bigrams=True, seed=0)
    t1 = "alpha beta gamma delta epsilon".split()
    t2 = list(reversed(t1))
    sig1 = sh.compute_simhash(t1, cfg)
    sig2 = sh.compute_simhash(t2, cfg)
    assert sig1 != sig2


def test_similarity_distance_smaller_than_dissimilarity():
    cfg = sh.SimHashConfig(n_bits=64, hash_name="blake2b", seed=42)
    base = ("lorem ipsum dolor sit amet " * 20).split()
    similar = base + "consectetur adipiscing elit".split()
    different = ("quantum flux capacitor neutrino brane axiom " * 20).split()

    sig_base = sh.compute_simhash(base, cfg)
    sig_sim = sh.compute_simhash(similar, cfg)
    sig_diff = sh.compute_simhash(different, cfg)

    d_sim = sh.hamming_distance(sig_base, sig_sim)
    d_diff = sh.hamming_distance(sig_base, sig_diff)
    assert d_sim < d_diff


def test_tfidf_rare_token_gets_higher_weight_than_common():
    tf_counts = Counter({"rare": 3, "common": 3})
    tfw = sh.tf_transform(tf_counts, mode="raw")
    N = 100
    df_map = {"rare": 1, "common": 100}
    weights = sh.apply_tfidf(tfw, df_map, doc_count=N, smooth=True)
    assert weights["rare"] > weights["common"]


def test_compute_simhash_requires_df_when_tfidf_enabled():
    cfg = sh.SimHashConfig(use_tfidf=True, hash_name="blake2b")
    with pytest.raises(ValueError):
        _ = sh.compute_simhash(["a", "b"], cfg, df_map=None, doc_count=0)


def test_seed_changes_signature():
    tokens = "a b c d e f g".split()
    sig1 = sh.compute_simhash(tokens, sh.SimHashConfig(hash_name="blake2b", seed=1))
    sig2 = sh.compute_simhash(tokens, sh.SimHashConfig(hash_name="blake2b", seed=2))
    assert sig1 != sig2


def test_hamming_distance_basic():
    assert sh.hamming_distance(0b1010, 0b0011) == 2
    a = 0xDEADBEEF
    assert sh.hamming_distance(a, a) == 0
    
    b = 0x12345678
    assert sh.hamming_distance(a, b) == sh.hamming_distance(b, a)


def test_empty_tokens_returns_zero_signature():
    cfg = sh.SimHashConfig(hash_name="blake2b")
    assert sh.compute_simhash([], cfg) == 0


def test_tie_break_sets_bit_to_one(monkeypatch):
    def fake_hash(data: bytes, seed: int = 0, hash_name: str = "blake2b") -> int:
        s = data.decode("utf-8")
        if s == "ones":
            return (1 << 64) - 1  
        if s == "zeros":
            return 0  
        return 0

    monkeypatch.setattr(sh, "_stable_hash_64", fake_hash)
    weights = {"ones": 1.0, "zeros": 1.0}
    sig = sh.simhash_from_weights(weights, n_bits=16, hash_name="blake2b", seed=0)
    assert sig == (1 << 16) - 1  


def test_max_features_cropping_matches_manual_topk():
    tokens = ["a"] * 10 + ["b"] * 5 + ["c"] * 3 + ["d"] * 2
    
    feats = sh.build_features(tokens, include_bigrams=False)
    tfw = sh.tf_transform(feats, mode="log")
    top1 = dict(sorted(tfw.items(), key=lambda kv: kv[1], reverse=True)[:1])
    manual_sig = sh.simhash_from_weights(top1, n_bits=64, hash_name="blake2b", seed=0)

    cfg_top1 = sh.SimHashConfig(n_bits=64, hash_name="blake2b", max_features=1, seed=0)
    cfg_all = sh.SimHashConfig(n_bits=64, hash_name="blake2b", max_features=None, seed=0)

    sig_top1 = sh.compute_simhash(tokens, cfg_top1)
    sig_all = sh.compute_simhash(tokens, cfg_all)

    assert sig_top1 == manual_sig
    
    assert sig_top1 != sig_all