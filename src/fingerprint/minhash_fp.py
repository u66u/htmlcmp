"""
MinHash on k-grams
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple
from src.preprocess import k_gram
try:
    import xxhash
    _HAS_XXHASH = True
except Exception:
    _HAS_XXHASH = False
import hashlib


@dataclass
class MinHashConfig:
    k_words: int = 5
    num_perm: int = 128
    use_xxhash: bool = True
    seed: int = 0

    def validate(self):
        assert self.k_words >= 1
        assert self.num_perm >= 1


def _stable_hash_64(data: bytes, seed: int = 0, use_xxhash: bool = True) -> int:
    if use_xxhash and _HAS_XXHASH:
        return xxhash.xxh3_64_intdigest(data, seed=seed & 0xFFFFFFFFFFFFFFFF)
    h = hashlib.blake2b(data, digest_size=8, person=b"minhash")
    return int.from_bytes(h.digest(), "big", signed=False)


def hashed_k_gram_set(tokens: List[str], cfg: MinHashConfig) -> Set[int]:
    s_list = k_gram(tokens, cfg.k_words)
    return {_stable_hash_64(s.encode("utf-8"), seed=cfg.seed, use_xxhash=cfg.use_xxhash) for s in s_list}


def minhash_signature(k_gram_set: Set[int], cfg: MinHashConfig) -> Tuple[int, ...]:
    cfg.validate()
    if not k_gram_set:
        return tuple([2**64 - 1] * cfg.num_perm)

    sig = [2**64 - 1] * cfg.num_perm
    for i in range(cfg.num_perm):
        seed_i = (cfg.seed + i) & 0xFFFFFFFFFFFFFFFF
        min_val = 2**64 - 1
        for s in k_gram_set:
            hv = _stable_hash_64(seed_i.to_bytes(8, "big", signed=False) + s.to_bytes(8, "big", signed=False),
                                 seed=0, use_xxhash=cfg.use_xxhash)
            if hv < min_val:
                min_val = hv
        sig[i] = min_val
    return tuple(sig)


def estimate_jaccard(sig_a: Tuple[int, ...], sig_b: Tuple[int, ...]) -> float:
    """Estimate Jaccard similarity as the fraction of equal components."""
    assert len(sig_a) == len(sig_b)
    if not sig_a:
        return 1.0
    eq = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return eq / len(sig_a)


def jaccard_of_sets(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)
