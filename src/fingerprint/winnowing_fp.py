"""
Winnowing over k-grams using Rabin–Karp rolling hash.
Might be sensitive to parameters, sensible defaults are k=5 words, window w=10,
this should give ~200–600 fingerprints per document.
Avoid window=1

TODO: dynamically infer k and window size from document length
TODO: test tiny-prime modulus hashing: https://osf.io/preprints/osf/yxjnp_v1
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Set
from collections import deque
try:
    import xxhash 
    _HAS_XXHASH = True
except Exception:
    _HAS_XXHASH = False
import hashlib

@dataclass
class WinnowConfig:
    k_words: int = 5          # k_gram size (in tokens)
    window_size: int = 10     # winnowing window size (in k-grams)
    base: int = 1_000_003     # rolling hash base
    use_xxhash: bool = True   # whether to prefer xxhash for token->int mapping
    seed: int = 0             # seed for token hashing

    def validate(self):
        assert self.k_words >= 1
        assert self.window_size >= 1


def _stable_hash_64(data: bytes, seed: int = 0, use_xxhash: bool = True) -> int:
    if use_xxhash and _HAS_XXHASH:
        return xxhash.xxh3_64_intdigest(data, seed=seed & 0xFFFFFFFFFFFFFFFF)
    h = hashlib.blake2b(data, digest_size=8, person=b"winnow")
    return int.from_bytes(h.digest(), "big", signed=False)


def _token_ints(tokens: List[str], seed: int, use_xxhash: bool) -> List[int]:
    return [_stable_hash_64(t.encode("utf-8"), seed=seed, use_xxhash=use_xxhash) & ((1 << 32) - 1)
            for t in tokens]


def kgram_hashes(tokens: List[str], cfg: WinnowConfig) -> List[int]:
    cfg.validate()
    n = len(tokens)
    k = cfg.k_words

    if n < k: return []
    mask = (1 << 64) - 1
    base = cfg.base & mask

    vals = _token_ints(tokens, seed=cfg.seed, use_xxhash=cfg.use_xxhash)
    pow_km1 = pow(base, k - 1, 1 << 64)

    h = 0
    for j in range(k):
        h = ((h * base) + vals[j]) & mask
    hashes = [h]

    for i in range(k, n):
        out_val = (vals[i - k] * pow_km1) & mask
        h = (h - out_val) & mask
        h = ((h * base) + vals[i]) & mask
        hashes.append(h)

    return hashes


def winnow_from_hashes(hashes: List[int], window_size: int) -> Tuple[Set[int], List[Tuple[int, int]]]:
    n = len(hashes)
    if n == 0: return set(), []
    w = max(1, min(window_size, n))

    # worst case: w = 1 → select every element (no winnowing)
    if w == 1:
        picks_seq = [(hashes[i], i) for i in range(n)]
        picks_set = set(hashes)
        return picks_set, picks_seq

    dq = deque()  
    picks_set: Set[int] = set()
    picks_seq: List[Tuple[int, int]] = []

    for i, hv in enumerate(hashes):
        while dq and hashes[dq[-1]] >= hv:
            dq.pop()
        dq.append(i)

        while dq and dq[0] <= i - w:
            dq.popleft()

        if i >= w - 1:
            min_idx = dq[0]
            if not picks_seq or picks_seq[-1][1] != min_idx:
                min_val = hashes[min_idx]
                picks_seq.append((min_val, min_idx))
                picks_set.add(min_val)

    return picks_set, picks_seq


def winnow(tokens: List[str], cfg: WinnowConfig) -> Tuple[Set[int], List[Tuple[int, int]]]:
    """
    Full pipeline: compute k-gram hashes and select fingerprints.
    """
    h = kgram_hashes(tokens, cfg)
    return winnow_from_hashes(h, cfg.window_size)