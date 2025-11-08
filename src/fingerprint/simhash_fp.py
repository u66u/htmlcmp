from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import Counter
from heapq import nlargest
import math
try:
    import xxhash
    _HAS_XXHASH = True
except Exception:
    _HAS_XXHASH = False

try:
    from blake3 import blake3 
    _HAS_BLAKE3 = True
except Exception:
    _HAS_BLAKE3 = False
import hashlib


@dataclass
class SimHashConfig:
    n_bits: int = 64
    include_bigrams: bool = False
    max_features: Optional[int] = 3000  # cap features by weight to stabilize/accelerate
    tf_mode: str = "log"                # "raw" or "log"
    use_tfidf: bool = False             # if True, weights = TF * IDF
    hash_name: str = "xxhash64"         # "xxhash64" | "blake3" | "blake2b"
    seed: int = 0                       # seed for hashing (permutes hyperplanes deterministically)


def _stable_hash_64(data: bytes, seed: int = 0, hash_name: str = "xxhash64") -> int:
    if hash_name == "xxhash64" and _HAS_XXHASH:
        return xxhash.xxh3_64_intdigest(data, seed=seed & 0xFFFFFFFFFFFFFFFF)
    elif hash_name == "blake3" and _HAS_BLAKE3:
        s = seed.to_bytes(8, "big", signed=False)
        return int.from_bytes(blake3(s + data).digest()[:8], "big", signed=False)
    else:
        s = seed.to_bytes(8, "big", signed=False)
        h = hashlib.blake2b(s + data, digest_size=8, person=b"simhash")
        return int.from_bytes(h.digest(), "big", signed=False)


def build_features(tokens: List[str], include_bigrams: bool = False) -> Counter:
    counts = Counter(tokens)
    if include_bigrams and len(tokens) >= 2:
        bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        counts.update(bigrams)
    return counts


def tf_transform(tf_counts: Counter, mode: str = "log") -> Dict[str, float]:
    """
    Transform raw term counts to TF weights.
    - "raw": w = tf
    - "log": w = 1 + log(tf)
    """
    if mode == "raw":
        return {t: float(c) for t, c in tf_counts.items()}
    return {t: 1.0 + math.log(c) for t, c in tf_counts.items() if c > 0}


def apply_tfidf(tf_weights: Dict[str, float],
                df_map: Dict[str, int],
                doc_count: int,
                smooth: bool = True) -> Dict[str, float]:
    """
    Apply IDF to TF weights. Caller supplies df_map[token] and total doc_count.
    IDF = log((N + 1)/(df + 1)) + 1 if smooth else log(N/df).
    """
    out: Dict[str, float] = {}
    N = max(doc_count, 1)
    for t, tfw in tf_weights.items():
        df = df_map.get(t, 0)
        if smooth:
            idf = math.log((N + 1) / (df + 1)) + 1.0
        else:
            idf = math.log(N / max(df, 1))
        out[t] = tfw * idf
    return out


def top_k_by_weight(weights: Dict[str, float], k: Optional[int]) -> Dict[str, float]:
    if k is None or k <= 0 or k >= len(weights):
        return weights
    topk = dict(nlargest(k, weights.items(), key=lambda kv: kv[1]))
    # topk = dict(sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:k])
    return topk


def simhash_from_weights(weights: Dict[str, float],
                         n_bits: int = 64,
                         hash_name: str = "xxhash64",
                         seed: int = 0) -> int:
    if not weights:
        return 0
    accum = [0.0] * n_bits
    mask64 = (1 << 64) - 1

    for feat, w in weights.items():
        if w == 0.0:
            continue
        h = _stable_hash_64(feat.encode("utf-8"), seed=seed, hash_name=hash_name) & mask64
        for b in range(n_bits):
            if (h >> b) & 1:
                accum[b] += w
            else:
                accum[b] -= w
    sig = 0
    for b in range(n_bits):
        if accum[b] >= 0.0: 
            sig |= (1 << b)
    return sig


def compute_simhash(tokens: List[str],
                    config: SimHashConfig,
                    df_map: Optional[Dict[str, int]] = None,
                    doc_count: int = 0) -> int:
    """
    - Build features (unigrams + optional bigrams).
    - TF transform (raw/log).
    - Keep top-K features (by final weight) if configured.
    - Compute SimHash.
    """
    feats = build_features(tokens, include_bigrams=config.include_bigrams)
    tfw = tf_transform(feats, mode=config.tf_mode)

    if config.use_tfidf:
        if df_map is None:
            raise ValueError("use_tfidf=True but df_map is None. Supply df_map[token] and doc_count.")
        weights = apply_tfidf(tfw, df_map=df_map, doc_count=doc_count, smooth=True)
    else:
        weights = tfw

    weights = top_k_by_weight(weights, config.max_features)
    return simhash_from_weights(weights,
                                n_bits=config.n_bits,
                                hash_name=config.hash_name,
                                seed=config.seed)


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()