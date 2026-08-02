"""Deterministic seed derivation.

A run has one master seed. Every sample, and every independently randomised aspect
of a sample, derives its own seed from that master seed by hashing. This is what
makes a single sample reproducible without replaying the ones before it -- important
because a 50k-image run will produce a handful of odd frames that you want to
re-render and inspect individually.

``hash()`` is deliberately not used: Python randomises string hashing per process,
so it is not stable across runs.
"""

from __future__ import annotations

import random
from hashlib import blake2b

#: Seeds are truncated to 63 bits so they stay positive and JSON-safe.
_SEED_BITS = 63
_SEED_MASK = (1 << _SEED_BITS) - 1


def derive_seed(master_seed: int, *parts: object) -> int:
    """Derive a stable child seed from ``master_seed`` and arbitrary key parts.

    >>> derive_seed(42, "sample", 7) == derive_seed(42, "sample", 7)
    True
    """
    digest = blake2b(digest_size=8)
    digest.update(str(int(master_seed)).encode("utf-8"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "big") & _SEED_MASK


def derive_rng(master_seed: int, *parts: object) -> random.Random:
    """Return a :class:`random.Random` seeded by :func:`derive_seed`."""
    return random.Random(derive_seed(master_seed, *parts))
