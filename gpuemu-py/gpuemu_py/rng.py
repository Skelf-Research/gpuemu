"""Seeded RNG for reproducible fuzzing across Rust and Python.

This module provides a deterministic RNG that can derive sub-RNGs
for different domains (shapes, dtypes, layouts, data) from a master seed.
The derivation uses Blake2b to ensure cross-language compatibility with Rust.
"""

import hashlib
from typing import List, TypeVar, Sequence

import numpy as np

T = TypeVar("T")


def derive_seed(seed: int, domain: str) -> int:
    """Derive a sub-seed from a master seed and domain string.

    Uses Blake2b-64 to hash the seed (little-endian bytes) concatenated
    with the domain string (UTF-8 bytes). This algorithm is implemented
    identically in Rust for cross-language reproducibility.

    Args:
        seed: Master seed (u64).
        domain: Domain string for derivation.

    Returns:
        Derived seed (u64).
    """
    # Ensure seed is within u64 range
    seed = seed & 0xFFFFFFFFFFFFFFFF

    # Hash seed (little-endian) + domain using Blake2b with 8-byte digest
    h = hashlib.blake2b(digest_size=8)
    h.update(seed.to_bytes(8, "little"))
    h.update(domain.encode("utf-8"))
    return int.from_bytes(h.digest(), "little")


def generate_seed() -> int:
    """Generate a timestamp-based seed (for when no seed is specified).

    Returns:
        A seed based on current time in nanoseconds.
    """
    import time

    return int(time.time_ns()) & 0xFFFFFFFFFFFFFFFF


class SeededRng:
    """Deterministic RNG wrapper for reproducible fuzzing.

    Uses numpy's PCG64 as the underlying RNG.
    Seed derivation uses Blake2b-64 for cross-language compatibility.

    Example:
        >>> master = SeededRng(12345)
        >>> shapes_rng = master.derive("shapes")
        >>> data_rng = master.derive("data")
        >>> # shapes_rng and data_rng are independent but deterministic
    """

    def __init__(self, seed: int):
        """Create a new RNG from a seed.

        Args:
            seed: The seed value (u64).
        """
        self._seed = seed & 0xFFFFFFFFFFFFFFFF
        self._rng = np.random.Generator(np.random.PCG64(self._seed))

    @property
    def seed(self) -> int:
        """Get the seed used to create this RNG."""
        return self._seed

    def derive(self, domain: str) -> "SeededRng":
        """Derive a sub-RNG for a specific domain.

        This uses Blake2b to hash the seed with the domain string,
        producing a deterministic sub-seed. The algorithm matches
        the Rust implementation for cross-language reproducibility.

        Args:
            domain: Domain string (e.g., "shapes", "dtypes", "data").

        Returns:
            A new SeededRng for the derived domain.
        """
        sub_seed = derive_seed(self._seed, domain)
        return SeededRng(sub_seed)

    def choice(self, options: Sequence[T]) -> T:
        """Choose a random element from a sequence.

        Args:
            options: Non-empty sequence to choose from.

        Returns:
            A randomly chosen element.

        Raises:
            ValueError: If options is empty.
        """
        if len(options) == 0:
            raise ValueError("Cannot choose from empty sequence")
        idx = int(self._rng.integers(0, len(options)))
        return options[idx]

    def choice_index(self, length: int) -> int:
        """Choose a random index.

        Args:
            length: The range [0, length) to choose from.

        Returns:
            A randomly chosen index.
        """
        if length <= 0:
            raise ValueError("Cannot choose from empty range")
        return int(self._rng.integers(0, length))

    def gen_range(self, low: int, high: int) -> int:
        """Generate a random integer in [low, high).

        Args:
            low: Lower bound (inclusive).
            high: Upper bound (exclusive).

        Returns:
            A random integer.
        """
        return int(self._rng.integers(low, high))

    def gen_u64(self) -> int:
        """Generate a random u64.

        Returns:
            A random 64-bit unsigned integer.
        """
        return int(self._rng.integers(0, 2**64, dtype=np.uint64))

    def gen_f32(self) -> float:
        """Generate a random float in [0, 1).

        Returns:
            A random float.
        """
        return float(self._rng.random())

    def gen_f64(self) -> float:
        """Generate a random float in [0, 1).

        Returns:
            A random float.
        """
        return float(self._rng.random())

    def randn(self, *shape: int) -> np.ndarray:
        """Generate random values from standard normal distribution.

        Args:
            *shape: Shape of the output array.

        Returns:
            Array of random normal values (float32).
        """
        return self._rng.standard_normal(shape).astype(np.float32)

    def randn_f64(self, *shape: int) -> np.ndarray:
        """Generate random values from standard normal distribution.

        Args:
            *shape: Shape of the output array.

        Returns:
            Array of random normal values (float64).
        """
        return self._rng.standard_normal(shape)

    def shuffle(self, sequence: List[T]) -> None:
        """Shuffle a list in place.

        Args:
            sequence: List to shuffle.
        """
        self._rng.shuffle(sequence)

    def gen_bool(self, probability: float = 0.5) -> bool:
        """Generate a random boolean.

        Args:
            probability: Probability of True.

        Returns:
            A random boolean.
        """
        return bool(self._rng.random() < probability)


# Cross-language compatibility tests
if __name__ == "__main__":
    # These values can be compared with Rust to verify compatibility
    print(f"derive_seed(42, 'test') = {derive_seed(42, 'test')}")

    rng = SeededRng(42)
    print(f"SeededRng(42).gen_u64() = {rng.gen_u64()}")

    # Test determinism
    rng1 = SeededRng(12345)
    rng2 = SeededRng(12345)
    for i in range(5):
        v1, v2 = rng1.gen_u64(), rng2.gen_u64()
        print(f"  gen_u64()[{i}]: {v1} == {v2}: {v1 == v2}")

    # Test derivation
    master = SeededRng(12345)
    shapes = master.derive("shapes")
    data = master.derive("data")
    print(f"\nDerived seeds:")
    print(f"  shapes: {shapes.seed}")
    print(f"  data: {data.seed}")
