"""Seeded RNG for reproducible fuzzing across Rust and Python.

This module provides a deterministic RNG that can derive sub-RNGs
for different domains (shapes, dtypes, layouts, data) from a master seed.
The derivation uses Blake2b to ensure cross-language compatibility with Rust.

The underlying PRNG uses xorshift128+ which is implemented identically
in Rust, ensuring bit-for-bit reproducibility between both languages.
"""

import hashlib
import math
from typing import List, TypeVar, Sequence

import numpy as np

T = TypeVar("T")


def _splitmix64(seed: int) -> int:
    """SplitMix64: deterministic state expansion from a single u64.

    Matches the Rust implementation exactly:
    z = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
    """
    MASK = 0xFFFFFFFFFFFFFFFF
    z = (seed + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    z = (z ^ (z >> 31)) & MASK
    return z


def _xorshift128plus(s0: int, s1: int) -> tuple:
    """xorshift128+ PRNG step. Returns (random_u64, new_s0, new_s1).

    This matches the Rust implementation exactly.
    """
    MASK = 0xFFFFFFFFFFFFFFFF
    x = s0
    y = s1
    s0_new = y
    x = (x ^ ((x << 23) & MASK)) & MASK
    x = (x ^ (x >> 17)) & MASK
    x = (x ^ y ^ (y >> 26)) & MASK
    s1_new = (x ^ y) & MASK
    result = (x + y) & MASK
    return result, s0_new, s1_new


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
    seed = seed & 0xFFFFFFFFFFFFFFFF

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

    Uses xorshift128+ as the underlying PRNG for cross-language
    compatibility with the Rust implementation. Seed derivation uses
    Blake2b-64. Initialization uses SplitMix64 to expand a single u64
    seed into the 128-bit state.

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
        # Initialize state using SplitMix64, matching Rust exactly
        s0 = _splitmix64(self._seed)
        s1 = _splitmix64((self._seed + 1) & 0xFFFFFFFFFFFFFFFF)
        # Ensure non-zero state
        if s0 == 0 and s1 == 0:
            s0 = 1
        self._s0 = s0
        self._s1 = s1

    @property
    def seed(self) -> int:
        """Get the seed used to create this RNG."""
        return self._seed

    def _next_u64(self) -> int:
        """Generate the next random u64 using xorshift128+."""
        result, self._s0, self._s1 = _xorshift128plus(self._s0, self._s1)
        return result

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
        idx = self._next_u64() % len(options)
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
        return self._next_u64() % length

    def gen_range(self, low: int, high: int) -> int:
        """Generate a random integer in [low, high).

        Args:
            low: Lower bound (inclusive).
            high: Upper bound (exclusive).

        Returns:
            A random integer.
        """
        width = high - low
        return low + (self._next_u64() % width)

    def gen_u64(self) -> int:
        """Generate a random u64.

        Returns:
            A random 64-bit unsigned integer.
        """
        return self._next_u64()

    def gen_f32(self) -> float:
        """Generate a random float in [0, 1).

        Returns:
            A random float32-range value.
        """
        x = self._next_u64()
        # Use upper 24 bits for mantissa, matching Rust
        return (x >> 40) / (1 << 24)

    def gen_f64(self) -> float:
        """Generate a random float in [0, 1).

        Returns:
            A random float.
        """
        x = self._next_u64()
        # Use upper 53 bits for mantissa, matching Rust
        return (x >> 11) / (1 << 53)

    def randn(self, *shape: int) -> np.ndarray:
        """Generate random values from standard normal distribution.

        Args:
            *shape: Shape of the output array.

        Returns:
            Array of random normal values (float32).
        """
        return self._box_muller_f32(shape).astype(np.float32)

    def randn_f64(self, *shape: int) -> np.ndarray:
        """Generate random values from standard normal distribution.

        Args:
            *shape: Shape of the output array.

        Returns:
            Array of random normal values (float64).
        """
        return self._box_muller_f64(shape)

    def _box_muller_f64(self, shape) -> np.ndarray:
        """Box-Muller transform for f64 normal distribution."""
        n = 1
        for s in shape:
            n *= s
        result = np.empty(n, dtype=np.float64)
        for i in range(0, n, 2):
            u1 = self.gen_f64()
            u2 = self.gen_f64()
            u1 = max(u1, 1e-300)
            r = math.sqrt(-2.0 * math.log(u1))
            theta = 2.0 * math.pi * u2
            result[i] = r * math.cos(theta)
            if i + 1 < n:
                result[i + 1] = r * math.sin(theta)
        return result.reshape(shape)

    def _box_muller_f32(self, shape) -> np.ndarray:
        """Box-Muller transform for f32 normal distribution."""
        return self._box_muller_f64(shape).astype(np.float32)

    def shuffle(self, sequence: List[T]) -> None:
        """Shuffle a list in place.

        Args:
            sequence: List to shuffle.
        """
        for i in range(len(sequence) - 1, 0, -1):
            j = self._next_u64() % (i + 1)
            sequence[i], sequence[j] = sequence[j], sequence[i]

    def gen_bool(self, probability: float = 0.5) -> bool:
        """Generate a random boolean.

        Args:
            probability: Probability of True.

        Returns:
            A random boolean.
        """
        return self.gen_f64() < probability


# Cross-language compatibility tests
if __name__ == "__main__":
    # These values must match the Rust implementation exactly
    print(f"derive_seed(42, 'test') = {derive_seed(42, 'test')}")

    rng = SeededRng(42)
    print(f"SeededRng(42).gen_u64() = {rng.gen_u64()}")
    print(f"SeededRng(42).gen_f64() = {rng.gen_f64()}")

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

    # Verify SplitMix64 initialization
    print(f"\nsplitmix64(42) = {_splitmix64(42)}")
    print(f"splitmix64(43) = {_splitmix64(43)}")
