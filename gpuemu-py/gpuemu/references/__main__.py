"""Run a built-in fp64 reference over the daemon protocol.

Usage: ``python -m gpuemu.references <op>`` where ``<op>`` is a key of
:data:`gpuemu.references.ops.REGISTRY`. Reads the daemon's stdin/stdout JSON.
"""

import sys

from .ops import REGISTRY, run


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in REGISTRY:
        keys = ", ".join(sorted(REGISTRY))
        sys.stderr.write(f"usage: python -m gpuemu.references <op>\nops: {keys}\n")
        raise SystemExit(2)
    run(REGISTRY[sys.argv[1]])


if __name__ == "__main__":
    main()
