"""Human-readable byte sizes (decimal KB / MB / GB)."""
from __future__ import annotations

_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def fmt_bytes(n: float | int) -> str:
    """Format byte count as decimal B / KB / MB / GB (1000-based)."""
    n = float(n)
    for unit in _UNITS:
        if abs(n) < 1000.0:
            if unit == "B":
                return f"{int(n)} {unit}" if n == int(n) else f"{n:.1f} {unit}"
            return f"{n:.1f} {unit}"
        n /= 1000.0
    return f"{n:.1f} PB"


def mib_to_bytes(mib: float) -> int:
    """nvidia-smi reports MiB — convert to exact bytes."""
    return int(round(mib * 1024 * 1024))