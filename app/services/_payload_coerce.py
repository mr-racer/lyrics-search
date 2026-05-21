"""Shared coercion helpers for messy Qdrant payload values.

Legacy import artefacts: `duration` may carry a bucket-range string like
"203-234" (older indexer overwrote the numeric seconds with the percentile
bucket label), and `year` may carry a decade range "1990-1999". Both shapes
crash naive int()/float() — these helpers absorb them so Pydantic models
that expect Optional[float]/Optional[int] don't 500.
"""

from __future__ import annotations

from typing import Optional


def coerce_float(val) -> Optional[float]:
    """Return val as float when possible. Hyphen ranges average to midpoint.
    Empty / unparseable returns None (callers can default to 0.0 if needed).
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            if "-" in s:
                try:
                    nums = [float(p.strip()) for p in s.split("-") if p.strip()]
                    if nums:
                        return sum(nums) / len(nums)
                except ValueError:
                    pass
    return None


def coerce_year(val) -> Optional[int]:
    """Return val as int year when sensible. Ranges 'YYYY-YYYY' yield the first."""
    if isinstance(val, int) and val > 0:
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            n = int(s)
            return n if n > 0 else None
        except ValueError:
            if "-" in s:
                for p in s.split("-"):
                    try:
                        n = int(p.strip())
                        if n > 0:
                            return n
                    except ValueError:
                        continue
    return None
