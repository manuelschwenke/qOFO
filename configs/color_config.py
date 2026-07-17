"""Shared TU Darmstadt corporate colour palette.

Keep visualisation modules on one stable ordering instead of copying
hex values into individual plotters.
"""
from __future__ import annotations

from typing import List


TU_COLOURS: List[str] = [
    "#B1BD00",  # 0  - 5c  Yellow-green (PANTONE 390)
    "#004E8A",  # 1  - 1c  Dark blue    (PANTONE 2945)
    "#CC4C03",  # 2  - 8c  Dark orange  (PANTONE 173)
    "#D7AC00",  # 3  - 6c  Gold         (PANTONE 110)
    "#008877",  # 4  - 3c  Teal         (PANTONE 3285)
    "#951169",  # 5  - 10c Magenta      (PANTONE 249)
    "#7FAB16",  # 6  - 4c  Olive green  (PANTONE 376)
    "#00689D",  # 7  - 2c  Mid blue     (PANTONE 3015)
    "#B90F22",  # 8  - 9c  Red          (PANTONE 193)
    "#D28700",  # 9  - 7c  Amber        (PANTONE 124)
    "#611C73",  # 10 - 11c Purple       (PANTONE 268)
]

# TU Darmstadt 5c is the primary colour across all visualisations.
TU_PRIMARY: str = TU_COLOURS[0]

__all__ = ["TU_COLOURS", "TU_PRIMARY"]
