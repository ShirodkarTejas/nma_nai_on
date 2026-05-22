"""Helpers for normalizing C. elegans IDs (DB04 format)."""

from __future__ import annotations

import re
from typing import Iterable, List


def normalize_celegans_id(name: str) -> str:
    """Normalize neuron IDs to uppercase and zero-padded numeric suffix for motor neurons.

    Examples:
        DB4 -> DB04
        VB6 -> VB06
        VA1 -> VA01
        VD3 -> VD03
        DD1 -> DD01
        DA2 -> DA02
        AS5 -> AS05
        DVA -> DVA (unchanged)
        MDL01 -> MDL01 (unchanged)
    """
    nid = name.strip().upper()
    motor_re = re.compile(r"^(DB|VB|VA|VD|DD|DA|AS)(\d+)$")
    m = motor_re.match(nid)
    if m:
        return f"{m.group(1)}{int(m.group(2)):02d}"
    return nid


def normalize_id_list(ids: Iterable[str]) -> List[str]:
    return [normalize_celegans_id(i) for i in ids]
