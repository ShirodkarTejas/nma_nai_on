"""Aggregate distance priors for NCAP connections."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from .c_elegans_geometry import load_neuron_positions_auto
from .c_elegans_connectome import DORSAL_B, VENTRAL_B
from .id_normalization import normalize_celegans_id, normalize_id_list

DATA_DIR = (Path(__file__).resolve().parent / "data")
DATA_DIR.mkdir(exist_ok=True)
DIST_CSV = DATA_DIR / "distance_priors.csv"


def _side(name: str) -> str:
    u = name.upper()
    return "dorsal" if "D" in u else ("ventral" if "V" in u else "unknown")


def _is_muscle_id(name: str) -> bool:
    u = name.upper()
    prefixes = ("MDL", "MDR", "MVL", "MVR", "MD", "MV")
    return any(u.startswith(p) for p in prefixes)


def _segment(name: str) -> int | None:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def load_ids(positions: Dict[str, np.ndarray]) -> Tuple[Iterable[str], Iterable[str]]:
    """Derive B-neuron and muscle IDs directly from available geometry."""
    norm_positions = {normalize_celegans_id(str(k)): np.asarray(v, dtype=np.float32) for k, v in positions.items()}
    b_candidates = normalize_id_list(DORSAL_B + VENTRAL_B)
    b_ids = [bid for bid in b_candidates if bid in norm_positions]
    muscle_ids = sorted([nid for nid in norm_positions.keys() if _is_muscle_id(nid)])
    return b_ids, muscle_ids


def _pair_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def aggregate_distance_priors() -> Dict[str, float]:
    positions = load_neuron_positions_auto()
    b_ids, m_ids = load_ids(positions)
    prop_id = normalize_celegans_id("DVA")

    def get_pos(name):
        return positions.get(name)

    def collect(pairs):
        vals = []
        for pre, post in pairs:
            pa, pb = get_pos(pre), get_pos(post)
            if pa is None or pb is None:
                continue
            vals.append(_pair_distance(np.asarray(pa, dtype=np.float32), np.asarray(pb, dtype=np.float32)))
        return np.array(vals, dtype=np.float32)

    b_to_m_ipsi = []
    b_to_m_contra = []
    m_to_b_ipsi = []
    m_to_b_contra = []
    b_to_b_next = []
    prop_to_b = []

    if m_ids:
        for pre in b_ids:
            for post in m_ids:
                side_eq = _side(pre) == _side(post) and _side(pre) != "unknown"
                if side_eq:
                    b_to_m_ipsi.append((pre, post))
                else:
                    b_to_m_contra.append((pre, post))

        for pre in m_ids:
            for post in b_ids:
                side_eq = _side(pre) == _side(post) and _side(pre) != "unknown"
                if side_eq:
                    m_to_b_ipsi.append((pre, post))
                else:
                    m_to_b_contra.append((pre, post))

    for pre in b_ids:
        for post in b_ids:
            sp, st = _segment(pre), _segment(post)
            if sp is not None and st is not None and st == sp + 1:
                b_to_b_next.append((pre, post))
    for post in b_ids:
        prop_to_b.append((prop_id, post))

    agg = {
        "b_to_m_ipsi": collect(b_to_m_ipsi).mean() if b_to_m_ipsi else 0.0,
        "b_to_m_contra": collect(b_to_m_contra).mean() if b_to_m_contra else 0.0,
        "m_to_b_ipsi": collect(m_to_b_ipsi).mean() if m_to_b_ipsi else 0.0,
        "m_to_b_contra": collect(m_to_b_contra).mean() if m_to_b_contra else 0.0,
        "b_to_b_next": collect(b_to_b_next).mean() if b_to_b_next else 0.0,
        "prop_to_b": collect(prop_to_b).mean() if prop_to_b else 0.0,
    }
    return agg


def write_distance_priors(path: Path = DIST_CSV) -> Path:
    agg = aggregate_distance_priors()
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "mean_distance"])
        for k, v in agg.items():
            writer.writerow([k, v])
    return path


if __name__ == "__main__":
    out = write_distance_priors()
    print(f"Wrote distance priors to {out}")
