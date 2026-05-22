"""Geometry utilities for C. elegans neurons."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from .id_normalization import normalize_celegans_id, normalize_id_list
from .c_elegans_connectome import DORSAL_B, VENTRAL_B, PROPRIO_IDS

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
CONNECTOME_ROOTS = [
    Path("/home/amin/Research/NCAP/external/ConnectomeToolbox"),
    WORKSPACE_ROOT / "external" / "ConnectomeToolbox",
    WORKSPACE_ROOT / "ConnectomeToolbox",
    WORKSPACE_ROOT.parent / "external" / "ConnectomeToolbox",
    WORKSPACE_ROOT.parent / "ConnectomeToolbox",
]
CONNECTOME_DATA_DIRS = [root / "cect" / "data" for root in CONNECTOME_ROOTS]
GEOMETRY_SOURCE = next(
    ((data_dir / "Worm2D.net.nml").resolve() for data_dir in CONNECTOME_DATA_DIRS if (data_dir / "Worm2D.net.nml").exists()),
    (CONNECTOME_DATA_DIRS[0] / "Worm2D.net.nml").resolve(),
)

# Commonly used unpaired neurons in locomotion/prior mapping that are easy to miss
# in reduced geometry sources.
UNPAIRED_REFERENCE_IDS = ("DVA", "AVG", "RID", "PVNL", "PVNR")


def _cell_centroid(cell) -> np.ndarray:
    """Compute centroid of all proximal/distal points in a NeuroML cell."""
    points = []
    morphology = getattr(cell, "morphology", None)
    if morphology is None or getattr(morphology, "segments", None) is None:
        return np.full(3, np.nan, dtype=np.float32)

    for segment in morphology.segments:
        proximal = getattr(segment, "proximal", None)
        distal = getattr(segment, "distal", None)
        if proximal is not None:
            points.append([proximal.x, proximal.y, proximal.z])
        if distal is not None:
            points.append([distal.x, distal.y, distal.z])

    if not points:
        return np.full(3, np.nan, dtype=np.float32)

    return np.asarray(points, dtype=np.float32).mean(axis=0)


def load_neuron_positions_from_neuroml(cells_dir: str) -> Dict[str, np.ndarray]:
    """Load neuron centroids from NeuroML cell files or net files."""
    import xml.etree.ElementTree as ET

    try:
        from neuroml.loaders import read_neuroml2_file
    except ImportError:
        read_neuroml2_file = None

    positions: Dict[str, np.ndarray] = {}
    root = Path(cells_dir)
    targets = list(root.rglob("*.cell.nml"))
    if not targets and root.is_file() and root.suffix.lower() == ".nml":
        targets = [root]

    if read_neuroml2_file:
        for cell_path in targets:
            doc = read_neuroml2_file(str(cell_path))
            for cell in getattr(doc, "cells", []):
                positions[normalize_celegans_id(cell.id)] = _cell_centroid(cell)
            for net in getattr(doc, "networks", []):
                for pop in getattr(net, "populations", []):
                    comp = getattr(pop, "component", None)
                    for inst in getattr(pop, "instances", []):
                        loc = getattr(inst, "location", None)
                        if loc is None:
                            continue
                        inst_id = getattr(inst, "id", None)
                        name = normalize_celegans_id(f"{comp or pop.id}{int(inst_id)+1 if inst_id is not None else ''}")
                        positions[name] = np.array(
                            [float(loc.x), float(loc.y), float(loc.z)], dtype=np.float32
                        )
        if positions:
            return positions

    # Fallback simple XML parsing (handles Worm2D.net.nml).
    generic_components = {
        "GENERICNEURONCELL",
        "GENERICMUSCLECELL",
        "GENERICCELL",
        "NEURON",
        "MUSCLE",
        "CELL",
    }

    for cell_path in targets:
        try:
            tree = ET.parse(cell_path)
        except Exception:
            continue
        root_el = tree.getroot()
        ns = {"nml": "http://www.neuroml.org/schema/neuroml2"}
        for pop in root_el.findall(".//nml:population", ns):
            pop_id = (pop.get("id", "") or "").strip()
            comp = (pop.get("component", "") or "").strip()
            n_instances = len(pop.findall("./nml:instance", ns))

            # Some NeuroML files use generic components (e.g. GenericNeuronCell) and
            # place the actual neuron ID in population id (e.g. AVBL).
            use_pop_id = bool(pop_id) and (
                not comp
                or comp.upper() in generic_components
                or n_instances == 1
            )
            base = pop_id if use_pop_id else comp

            for inst in pop.findall("./nml:instance", ns):
                inst_id = inst.get("id")
                loc = inst.find("./nml:location", ns)
                if loc is None:
                    continue
                try:
                    x = float(loc.get("x", "0"))
                    y = float(loc.get("y", "0"))
                    z = float(loc.get("z", "0"))
                except Exception:
                    continue
                try:
                    idx = int(inst_id) + 1 if inst_id is not None else 0
                    if base and n_instances > 1:
                        name = f"{base}{idx:02d}"
                    elif base:
                        name = base
                    else:
                        name = f"pop_{len(positions)}"
                except Exception:
                    name = f"{base}{inst_id}" if base else f"pop_{len(positions)}"
                positions[normalize_celegans_id(name)] = np.array([x, y, z], dtype=np.float32)

    if not positions:
        raise FileNotFoundError(f"No NeuroML positions found in {cells_dir}")
    return positions


def load_neuron_positions_from_csv(
    path: str, name_col: str, x_col: str, y_col: str, z_col: str
) -> Dict[str, np.ndarray]:
    """Load neuron centroids from a CSV export."""
    df = pd.read_csv(path)
    positions = {}
    for row in df.itertuples(index=False):
        name = getattr(row, name_col)
        pos = np.array(
            [float(getattr(row, x_col)), float(getattr(row, y_col)), float(getattr(row, z_col))],
            dtype=np.float32,
        )
        positions[normalize_celegans_id(name)] = pos
    return positions


def discover_neuron_geometry(
    preferred_path: Optional[str] = None, search_roots: Optional[Iterable[str]] = None
) -> Tuple[str, str]:
    """Find neuron geometry. Prefers Worm2D.net.nml then CSV with x/y/z columns."""
    if preferred_path:
        p = Path(preferred_path)
        if p.exists():
            kind = "neuroml" if (p.is_dir() or p.suffix.lower() == ".nml") else "csv"
            return str(p), kind

    for data_dir in CONNECTOME_DATA_DIRS:
        worm2d = (data_dir / "Worm2D.net.nml").resolve()
        if worm2d.exists():
            return str(worm2d), "neuroml"
    # Fallback: search for any Worm2D*.net.nml within candidate ConnectomeToolbox roots.
    for root in CONNECTOME_ROOTS:
        if not root.exists():
            continue
        for cand in root.rglob("Worm2D*.net.nml"):
            return str(cand.resolve()), "neuroml"

    if search_roots is None:
        search_roots = [
            str(WORKSPACE_ROOT / "external" / "ConnectomeToolbox" / "cect" / "data"),
            str(WORKSPACE_ROOT.parent / "external" / "ConnectomeToolbox" / "cect" / "data"),
            str(WORKSPACE_ROOT / "ConnectomeToolbox" / "cect" / "data"),
            "ConnectomeToolbox",
            "connectome_toolbox",
            "../ConnectomeToolbox",
            "CElegansNeuroML",
            "../CElegansNeuroML",
        ]

    csv_candidates = []
    neuroml_candidates = []
    for root in search_roots:
        root_path = Path(root)
        if root_path.is_file():
            if root_path.suffix.lower() == ".nml":
                neuroml_candidates.append(root_path)
            else:
                csv_candidates.append(root_path)
            continue
        if not root_path.exists():
            continue
        neuroml_candidates.extend(root_path.rglob("generatedNeuroML2"))
        neuroml_candidates.extend(root_path.rglob("*.net.nml"))
        csv_candidates.extend(root_path.rglob("*.csv"))
        csv_candidates.extend(root_path.rglob("*.tsv"))

    def _score_csv(path: Path) -> int:
        try:
            df = pd.read_csv(path, nrows=50, sep=None, engine="python")
        except Exception:
            return -1
        cols = {c.lower() for c in df.columns}
        needed = {"x", "y", "z"}
        name_like = {"name", "cell", "neuron", "id", "label"}
        if not needed.issubset(cols):
            return 0
        if not (cols & name_like):
            return 0
        return len(cols & needed) + len(cols & name_like)

    best_csv = None
    best_score = -1
    for cand in csv_candidates:
        score = _score_csv(cand)
        if score > best_score:
            best_score = score
            best_csv = cand

    if best_csv is not None and best_score > 0:
        try:
            df = pd.read_csv(best_csv, nrows=1, sep=None, engine="python")
            print(f"Using neuron positions from: {best_csv.resolve()} with columns {list(df.columns)}")
        except Exception:
            pass
        return str(best_csv), "csv"

    for cand in neuroml_candidates:
        if list(cand.rglob("*.cell.nml")):
            return str(cand), "neuroml"
        if cand.is_file() and cand.suffix.lower() == ".nml":
            return str(cand), "neuroml"

    raise FileNotFoundError(
        "Could not find neuron geometry with x/y/z columns in ConnectomeToolbox. "
        "Please provide a WormNeuroAtlas export or NeuroML cells."
    )


def load_neuron_positions_auto(
    preferred_path: Optional[str] = None,
    name_col: str = "name",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
) -> Dict[str, np.ndarray]:
    """Discover geometry source and load neuron positions."""
    try:
        path, kind = discover_neuron_geometry(preferred_path=preferred_path)
    except FileNotFoundError:
        # Last-resort: try hardcoded Worm2D path
        worm2d = GEOMETRY_SOURCE
        if worm2d.exists():
            print(f"Falling back to geometry from {worm2d}")
            path, kind = str(worm2d), "neuroml"
        else:
            raise

    if kind == "neuroml":
        return load_neuron_positions_from_neuroml(path)
    return load_neuron_positions_from_csv(path, name_col=name_col, x_col=x_col, y_col=y_col, z_col=z_col)


def build_distance_matrix(positions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Build a pairwise Euclidean distance matrix."""
    neurons = sorted(positions)
    coords = np.stack([positions[n] for n in neurons], axis=0)  # (N, 3)
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.linalg.norm(diff, axis=-1)
    return D.astype(np.float32), neurons


def build_distance_priors(positions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Construct distance matrices with masks for known entries."""
    import torch

    D_all, names_all = build_distance_matrix(positions)
    dist = torch.as_tensor(D_all, dtype=torch.float32)
    mask = torch.ones_like(dist, dtype=torch.bool)

    # Build masks for substructures.
    b_re = re.compile(r"^.[VB]\\d+", flags=re.IGNORECASE)
    m_re = re.compile(r"^M[VD][LR]?\\d+", flags=re.IGNORECASE)
    b_names = [n for n in names_all if b_re.match(n)]
    m_names = [n for n in names_all if m_re.match(n)]

    def submatrix(src_list, tgt_list):
        if not src_list or not tgt_list:
            return torch.zeros((0, 0), dtype=torch.float32), torch.zeros((0, 0), dtype=torch.bool)
        idx_src = [names_all.index(n) for n in src_list]
        idx_tgt = [names_all.index(n) for n in tgt_list]
        return dist[idx_src][:, idx_tgt], mask[idx_src][:, idx_tgt]

    D_bb, M_bb = submatrix(b_names, b_names)
    D_bm, M_bm = submatrix(b_names, m_names)
    D_mb, M_mb = submatrix(m_names, b_names)
    D_m, M_m = submatrix(m_names, m_names)

    return {
        "all": {"dist": dist, "mask": mask, "names": names_all},
        "bb": {"dist": D_bb, "mask": M_bb, "names_src": b_names, "names_tgt": b_names},
        "bm": {"dist": D_bm, "mask": M_bm, "names_src": b_names, "names_tgt": m_names},
        "mb": {"dist": D_mb, "mask": M_mb, "names_src": m_names, "names_tgt": b_names},
        "mm": {"dist": D_m, "mask": M_m, "names_src": m_names, "names_tgt": m_names},
    }


def align_matrices(
    A: np.ndarray, neurons_A: List[str], D: np.ndarray, neurons_D: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Align adjacency and distance matrices to common neuron ordering."""
    common = sorted(set(neurons_A).intersection(neurons_D))
    idx_A = {n: i for i, n in enumerate(neurons_A)}
    idx_D = {n: i for i, n in enumerate(neurons_D)}

    A_aligned = np.zeros((len(common), len(common)), dtype=A.dtype)
    D_aligned = np.zeros((len(common), len(common)), dtype=D.dtype)
    for i, n_i in enumerate(common):
        for j, n_j in enumerate(common):
            A_aligned[i, j] = A[idx_A[n_i], idx_A[n_j]]
            D_aligned[i, j] = D[idx_D[n_i], idx_D[n_j]]
    return A_aligned, D_aligned, common


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"n": 0, "mean": math.nan, "median": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def distance_statistics(A: np.ndarray, D: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute summary stats for edge vs non-edge distances."""
    edge_mask = A > 0
    non_edge_mask = ~edge_mask
    np.fill_diagonal(non_edge_mask, False)

    edge_distances = D[edge_mask]
    non_edge_distances = D[non_edge_mask]

    return {
        "edge": _summary_stats(edge_distances),
        "non_edge": _summary_stats(non_edge_distances),
    }


def build_bneuron_muscle_distances(
    positions: Dict[str, np.ndarray],
    neuron_name_pattern: str = r"^.[VB][0-9]+",
    muscle_name_pattern: str = r"^M[VD][LR]?[0-9]+",
) -> Dict[str, object]:
    """Filter positions to B-neurons and muscles and build distance matrices."""
    import re

    b_re = re.compile(neuron_name_pattern, flags=re.IGNORECASE)
    m_re = re.compile(muscle_name_pattern, flags=re.IGNORECASE)

    b_positions = {k: v for k, v in positions.items() if isinstance(k, str) and b_re.match(k)}
    m_positions = {k: v for k, v in positions.items() if isinstance(k, str) and m_re.match(k)}

    D_all, names_all = build_distance_matrix({**b_positions, **m_positions})
    D_b, b_names = build_distance_matrix(b_positions) if b_positions else (np.zeros((0, 0)), [])
    D_m, m_names = build_distance_matrix(m_positions) if m_positions else (np.zeros((0, 0)), [])

    def _side(name: str) -> str:
        u = name.upper()
        return "dorsal" if "D" in u else ("ventral" if "V" in u else "unknown")

    def _seg(name: str) -> Optional[int]:
        import re as _re
        m = _re.search(r"(\\d+)", name)
        return int(m.group(1)) if m else None

    b_side = {n: _side(n) for n in b_names}
    m_side = {n: _side(n) for n in m_names}
    b_seg = {n: _seg(n) for n in b_names}
    m_seg = {n: _seg(n) for n in m_names}

    D_bm_ipsi = np.zeros((len(b_names), len(m_names)), dtype=np.float32)
    D_bm_contra = np.zeros_like(D_bm_ipsi)
    for i, bn in enumerate(b_names):
        for j, mn in enumerate(m_names):
            if b_side[bn] != "unknown" and m_side[mn] != "unknown" and b_side[bn] == m_side[mn]:
                D_bm_ipsi[i, j] = np.linalg.norm(b_positions[bn] - m_positions[mn])
            else:
                D_bm_contra[i, j] = np.linalg.norm(b_positions[bn] - m_positions[mn])

    D_mb_ipsi = D_bm_ipsi.T
    D_mb_contra = D_bm_contra.T

    return {
        "b_neurons": b_names,
        "muscles": m_names,
        "positions_b": b_positions,
        "positions_m": m_positions,
        "D_all": D_all,
        "D_b": D_b,
        "D_m": D_m,
        "D_bm_ipsi": D_bm_ipsi,
        "D_bm_contra": D_bm_contra,
        "D_mb_ipsi": D_mb_ipsi,
        "D_mb_contra": D_mb_contra,
        "b_segments": b_seg,
        "m_segments": m_seg,
        "names_all": names_all,
    }


def _is_muscle_id(name: str) -> bool:
    """Heuristic muscle detector for inventory/priors."""
    u = name.upper()
    prefixes = ("MDL", "MDR", "MVL", "MVR", "MD", "MV")
    return any(u.startswith(p) for p in prefixes)


def extract_geometry_ids(positions: Dict[str, np.ndarray], source: Path) -> pd.DataFrame:
    """Inventory of geometry IDs."""
    rows = []
    seen_ids = set()
    for nid, coord in positions.items():
        norm_id = normalize_celegans_id(str(nid))
        seen_ids.add(norm_id)
        coord_arr = np.asarray(coord)
        has_xyz = coord_arr.shape == (3,) and np.isfinite(coord_arr).all()
        rows.append(
            {
                "id": norm_id,
                "type": "muscle" if _is_muscle_id(norm_id) else "neuron",
                "has_xyz": bool(has_xyz),
                "x": coord_arr[0] if has_xyz else "",
                "y": coord_arr[1] if has_xyz else "",
                "z": coord_arr[2] if has_xyz else "",
                "source": str(source),
            }
        )
    for ref_id in UNPAIRED_REFERENCE_IDS:
        if ref_id in seen_ids:
            continue
        rows.append(
            {
                "id": ref_id,
                "type": "neuron",
                "has_xyz": False,
                "x": "",
                "y": "",
                "z": "",
                "source": str(source),
            }
        )
    return pd.DataFrame(rows)


def _distance_with_mask(positions: Dict[str, np.ndarray], src: str, dst: str) -> tuple[float, bool]:
    src_pos = positions.get(src)
    dst_pos = positions.get(dst)
    if src_pos is None or dst_pos is None:
        return 0.0, False
    src_arr = np.asarray(src_pos, dtype=np.float32)
    dst_arr = np.asarray(dst_pos, dtype=np.float32)
    if src_arr.shape != (3,) or dst_arr.shape != (3,) or not np.isfinite(src_arr).all() or not np.isfinite(dst_arr).all():
        return 0.0, False
    return float(np.linalg.norm(src_arr - dst_arr)), True


def build_distance_priors_from_geometry(positions: Dict[str, np.ndarray], n_joints: int = 6) -> Dict[str, object]:
    """Build distance priors with masks based on geometry."""
    norm_positions = {normalize_celegans_id(str(k)): np.asarray(v, dtype=np.float32) for k, v in positions.items()}
    prop_id = normalize_celegans_id("DVA")

    dorsal_ids = normalize_id_list(DORSAL_B[: n_joints + 1])
    ventral_ids = normalize_id_list(VENTRAL_B[: n_joints + 1])

    def _pairwise(ids: List[str]) -> List[tuple[str, str]]:
        return [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]

    pairs_d = _pairwise(dorsal_ids)
    pairs_v = _pairwise(ventral_ids)

    dist_d, mask_d = [], []
    for dst in dorsal_ids:
        d, m = _distance_with_mask(norm_positions, prop_id, dst)
        dist_d.append(d)
        mask_d.append(m)

    dist_v, mask_v = [], []
    for dst in ventral_ids:
        d, m = _distance_with_mask(norm_positions, prop_id, dst)
        dist_v.append(d)
        mask_v.append(m)

    bb_dist_d, bb_mask_d = [], []
    for src, dst in pairs_d:
        d, m = _distance_with_mask(norm_positions, src, dst)
        bb_dist_d.append(d)
        bb_mask_d.append(m)

    bb_dist_v, bb_mask_v = [], []
    for src, dst in pairs_v:
        d, m = _distance_with_mask(norm_positions, src, dst)
        bb_dist_v.append(d)
        bb_mask_v.append(m)

    muscles = [nid for nid in norm_positions.keys() if _is_muscle_id(nid)]

    def _muscle_side(mid: str) -> Optional[str]:
        u = mid.upper()
        if u.startswith("MD"):
            return "dorsal"
        if u.startswith("MV"):
            return "ventral"
        return None

    dorsal_muscles = [m for m in muscles if _muscle_side(m) == "dorsal"]
    ventral_muscles = [m for m in muscles if _muscle_side(m) == "ventral"]

    dist_b_to_m_d = np.zeros((len(dorsal_ids), len(dorsal_muscles)), dtype=np.float32)
    mask_b_to_m_d = np.zeros_like(dist_b_to_m_d, dtype=bool)
    for i, b_id in enumerate(dorsal_ids):
        for j, m_id in enumerate(dorsal_muscles):
            d, m = _distance_with_mask(norm_positions, b_id, m_id)
            dist_b_to_m_d[i, j] = d
            mask_b_to_m_d[i, j] = m

    dist_b_to_m_v = np.zeros((len(ventral_ids), len(ventral_muscles)), dtype=np.float32)
    mask_b_to_m_v = np.zeros_like(dist_b_to_m_v, dtype=bool)
    for i, b_id in enumerate(ventral_ids):
        for j, m_id in enumerate(ventral_muscles):
            d, m = _distance_with_mask(norm_positions, b_id, m_id)
            dist_b_to_m_v[i, j] = d
            mask_b_to_m_v[i, j] = m

    return {
        "prop_to_B": {
            "src_id": prop_id,
            "dst_ids_d": dorsal_ids,
            "dst_ids_v": ventral_ids,
            "dist_d": torch.as_tensor(dist_d, dtype=torch.float32),
            "mask_d": torch.as_tensor(mask_d, dtype=torch.bool),
            "dist_v": torch.as_tensor(dist_v, dtype=torch.float32),
            "mask_v": torch.as_tensor(mask_v, dtype=torch.bool),
        },
        "B_to_B_next": {
            "pairs_d": pairs_d,
            "pairs_v": pairs_v,
            "dist_d": torch.as_tensor(bb_dist_d, dtype=torch.float32),
            "mask_d": torch.as_tensor(bb_mask_d, dtype=torch.bool),
            "dist_v": torch.as_tensor(bb_dist_v, dtype=torch.float32),
            "mask_v": torch.as_tensor(bb_mask_v, dtype=torch.bool),
        },
        "B_to_muscle": {
            "b_ids_d": dorsal_ids,
            "b_ids_v": ventral_ids,
            "muscle_ids_d": dorsal_muscles,
            "muscle_ids_v": ventral_muscles,
            "dist_b_to_m_d": torch.as_tensor(dist_b_to_m_d, dtype=torch.float32),
            "mask_b_to_m_d": torch.as_tensor(mask_b_to_m_d, dtype=torch.bool),
            "dist_b_to_m_v": torch.as_tensor(dist_b_to_m_v, dtype=torch.float32),
            "mask_b_to_m_v": torch.as_tensor(mask_b_to_m_v, dtype=torch.bool),
        },
    }


def segment_centroids(positions: Dict[str, np.ndarray], segment_regex: str = r"(\\d{1,2})") -> Dict[int, np.ndarray]:
    """Average positions per body segment based on digits in the neuron name."""
    import re

    pattern = re.compile(segment_regex)
    accumulator: Dict[int, List[np.ndarray]] = {}
    for name, pos in positions.items():
        match = pattern.search(name)
        if match:
            try:
                seg = int(match.group(1))
            except ValueError:
                continue
            accumulator.setdefault(seg, []).append(pos)
    centroids = {}
    for seg, pts in accumulator.items():
        centroids[seg] = np.stack(pts).mean(axis=0)
    return centroids
