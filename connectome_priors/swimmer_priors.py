"""Segment-prior utilities and inventory refresh helpers for NCAP."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Dict, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .c_elegans_connectome import (
    CANONICAL_EDGE_PATH,
    extract_edge_ids,
    load_canonical_connectome_edgelist,
)
from .c_elegans_geometry import (
    GEOMETRY_SOURCE,
    extract_geometry_ids,
    load_neuron_positions_auto,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent


def _resolve_connectome_artifact(filename: str) -> Path:
    data_rel = Path("cect") / "data" / filename
    candidates = (
        Path("/home/amin/Research/NCAP/external/ConnectomeToolbox") / data_rel,
        WORKSPACE_ROOT / "external" / "ConnectomeToolbox" / data_rel,
        WORKSPACE_ROOT / "ConnectomeToolbox" / data_rel,
        WORKSPACE_ROOT.parent / "external" / "ConnectomeToolbox" / data_rel,
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate.resolve()
    return candidates[0].resolve()


COOK2019_ADJ_XLSX = _resolve_connectome_artifact("SI 5 Connectome adjacency matrices.xlsx")
C302_GEOMETRY_NML = _resolve_connectome_artifact("c302_C2_FW.net.nml")
COOK_CHEM_SHEET = "hermaphrodite chemical"
COOK_GAP_SHEET = "herm gap jn symmetric"

COOK_DB_NEURONS = [f"DB{i:02d}" for i in range(1, 8)]
COOK_VB_NEURONS = [f"VB{i:02d}" for i in range(1, 12)]
COOK_DD_NEURONS = [f"DD{i:02d}" for i in range(1, 7)]
COOK_VD_NEURONS = [f"VD{i:02d}" for i in range(1, 14)]
COOK_DORSAL_MUSCLES = [f"dBWML{i}" for i in range(1, 25)] + [f"dBWMR{i}" for i in range(1, 25)]
COOK_VENTRAL_MUSCLES = [f"vBWML{i}" for i in range(1, 25)] + [f"vBWMR{i}" for i in range(1, 25)]


def _write_inventory(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False)


def _refresh_inventories(
    connectome_path: Optional[str] = None,
    geometry_path: Optional[str] = None,
) -> Dict[str, object]:
    """Regenerate edge/geometry inventory CSVs and return refresh metadata."""
    data_dir = Path(__file__).resolve().parent / "data"
    edges_inventory_path = data_dir / "edges_id.csv"
    geom_inventory_path = data_dir / "geometry_id.csv"

    metadata: Dict[str, object] = {
        "sources": {},
        "counts": {},
        "warnings": [],
    }

    edges_df = pd.DataFrame(columns=["pre", "post", "n_syn", "synapse_type"])
    chosen_edge_path: Optional[Path] = None
    try:
        edges_df = load_canonical_connectome_edgelist(connectome_path)
        chosen_edge_path = Path(connectome_path).resolve() if connectome_path else CANONICAL_EDGE_PATH
    except Exception as exc:
        metadata["warnings"].append(f"connectome_load_failed: {exc}")
    metadata["sources"]["connectome"] = str(chosen_edge_path) if chosen_edge_path else None
    metadata["counts"]["edge_rows"] = int(len(edges_df))

    try:
        edge_ids_df = extract_edge_ids(edges_df)
    except Exception as exc:
        metadata["warnings"].append(f"edge_inventory_failed: {exc}")
        edge_ids_df = pd.DataFrame(columns=["id", "type", "has_edge_data", "note"])
    _write_inventory(edge_ids_df, edges_inventory_path)

    positions: Dict[str, object] = {}
    chosen_geom_path: Optional[Path] = None
    try:
        positions = load_neuron_positions_auto(preferred_path=geometry_path)
        chosen_geom_path = Path(geometry_path).resolve() if geometry_path else GEOMETRY_SOURCE
    except Exception as exc:
        metadata["warnings"].append(f"geometry_load_failed: {exc}")
    metadata["sources"]["geometry"] = str(chosen_geom_path) if chosen_geom_path else None
    metadata["counts"]["geometry_nodes"] = int(len(positions))

    try:
        geom_ids_df = extract_geometry_ids(positions, chosen_geom_path or Path(""))
    except Exception as exc:
        metadata["warnings"].append(f"geometry_inventory_failed: {exc}")
        geom_ids_df = pd.DataFrame(columns=["id", "type", "has_xyz", "x", "y", "z", "source"])
    _write_inventory(geom_ids_df, geom_inventory_path)

    return metadata


def refresh_inventory_files(
    connectome_path: Optional[str] = None,
    geometry_path: Optional[str] = None,
) -> Dict[str, object]:
    """Public utility: regenerate `edges_id.csv` and `geometry_id.csv` only."""
    return _refresh_inventories(connectome_path=connectome_path, geometry_path=geometry_path)


def _normalize_cook_name(raw_name: object) -> Optional[str]:
    if raw_name is None:
        return None
    name = re.sub(r"\s+", "", str(raw_name).strip())
    if not name or name.lower() == "nan":
        return None

    neuron = re.fullmatch(r"(DB|VB|DD|VD)0*([0-9]+)", name, flags=re.IGNORECASE)
    if neuron:
        return f"{neuron.group(1).upper()}{int(neuron.group(2)):02d}"

    muscle = re.fullmatch(r"([dv])BWM([LR])0*([0-9]+)", name, flags=re.IGNORECASE)
    if muscle:
        side = muscle.group(2).upper()
        idx = int(muscle.group(3))
        return f"{muscle.group(1).lower()}BWM{side}{idx}"

    if name.upper() == "DVA":
        return "DVA"
    return name


def _parse_cook_connectome_matrix(xlsx_path: Path, sheet_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Cook 2019 adjacency matrix not found: {xlsx_path}")
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    post_cells = np.asarray([_normalize_cook_name(v) for v in df.iloc[2, 3:].to_numpy()], dtype=object)
    pre_cells = np.asarray([_normalize_cook_name(v) for v in df.iloc[3:, 2].to_numpy()], dtype=object)
    syn_counts = pd.to_numeric(df.iloc[3:, 3:].stack(), errors="coerce").unstack(fill_value=0.0).to_numpy(dtype=float)
    return pre_cells, post_cells, syn_counts


def _parse_c302_coordinates(nml_path: Path) -> Dict[str, np.ndarray]:
    if not nml_path.exists():
        raise FileNotFoundError(f"c302 geometry file not found: {nml_path}")

    root = ET.parse(nml_path).getroot()
    ns: Dict[str, str] = {}
    if root.tag.startswith("{"):
        ns["n"] = root.tag.split("}")[0].strip("{")
        pop_xpath = ".//n:population"
        inst_xpath = "n:instance"
        loc_xpath = "n:location"
    else:
        pop_xpath = ".//population"
        inst_xpath = "instance"
        loc_xpath = "location"

    coords: Dict[str, np.ndarray] = {}
    for pop in root.findall(pop_xpath, ns):
        pop_id = pop.get("id")
        if not pop_id:
            continue
        instance = pop.find(inst_xpath, ns)
        if instance is None:
            continue
        location = instance.find(loc_xpath, ns)
        if location is None:
            continue
        try:
            x = float(location.get("x"))
            y = float(location.get("y"))
            z = float(location.get("z"))
        except (TypeError, ValueError):
            continue
        coords[pop_id.strip()] = np.array([x, y, z], dtype=np.float64)
    return coords


def _translate_cook_neuron_to_c302(cook_name: str) -> Optional[str]:
    m = re.fullmatch(r"(DB|VB|DD|VD)([0-9]{2})", cook_name, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}{int(m.group(2))}"
    if cook_name.upper() == "DVA":
        return "DVA"
    return cook_name


def _translate_cook_muscle_to_c302(cook_name: str) -> str:
    m = re.fullmatch(r"([dv])BWM([LR])([0-9]+)", cook_name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Not a valid Cook muscle name: {cook_name!r}")
    idx = int(m.group(3))
    if idx < 1 or idx > 24:
        raise ValueError(f"Cook muscle index out of range [1,24]: {cook_name!r}")
    dorsal = m.group(1).lower() == "d"
    side = m.group(2).upper()
    prefix = "MD" if dorsal else "MV"
    return f"{prefix}{side}{idx:02d}"


def _translate_cook_name_to_c302(cook_name: str) -> Optional[str]:
    if cook_name.startswith(("dBWM", "vBWM")):
        return _translate_cook_muscle_to_c302(cook_name)
    return _translate_cook_neuron_to_c302(cook_name)


def _build_name_index(names: np.ndarray) -> Dict[str, int]:
    index: Dict[str, int] = {}
    for i, raw_name in enumerate(names):
        if raw_name is None:
            continue
        name = str(raw_name).strip()
        if not name or name.lower() == "nan":
            continue
        if name not in index:
            index[name] = i
    return index


def _matrix_count(
    matrix: np.ndarray,
    pre_idx: Dict[str, int],
    post_idx: Dict[str, int],
    pre_name: str,
    post_name: str,
) -> float:
    i = pre_idx.get(pre_name)
    j = post_idx.get(post_name)
    if i is None or j is None:
        return 0.0
    if i >= matrix.shape[0] or j >= matrix.shape[1]:
        return 0.0
    value = matrix[i, j]
    if not np.isfinite(value) or value <= 0:
        return 0.0
    return float(value)


def _symmetric_gap_count(
    gap_matrix: np.ndarray,
    gap_pre_idx: Dict[str, int],
    gap_post_idx: Dict[str, int],
    cell_a: str,
    cell_b: str,
) -> float:
    forward = _matrix_count(gap_matrix, gap_pre_idx, gap_post_idx, cell_a, cell_b)
    backward = _matrix_count(gap_matrix, gap_pre_idx, gap_post_idx, cell_b, cell_a)
    return max(forward, backward)


def _distance_from_cook_cells(coords: Dict[str, np.ndarray], pre_cook: str, post_cook: str) -> Optional[float]:
    pre_c302 = _translate_cook_name_to_c302(pre_cook)
    post_c302 = _translate_cook_name_to_c302(post_cook)
    if pre_c302 is None or post_c302 is None:
        return None
    pre_xyz = coords.get(pre_c302)
    post_xyz = coords.get(post_c302)
    if pre_xyz is None or post_xyz is None:
        return None
    return float(np.linalg.norm(pre_xyz - post_xyz))


def generate_ncap_segment_priors(num_segments: int) -> Dict[str, object]:
    """Generate sparse pathway priors from Cook2019 + OpenWorm c302 geometry.

    Note:
      - `num_segments` is retained for API compatibility with existing call sites.
      - Returned priors are scalar pathway averages, not dense matrices.
    """
    if not isinstance(num_segments, int) or num_segments <= 0:
        raise ValueError(f"num_segments must be a positive integer, got {num_segments!r}")

    edge_list_summary = {
        "path": str(CANONICAL_EDGE_PATH) if CANONICAL_EDGE_PATH else None,
        "rows": 0,
        "chemical_rows": 0,
        "electrical_rows": 0,
        "warning": None,
    }
    try:
        canonical_edges = load_canonical_connectome_edgelist()
        edge_list_summary["rows"] = int(len(canonical_edges))
        if "synapse_type" in canonical_edges.columns:
            edge_list_summary["chemical_rows"] = int((canonical_edges["synapse_type"] == "chemical").sum())
            edge_list_summary["electrical_rows"] = int((canonical_edges["synapse_type"] == "electrical").sum())
    except Exception as exc:
        edge_list_summary["warning"] = str(exc)

    chem_pre, chem_post, chem_counts = _parse_cook_connectome_matrix(COOK2019_ADJ_XLSX, COOK_CHEM_SHEET)
    gap_pre, gap_post, gap_counts = _parse_cook_connectome_matrix(COOK2019_ADJ_XLSX, COOK_GAP_SHEET)
    coords = _parse_c302_coordinates(C302_GEOMETRY_NML)

    chem_pre_idx = _build_name_index(chem_pre)
    chem_post_idx = _build_name_index(chem_post)
    gap_pre_idx = _build_name_index(gap_pre)
    gap_post_idx = _build_name_index(gap_post)

    pathway_stats = {
        "ipsi_db": {"syn_sum": 0.0, "dist_sum": 0.0, "count": 0},
        "ipsi_vb": {"syn_sum": 0.0, "dist_sum": 0.0, "count": 0},
        "contra_db": {"syn_sum": 0.0, "dist_sum": 0.0, "count": 0},
        "contra_vb": {"syn_sum": 0.0, "dist_sum": 0.0, "count": 0},
        "next_db": {"syn_sum": 0.0, "dist_sum": 0.0, "count": 0},
        "next_vb": {"syn_sum": 0.0, "dist_sum": 0.0, "count": 0},
    }

    def _accumulate(pathway: str, syn_value: float, dist_value: Optional[float]) -> None:
        if syn_value <= 0 or dist_value is None or not np.isfinite(dist_value):
            return
        pathway_stats[pathway]["syn_sum"] += float(syn_value)
        pathway_stats[pathway]["dist_sum"] += float(dist_value)
        pathway_stats[pathway]["count"] += 1

    # 1) Ipsi pathways (chemical)
    for db in COOK_DB_NEURONS:
        for dorsal_m in COOK_DORSAL_MUSCLES:
            n_syn = _matrix_count(chem_counts, chem_pre_idx, chem_post_idx, db, dorsal_m)
            _accumulate("ipsi_db", n_syn, _distance_from_cook_cells(coords, db, dorsal_m))

    for vb in COOK_VB_NEURONS:
        for ventral_m in COOK_VENTRAL_MUSCLES:
            n_syn = _matrix_count(chem_counts, chem_pre_idx, chem_post_idx, vb, ventral_m)
            _accumulate("ipsi_vb", n_syn, _distance_from_cook_cells(coords, vb, ventral_m))

    # 2) Contra pathways (series conductance abstraction through D-neurons)
    for db in COOK_DB_NEURONS:
        for dd in COOK_DD_NEURONS:
            n_db_to_dd = _matrix_count(chem_counts, chem_pre_idx, chem_post_idx, db, dd)
            if n_db_to_dd <= 0:
                continue
            for ventral_m in COOK_VENTRAL_MUSCLES:
                n_dd_to_vm = _matrix_count(chem_counts, chem_pre_idx, chem_post_idx, dd, ventral_m)
                if n_dd_to_vm <= 0:
                    continue
                denom = n_db_to_dd + n_dd_to_vm
                n_eff = (n_db_to_dd * n_dd_to_vm) / denom if denom > 0 else 0.0
                _accumulate("contra_db", n_eff, _distance_from_cook_cells(coords, db, ventral_m))

    for vb in COOK_VB_NEURONS:
        for vd in COOK_VD_NEURONS:
            n_vb_to_vd = _matrix_count(chem_counts, chem_pre_idx, chem_post_idx, vb, vd)
            if n_vb_to_vd <= 0:
                continue
            for dorsal_m in COOK_DORSAL_MUSCLES:
                n_vd_to_dm = _matrix_count(chem_counts, chem_pre_idx, chem_post_idx, vd, dorsal_m)
                if n_vd_to_dm <= 0:
                    continue
                denom = n_vb_to_vd + n_vd_to_dm
                n_eff = (n_vb_to_vd * n_vd_to_dm) / denom if denom > 0 else 0.0
                _accumulate("contra_vb", n_eff, _distance_from_cook_cells(coords, vb, dorsal_m))

    # 3) Next-B pathways (gap junctions)
    for i in range(1, len(COOK_DB_NEURONS)):
        db_pre = f"DB{i:02d}"
        db_post = f"DB{i + 1:02d}"
        n_gap = _symmetric_gap_count(gap_counts, gap_pre_idx, gap_post_idx, db_pre, db_post)
        _accumulate("next_db", n_gap, _distance_from_cook_cells(coords, db_pre, db_post))

    for i in range(1, len(COOK_VB_NEURONS)):
        vb_pre = f"VB{i:02d}"
        vb_post = f"VB{i + 1:02d}"
        n_gap = _symmetric_gap_count(gap_counts, gap_pre_idx, gap_post_idx, vb_pre, vb_post)
        _accumulate("next_vb", n_gap, _distance_from_cook_cells(coords, vb_pre, vb_post))

    syn_avg: Dict[str, float] = {}
    dist_avg_raw: Dict[str, float] = {}
    for pathway, stat in pathway_stats.items():
        count = int(stat["count"])
        if count > 0:
            syn_avg[pathway] = float(stat["syn_sum"] / count)
            dist_avg_raw[pathway] = float(stat["dist_sum"] / count)
        else:
            syn_avg[pathway] = 0.0
            dist_avg_raw[pathway] = float("nan")

    finite_dists = [v for v in dist_avg_raw.values() if np.isfinite(v)]
    max_dist = max(finite_dists) if finite_dists else 1.0
    max_dist = max(max_dist, 1e-8)
    dist_norm = {
        pathway: (float(value / max_dist) if np.isfinite(value) else 1.0)
        for pathway, value in dist_avg_raw.items()
    }
    dist_norm = {k: float(np.clip(v, 0.0, 1.0)) for k, v in dist_norm.items()}
    missing_pathways = [k for k, v in dist_avg_raw.items() if not np.isfinite(v)]

    return {
        "num_segments": int(num_segments),
        "dist_ipsi_db": dist_norm["ipsi_db"],
        "dist_ipsi_vb": dist_norm["ipsi_vb"],
        "dist_contra_db": dist_norm["contra_db"],
        "dist_contra_vb": dist_norm["contra_vb"],
        "dist_next_db": dist_norm["next_db"],
        "dist_next_vb": dist_norm["next_vb"],
        "syn_ipsi_db": syn_avg["ipsi_db"],
        "syn_ipsi_vb": syn_avg["ipsi_vb"],
        "syn_contra_db": syn_avg["contra_db"],
        "syn_contra_vb": syn_avg["contra_vb"],
        "syn_next_db": syn_avg["next_db"],
        "syn_next_vb": syn_avg["next_vb"],
        "count_ipsi_db": int(pathway_stats["ipsi_db"]["count"]),
        "count_ipsi_vb": int(pathway_stats["ipsi_vb"]["count"]),
        "count_contra_db": int(pathway_stats["contra_db"]["count"]),
        "count_contra_vb": int(pathway_stats["contra_vb"]["count"]),
        "count_next_db": int(pathway_stats["next_db"]["count"]),
        "count_next_vb": int(pathway_stats["next_vb"]["count"]),
        "sources": {
            "cook2019_adjacency_xlsx": str(COOK2019_ADJ_XLSX),
            "cook2019_chemical_sheet": COOK_CHEM_SHEET,
            "cook2019_gap_sheet": COOK_GAP_SHEET,
            "c302_geometry_nml": str(C302_GEOMETRY_NML),
            "canonical_edge_list_csv": edge_list_summary["path"],
        },
        "translator_examples": {
            "DB01": _translate_cook_name_to_c302("DB01"),
            "VB01": _translate_cook_name_to_c302("VB01"),
            "DD01": _translate_cook_name_to_c302("DD01"),
            "VD01": _translate_cook_name_to_c302("VD01"),
            "dBWML1": _translate_cook_name_to_c302("dBWML1"),
            "vBWMR1": _translate_cook_name_to_c302("vBWMR1"),
        },
        "metadata": {
            "max_distance_mm_before_norm": float(max_dist),
            "missing_pathways": missing_pathways,
            "pathway_sources": {
                "ipsi_db": "chemical",
                "ipsi_vb": "chemical",
                "contra_db": "chemical",
                "contra_vb": "chemical",
                "next_db": "gap_junction",
                "next_vb": "gap_junction",
            },
            "edge_list_summary": edge_list_summary,
        },
    }
