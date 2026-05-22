"""Notebook helpers for visualizing the exact sparse priors used in training."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .swimmer_priors import (
    C302_GEOMETRY_NML,
    COOK2019_ADJ_XLSX,
    COOK_CHEM_SHEET,
    COOK_DB_NEURONS,
    COOK_DD_NEURONS,
    COOK_DORSAL_MUSCLES,
    COOK_GAP_SHEET,
    COOK_VB_NEURONS,
    COOK_VD_NEURONS,
    COOK_VENTRAL_MUSCLES,
    _build_name_index,
    _distance_from_cook_cells,
    _matrix_count,
    _parse_c302_coordinates,
    _parse_cook_connectome_matrix,
    _symmetric_gap_count,
    _translate_cook_name_to_c302,
    generate_ncap_segment_priors,
)


PATHWAY_ORDER = (
    "ipsi_db",
    "ipsi_vb",
    "contra_db",
    "contra_vb",
    "next_db",
    "next_vb",
)

PATHWAY_COLORS = {
    "ipsi_db": "#1f77b4",
    "ipsi_vb": "#17becf",
    "contra_db": "#d62728",
    "contra_vb": "#ff7f0e",
    "next_db": "#2ca02c",
    "next_vb": "#9467bd",
}

PATHWAY_SOURCES = {
    "ipsi_db": "chemical",
    "ipsi_vb": "chemical",
    "contra_db": "chemical",
    "contra_vb": "chemical",
    "next_db": "gap_junction",
    "next_vb": "gap_junction",
}


def _node_class(node_name: str) -> str:
    name = str(node_name)
    if name.startswith("DB"):
        return "DB"
    if name.startswith("VB"):
        return "VB"
    if name.startswith("DD"):
        return "DD"
    if name.startswith("VD"):
        return "VD"
    if name.startswith("dBWM"):
        return "dorsal_muscle"
    if name.startswith("vBWM"):
        return "ventral_muscle"
    return "other"


def node_class_colors() -> Dict[str, str]:
    """Default node color palette for motor-circuit network plots."""
    return {
        "DB": "#4e79a7",
        "VB": "#76b7b2",
        "DD": "#e15759",
        "VD": "#f28e2b",
        "dorsal_muscle": "#59a14f",
        "ventral_muscle": "#b07aa1",
        "other": "#7f7f7f",
    }


def _iter_training_samples() -> List[dict]:
    """Reconstruct per-sample pathway contributions using the same logic as training priors."""
    chem_pre, chem_post, chem_counts = _parse_cook_connectome_matrix(COOK2019_ADJ_XLSX, COOK_CHEM_SHEET)
    gap_pre, gap_post, gap_counts = _parse_cook_connectome_matrix(COOK2019_ADJ_XLSX, COOK_GAP_SHEET)
    coords = _parse_c302_coordinates(C302_GEOMETRY_NML)

    chem_pre_idx = _build_name_index(chem_pre)
    chem_post_idx = _build_name_index(chem_post)
    gap_pre_idx = _build_name_index(gap_pre)
    gap_post_idx = _build_name_index(gap_post)

    rows: List[dict] = []

    def add_row(
        pathway: str,
        syn_value: float,
        dist_value: float | None,
        pre: str,
        post: str,
        mid: str | None = None,
    ) -> None:
        if syn_value <= 0:
            return
        rows.append(
            {
                "pathway": pathway,
                "pre": pre,
                "mid": mid,
                "post": post,
                "syn": float(syn_value),
                "dist_um": float(dist_value) if dist_value is not None and np.isfinite(dist_value) else np.nan,
                "source": PATHWAY_SOURCES[pathway],
            }
        )

    for db in COOK_DB_NEURONS:
        for dorsal_m in COOK_DORSAL_MUSCLES:
            n_syn = _matrix_count(chem_counts, chem_pre_idx, chem_post_idx, db, dorsal_m)
            dist = _distance_from_cook_cells(coords, db, dorsal_m)
            add_row("ipsi_db", n_syn, dist, pre=db, post=dorsal_m)

    for vb in COOK_VB_NEURONS:
        for ventral_m in COOK_VENTRAL_MUSCLES:
            n_syn = _matrix_count(chem_counts, chem_pre_idx, chem_post_idx, vb, ventral_m)
            dist = _distance_from_cook_cells(coords, vb, ventral_m)
            add_row("ipsi_vb", n_syn, dist, pre=vb, post=ventral_m)

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
                dist = _distance_from_cook_cells(coords, db, ventral_m)
                add_row("contra_db", n_eff, dist, pre=db, mid=dd, post=ventral_m)

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
                dist = _distance_from_cook_cells(coords, vb, dorsal_m)
                add_row("contra_vb", n_eff, dist, pre=vb, mid=vd, post=dorsal_m)

    for i in range(1, len(COOK_DB_NEURONS)):
        db_pre = f"DB{i:02d}"
        db_post = f"DB{i + 1:02d}"
        n_gap = _symmetric_gap_count(gap_counts, gap_pre_idx, gap_post_idx, db_pre, db_post)
        dist = _distance_from_cook_cells(coords, db_pre, db_post)
        add_row("next_db", n_gap, dist, pre=db_pre, post=db_post)

    for i in range(1, len(COOK_VB_NEURONS)):
        vb_pre = f"VB{i:02d}"
        vb_post = f"VB{i + 1:02d}"
        n_gap = _symmetric_gap_count(gap_counts, gap_pre_idx, gap_post_idx, vb_pre, vb_post)
        dist = _distance_from_cook_cells(coords, vb_pre, vb_post)
        add_row("next_vb", n_gap, dist, pre=vb_pre, post=vb_post)

    return rows


def collect_training_pathway_tables(num_segments: int = 8) -> Dict[str, object]:
    """Return sample-level and summary-level tables for training priors."""
    priors = generate_ncap_segment_priors(num_segments=int(num_segments))
    sample_df = pd.DataFrame(_iter_training_samples())
    if sample_df.empty:
        sample_df = pd.DataFrame(columns=["pathway", "pre", "mid", "post", "syn", "dist_um", "source"])

    rows = []
    for pathway in PATHWAY_ORDER:
        sdf = sample_df[sample_df["pathway"] == pathway]
        raw_count = int(len(sdf))
        used_df = sdf[np.isfinite(sdf["dist_um"].to_numpy())] if raw_count else sdf
        used_count = int(len(used_df))
        if used_count:
            syn_sum = float(used_df["syn"].sum())
            syn_mean = float(used_df["syn"].mean())
            dist_mean = float(used_df["dist_um"].mean())
            dist_median = float(used_df["dist_um"].median())
            dist_std = float(used_df["dist_um"].std(ddof=0))
        else:
            syn_sum = 0.0
            syn_mean = 0.0
            dist_mean = float("nan")
            dist_median = float("nan")
            dist_std = float("nan")

        rows.append(
            {
                "pathway": pathway,
                "source": PATHWAY_SOURCES[pathway],
                "raw_edge_count_from_connectome": raw_count,
                "used_edge_count_with_geometry": used_count,
                "syn_sum_from_samples": syn_sum,
                "syn_mean_from_samples": syn_mean,
                "dist_mean_um": dist_mean,
                "dist_median_um": dist_median,
                "dist_std_um": dist_std,
                "count_used_in_training": int(priors.get(f"count_{pathway}", 0)),
                "syn_used_in_training": float(priors.get(f"syn_{pathway}", 0.0)),
                "dist_norm_used_in_training": float(priors.get(f"dist_{pathway}", 1.0)),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df["count_match_training"] = (
        summary_df["used_edge_count_with_geometry"] == summary_df["count_used_in_training"]
    )
    return {
        "samples": sample_df,
        "summary": summary_df,
        "priors": priors,
    }


def top_edges_by_pathway(samples_df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """Select top-k strongest edges per pathway for readable network plots."""
    if samples_df.empty:
        return samples_df.copy()
    pieces = []
    for pathway in PATHWAY_ORDER:
        sdf = samples_df[samples_df["pathway"] == pathway]
        if sdf.empty:
            continue
        pieces.append(sdf.sort_values("syn", ascending=False).head(int(top_k)))
    return pd.concat(pieces, ignore_index=True) if pieces else samples_df.iloc[0:0].copy()


def cook_positions_for_nodes(nodes: Iterable[str], use_xz: bool = True) -> Dict[str, tuple[float, float]]:
    """Map Cook IDs to 2D positions from c302 geometry for network plotting."""
    coords = _parse_c302_coordinates(C302_GEOMETRY_NML)
    positions: Dict[str, tuple[float, float]] = {}
    for node in nodes:
        c302_name = _translate_cook_name_to_c302(str(node))
        xyz = coords.get(c302_name) if c302_name else None
        if xyz is None or not np.isfinite(xyz).all():
            continue
        x = float(xyz[0])
        y = float(xyz[2] if use_xz else xyz[1])
        positions[str(node)] = (x, y)
    return positions


def node_class(node_name: str) -> str:
    """Public wrapper for notebook node typing."""
    return _node_class(node_name)

