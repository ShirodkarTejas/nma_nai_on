"""Utilities for loading and manipulating the C. elegans connectome."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


CONNECTOME_COLUMN_SYNONYMS = {
    "pre": ("pre", "source", "neuron1", "from"),
    "post": ("post", "target", "neuron2", "to"),
    "synapse_type": ("synapse_type", "type", "syn_type", "synapse"),
    "n_syn": ("n_syn", "count", "num", "synapses", "weight", "strength"),
}

SYNAPSE_TYPE_ALIASES = {
    "chem": "chemical",
    "chemical": "chemical",
    "elec": "electrical",
    "electrical": "electrical",
    "gap": "electrical",
    "gap_junction": "electrical",
    "gap_junctions": "electrical",
}

from .id_normalization import normalize_celegans_id, normalize_id_list

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent


def _connectome_data_dirs() -> list[Path]:
    candidates = [
        Path("/home/amin/Research/NCAP/external/ConnectomeToolbox/cect/data"),
        WORKSPACE_ROOT / "external" / "ConnectomeToolbox" / "cect" / "data",
        WORKSPACE_ROOT / "ConnectomeToolbox" / "cect" / "data",
        WORKSPACE_ROOT.parent / "external" / "ConnectomeToolbox" / "cect" / "data",
        WORKSPACE_ROOT.parent / "ConnectomeToolbox" / "cect" / "data",
    ]
    deduped: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in deduped:
            deduped.append(resolved)
    return deduped


CONNECTOME_DATA_DIRS = _connectome_data_dirs()
DEFAULT_CONNECTOME_DIR = next((p for p in CONNECTOME_DATA_DIRS if p.exists()), CONNECTOME_DATA_DIRS[0])
DEFAULT_CONNECTOME_PATHS = []
for data_dir in CONNECTOME_DATA_DIRS:
    DEFAULT_CONNECTOME_PATHS.extend(
        [
            data_dir / "herm_full_edgelist_MODIFIED.csv",
            data_dir / "herm_full_edgelist.csv",
        ]
    )
# Canonical IDs
DORSAL_B = [f"DB{str(i).zfill(2)}" for i in range(1, 8)]
VENTRAL_B = [f"VB{str(i).zfill(2)}" for i in range(1, 12)]
PROPRIO_IDS = ["DVA"]
REQUIRED_IDS = DORSAL_B + VENTRAL_B + PROPRIO_IDS


def _normalize_synapse_type_label(label: object) -> str:
    """Canonicalize a synapse label (e.g. chem->chemical, gap->electrical)."""
    key = str(label).strip().lower()
    return SYNAPSE_TYPE_ALIASES.get(key, key)


def _normalize_synapse_type_filter(labels: Iterable[str]) -> set[str]:
    """Canonicalize synapse-type filters into labels used in canonical DataFrames."""
    normalized: set[str] = set()
    for label in labels:
        if label is None:
            continue
        key = str(label).strip().lower()
        if not key:
            continue
        normalized.add(_normalize_synapse_type_label(key))
    return normalized


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common connectome columns to canonical names."""
    col_map = {c.lower(): c for c in df.columns}
    renamed = {}

    def first_match(options: Sequence[str]) -> Optional[str]:
        for opt in options:
            if opt in col_map:
                return col_map[opt]
        return None

    pre_col = first_match(CONNECTOME_COLUMN_SYNONYMS["pre"])
    post_col = first_match(CONNECTOME_COLUMN_SYNONYMS["post"])
    type_col = first_match(CONNECTOME_COLUMN_SYNONYMS["synapse_type"])
    count_col = first_match(CONNECTOME_COLUMN_SYNONYMS["n_syn"])

    for new_name, old_name in [
        ("pre", pre_col),
        ("post", post_col),
        ("synapse_type", type_col),
        ("n_syn", count_col),
    ]:
        if old_name:
            renamed[old_name] = new_name

    df = df.rename(columns=renamed)
    required = ["pre", "post", "n_syn"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' after normalization.")

    if "synapse_type" not in df.columns:
        df["synapse_type"] = "unknown"

    df["pre"] = df["pre"].astype(str).str.strip().apply(normalize_celegans_id)
    df["post"] = df["post"].astype(str).str.strip().apply(normalize_celegans_id)
    df["synapse_type"] = df["synapse_type"].astype(str).apply(_normalize_synapse_type_label)
    df["n_syn"] = pd.to_numeric(df["n_syn"], errors="coerce").fillna(0).astype(float)
    return df


def normalize_neuron_id(neuron_id: str) -> str:
    """Backward-compatible alias for normalize_celegans_id."""
    return normalize_celegans_id(neuron_id)


def load_connectome_edgelist(path: str, synapse_filter: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Load a C. elegans edgelist CSV.

    Args:
        path: Path to herm_full_edgelist.csv (or equivalent).
        synapse_filter: Optional iterable of synapse types to keep (case-insensitive).

    Returns:
        DataFrame with canonical columns: pre, post, synapse_type, n_syn.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Connectome CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _canonicalize_columns(df)

    if synapse_filter:
        allowed = _normalize_synapse_type_filter(synapse_filter)
        if not allowed:
            return df.iloc[0:0].copy()
        df = df[df["synapse_type"].isin(allowed)]

    return df


def _pick_canonical_edgelist() -> Optional[Path]:
    """Pick the canonical hermaphrodite edgelist shipped with ConnectomeToolbox."""
    candidates = [p for p in DEFAULT_CONNECTOME_PATHS if p.exists()]
    if not candidates:
        for data_dir in CONNECTOME_DATA_DIRS:
            if data_dir.exists():
                candidates.extend(sorted(data_dir.glob("herm_full_edgelist*.csv")))
    if candidates:
        deduped: list[Path] = []
        for path in candidates:
            if path not in deduped:
                deduped.append(path)
        candidates = deduped
    if not candidates:
        return None
    best_path = None
    best_rows = -1
    for path in candidates:
        try:
            rows = sum(1 for _ in Path(path).open())
        except Exception:
            rows = -1
        if rows > best_rows:
            best_rows = rows
            best_path = path
    return best_path


CANONICAL_EDGE_PATH = _pick_canonical_edgelist()


def load_canonical_connectome_edgelist(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the canonical C. elegans hermaphrodite connectome edgelist from ConnectomeToolbox.

    Uses the file that ships with the repo:
        ConnectomeToolbox/cect/data/herm_full_edgelist*.csv
    """
    chosen = Path(path) if path else CANONICAL_EDGE_PATH or _pick_canonical_edgelist()
    if chosen is None or not Path(chosen).exists():
        raise FileNotFoundError(f"Canonical connectome edgelist not found in {DEFAULT_CONNECTOME_DIR}")
    return load_connectome_edgelist(str(chosen))


def extract_edge_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Build an inventory of IDs present in the edgelist."""
    ids = set(df["pre"]).union(df["post"]) if not df.empty else set()
    norm_ids = {normalize_celegans_id(str(i)) for i in ids if pd.notna(i)}

    rows = []
    for nid in sorted(norm_ids):
        rows.append({"id": nid, "type": "neuron", "has_edge_data": True, "note": ""})

    for nid in REQUIRED_IDS:
        if nid not in norm_ids:
            rows.append({"id": nid, "type": "neuron", "has_edge_data": False, "note": "missing from edgelist"})
    return pd.DataFrame(rows)


def build_edge_init_priors(df: pd.DataFrame) -> Dict[str, Optional[torch.Tensor]]:
    """Build edge-based initialization priors (no geometry dependency)."""
    priors: Dict[str, Optional[torch.Tensor]] = {
        "A_bb": None,
        "A_bm_ipsi": None,
        "A_bm_contra": None,
        "A_prop_to_b": None,
    }

    # Segment B-neuron adjacency.
    seg = build_segmented_bneuron_muscle_adjacency(df)
    if seg["b_neurons"]:
        priors["A_bb"] = adjacency_to_torch(seg["A_bb"])
    # NMJ counts likely absent; leave bm matrices None unless muscles present.
    if seg["muscles"]:
        priors["A_bm_ipsi"] = adjacency_to_torch(seg["A_bm_ipsi"])
        priors["A_bm_contra"] = adjacency_to_torch(seg["A_bm_contra"])

    # Proprioceptive DVA -> B neurons.
    b_set = set(seg["b_neurons"])
    if b_set and "DVA" in set(df["pre"]):
        A, neurons = build_adjacency(df, synapse_types=("chemical",))
        idx = {n: i for i, n in enumerate(neurons)}
        prop_vec = []
        for b in seg["b_neurons"]:
            prop_vec.append(A[idx["DVA"], idx[b]] if "DVA" in idx and b in idx else 0.0)
        priors["A_prop_to_b"] = adjacency_to_torch(np.array(prop_vec, dtype=np.float32)[None, :])

    return priors


def _sum_synapses(
    df: pd.DataFrame,
    pre: str,
    post: str,
    synapse_types: Optional[Iterable[str]] = None,
) -> tuple[float, bool]:
    """Sum synapse counts for a given directed pair."""
    try:
        subset = df[(df["pre"] == pre) & (df["post"] == post)]
    except KeyError:
        return 0.0, False
    if synapse_types:
        allowed = _normalize_synapse_type_filter(synapse_types)
        if not allowed:
            return 0.0, False
        if "synapse_type" not in subset.columns:
            return 0.0, False
        subset = subset[subset["synapse_type"].isin(allowed)]
    if subset.empty:
        return 0.0, False
    return float(subset["n_syn"].sum()), True


def build_init_priors_from_edges(
    df: pd.DataFrame,
    n_joints: int = 6,
    prop_synapse_types: Tuple[str, ...] = ("chemical",),
    next_b_synapse_types: Tuple[str, ...] = ("electrical",),
) -> Dict[str, object]:
    """Build initialization priors with masks from an edgelist.

    By default:
      - DVA->B uses chemical synapses.
      - B_i->B_{i+1} uses electrical synapses (gap junctions).
    """
    dorsal_ids = normalize_id_list(DORSAL_B[: n_joints + 1])
    ventral_ids = normalize_id_list(VENTRAL_B[: n_joints + 1])

    prop_src = [normalize_celegans_id("DVA")]

    def _pairwise(ids: List[str]) -> List[Tuple[str, str]]:
        return [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]

    pairs_d = _pairwise(dorsal_ids)
    pairs_v = _pairwise(ventral_ids)

    n_syn_d, mask_d = [], []
    for dst in dorsal_ids:
        val, ok = _sum_synapses(df, prop_src[0], dst, synapse_types=prop_synapse_types)
        n_syn_d.append(val)
        mask_d.append(ok)

    n_syn_v, mask_v = [], []
    for dst in ventral_ids:
        val, ok = _sum_synapses(df, prop_src[0], dst, synapse_types=prop_synapse_types)
        n_syn_v.append(val)
        mask_v.append(ok)

    bb_n_syn_d, bb_mask_d = [], []
    for src, dst in pairs_d:
        val, ok = _sum_synapses(df, src, dst, synapse_types=next_b_synapse_types)
        bb_n_syn_d.append(val)
        bb_mask_d.append(ok)

    bb_n_syn_v, bb_mask_v = [], []
    for src, dst in pairs_v:
        val, ok = _sum_synapses(df, src, dst, synapse_types=next_b_synapse_types)
        bb_n_syn_v.append(val)
        bb_mask_v.append(ok)

    return {
        "B": {"dorsal_ids": dorsal_ids, "ventral_ids": ventral_ids},
        "prop_to_B": {
            "src_ids": prop_src,
            "dst_ids_d": dorsal_ids,
            "dst_ids_v": ventral_ids,
            "n_syn_d": torch.as_tensor(n_syn_d, dtype=torch.float32),
            "mask_d": torch.as_tensor(mask_d, dtype=torch.bool),
            "n_syn_v": torch.as_tensor(n_syn_v, dtype=torch.float32),
            "mask_v": torch.as_tensor(mask_v, dtype=torch.bool),
        },
        "B_to_B_next": {
            "pairs_d": pairs_d,
            "pairs_v": pairs_v,
            "n_syn_d": torch.as_tensor(bb_n_syn_d, dtype=torch.float32),
            "mask_d": torch.as_tensor(bb_mask_d, dtype=torch.bool),
            "n_syn_v": torch.as_tensor(bb_n_syn_v, dtype=torch.float32),
            "mask_v": torch.as_tensor(bb_mask_v, dtype=torch.bool),
        },
        "B_to_muscle": None,
    }


def _score_connectome_candidate(path: Path, nrows: int = 100) -> int:
    """Heuristically score whether a CSV/TSV looks like a connectome edgelist."""
    try:
        df = pd.read_csv(path, nrows=nrows, sep=None, engine="python")
    except Exception:
        return -1
    cols = {c.lower() for c in df.columns}
    score = 0
    for syns in CONNECTOME_COLUMN_SYNONYMS.values():
        if any(s in cols for s in syns):
            score += 1
    return score


def discover_connectome_file(
    preferred_path: Optional[str] = None,
    search_roots: Optional[Iterable[str]] = None,
    min_score: int = 2,
) -> Optional[Path]:
    """Find a connectome edgelist CSV/TSV.

    Primary: explicit ConnectomeToolbox files shipped in this workspace.
    Fallback: heuristic search.
    """
    candidates: List[Path] = []

    if preferred_path:
        p = Path(preferred_path)
        if p.exists():
            candidates.append(p)

    # Prioritize known paths.
    for p in DEFAULT_CONNECTOME_PATHS:
        if p.exists():
            candidates.append(p)

    # Heuristic fallback if needed.
    if not candidates:
        if search_roots is None:
            search_roots = [
                str(WORKSPACE_ROOT / "external" / "ConnectomeToolbox"),
                str(WORKSPACE_ROOT.parent / "external" / "ConnectomeToolbox"),
                "ConnectomeToolbox",
                "connectome_toolbox",
                "../ConnectomeToolbox",
                "CElegansNeuroML",
                "../CElegansNeuroML",
            ]
        for root in search_roots:
            root_path = Path(root)
            if not root_path.exists():
                continue
            for ext in ("*.csv", "*.tsv"):
                candidates.extend(root_path.rglob(ext))

    best_path: Optional[Path] = None
    best_score = -1
    for path in candidates:
        score = _score_connectome_candidate(path)
        if score > best_score and score >= min_score:
            best_path = path
            best_score = score

    if best_path:
        try:
            df = pd.read_csv(best_path)
            print(f"Using C. elegans edgelist from: {best_path.resolve()} with {len(df)} rows")
        except Exception:
            pass

    return best_path


def build_adjacency(df: pd.DataFrame, synapse_types: Tuple[str, ...] = ("chemical",)) -> tuple[np.ndarray, list[str]]:
    """Build adjacency matrix from an edgelist.

    Args:
        df: DataFrame with columns pre, post, synapse_type, n_syn.
        synapse_types: Synapse types to include (default: chemical synapses only).

    Returns:
        (A, neurons) where A[i, j] is synapse count from neurons[i] to neurons[j].
    """
    if synapse_types:
        allowed = _normalize_synapse_type_filter(synapse_types)
        if not allowed:
            df = df.iloc[0:0]
        else:
            df = df[df["synapse_type"].isin(allowed)]

    neurons = sorted(set(df["pre"]).union(df["post"]))
    idx = {n: i for i, n in enumerate(neurons)}
    A = np.zeros((len(neurons), len(neurons)), dtype=np.float32)

    for row in df.itertuples(index=False):
        pre = getattr(row, "pre")
        post = getattr(row, "post")
        n_syn = getattr(row, "n_syn", 1)
        A[idx[pre], idx[post]] += float(n_syn)

    return A, neurons


def build_segmented_bneuron_muscle_adjacency(
    df: pd.DataFrame,
    neuron_name_pattern: str = r"^[VD][AB]|^B[VD]",  # DB01, VB02 style, permissive for VA/VB/VB/DB
    muscle_name_pattern: str = r"^M[VD]",  # body-wall muscles MV?, MD?, etc.
) -> Dict[str, object]:
    """Build adjacency focused on B-neurons and body-wall muscles.

    Returns:
        {
            'b_neurons': list[str],
            'muscles': list[str],
            'A_bb': np.ndarray,
            'A_bm_ipsi': np.ndarray,
            'A_bm_contra': np.ndarray,
            'A_mb': np.ndarray,
            'A_mm': np.ndarray,
        }
    """
    b_re = re.compile(neuron_name_pattern, flags=re.IGNORECASE)
    m_re = re.compile(muscle_name_pattern, flags=re.IGNORECASE)

    all_neurons = set(df["pre"]).union(df["post"])
    b_neurons = sorted([n for n in all_neurons if isinstance(n, str) and b_re.match(n)])
    muscles = sorted([n for n in all_neurons if isinstance(n, str) and m_re.match(n)])

    b_idx = {n: i for i, n in enumerate(b_neurons)}
    m_idx = {n: i for i, n in enumerate(muscles)}

    A_bb = np.zeros((len(b_neurons), len(b_neurons)), dtype=np.float32)
    A_bm_ipsi = np.zeros((len(b_neurons), len(muscles)), dtype=np.float32)
    A_bm_contra = np.zeros_like(A_bm_ipsi)
    A_mb = np.zeros((len(muscles), len(b_neurons)), dtype=np.float32)
    A_mm = np.zeros((len(muscles), len(muscles)), dtype=np.float32)

    def _side(name: str) -> Optional[str]:
        n = name.upper()
        if "D" in n:
            return "dorsal"
        if "V" in n:
            return "ventral"
        return None

    for row in df.itertuples(index=False):
        pre = getattr(row, "pre")
        post = getattr(row, "post")
        n_syn = float(getattr(row, "n_syn", 1))

        pre_side = _side(str(pre))
        post_side = _side(str(post))

        if pre in b_idx and post in b_idx:
            A_bb[b_idx[pre], b_idx[post]] += n_syn
        elif pre in b_idx and post in m_idx:
            if pre_side and post_side and pre_side == post_side:
                A_bm_ipsi[b_idx[pre], m_idx[post]] += n_syn
            else:
                A_bm_contra[b_idx[pre], m_idx[post]] += n_syn
        elif pre in m_idx and post in b_idx:
            A_mb[m_idx[pre], b_idx[post]] += n_syn
        elif pre in m_idx and post in m_idx:
            A_mm[m_idx[pre], m_idx[post]] += n_syn

    return {
        "b_neurons": b_neurons,
        "muscles": muscles,
        "A_bb": A_bb,
        "A_bm_ipsi": A_bm_ipsi,
        "A_bm_contra": A_bm_contra,
        "A_mb": A_mb,
        "A_mm": A_mm,
    }


def map_neurons_to_segments(neuron_names: List[str], segment_regex: str = r"(\\d{1,2})") -> Dict[str, int]:
    """Assign neurons to body segments by parsing trailing numbers."""
    mapping: Dict[str, int] = {}
    pattern = re.compile(segment_regex)
    for name in neuron_names:
        match = pattern.search(name)
        if match:
            try:
                mapping[name] = int(match.group(1))
            except ValueError:
                continue
    return mapping


def aggregate_by_segment(A: np.ndarray, neurons: List[str], mapping: Dict[str, int]) -> np.ndarray:
    """Aggregate neuron-level adjacency into segment-level adjacency."""
    if not mapping:
        return np.zeros((0, 0), dtype=A.dtype)
    segments = sorted(set(mapping.values()))
    seg_to_idx = {s: i for i, s in enumerate(segments)}
    S = np.zeros((len(segments), len(segments)), dtype=A.dtype)
    idx = {n: i for i, n in enumerate(neurons)}
    for pre, s_pre in mapping.items():
        for post, s_post in mapping.items():
            if pre in idx and post in idx:
                S[seg_to_idx[s_pre], seg_to_idx[s_post]] += A[idx[pre], idx[post]]
    return S


def normalize_adjacency(A: np.ndarray, mode: str = "log1p", per_pre: bool = True) -> np.ndarray:
    """Normalize adjacency matrix for use as priors."""
    if mode == "none" or mode is None:
        return A
    if mode == "log1p":
        return np.log1p(A)
    if mode == "row":
        if per_pre:
            max_per_row = A.max(axis=1, keepdims=True)
            max_per_row[max_per_row == 0] = 1.0
            return A / max_per_row
        max_val = A.max()
        return A / max_val if max_val > 0 else A
    raise ValueError(f"Unknown normalization mode: {mode}")


def adjacency_to_torch(A: np.ndarray, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert adjacency matrix to a torch Tensor."""
    return torch.as_tensor(A, dtype=dtype, device=device)
