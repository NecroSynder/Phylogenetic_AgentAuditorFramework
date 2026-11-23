"""
Shared environment discovery, WSL2 LM Studio server resolution, canonical dataset
path management, and mtime-keyed in-memory file caching to eliminate redundant disk I/O
across large multi-taxon trees (N <= 200 taxa).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio import AlignIO
from ete3 import Tree


def resolve_conda_binary(binary_name: str) -> str:
    """Locates a compiled bioinformatics binary (e.g., iqtree2, raxml-ng, fasttree, fastme) in the active Conda env."""
    env_bin_dir = str(Path(sys.executable).parent)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    search_paths = [env_bin_dir]
    if conda_prefix:
        search_paths.append(str(Path(conda_prefix) / "bin"))
    for sp in search_paths:
        found = shutil.which(binary_name, path=sp)
        if found:
            return found
    return shutil.which(binary_name) or binary_name


def default_cpu_threads(max_cap: int = 8) -> int:
    """Returns a safe default thread/worker count bounded by available CPU cores and max_cap."""
    cores = os.cpu_count() or 2
    return max(1, min(cores, max_cap))


def detect_wsl_host_ip() -> str | None:
    """Detects the Windows host IP when running inside WSL2 (/etc/resolv.conf)."""
    try:
        resolv_path = Path("/etc/resolv.conf")
        if resolv_path.exists():
            for line in resolv_path.read_text().splitlines():
                if line.strip().startswith("nameserver"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1].strip()
    except Exception:
        pass
    return None


def resolve_lm_studio_url(base_url: str, timeout: float = 3.0) -> tuple[str, list[str]]:
    """
    Verifies connectivity to a local LM Studio OpenAI-compatible server via GET /models,
    automatically trying localhost, 127.0.0.1, and the WSL2 Windows host IP if needed.
    Returns (resolved_base_url, loaded_model_ids).
    """
    clean_base = base_url.rstrip("/")
    candidates = [clean_base]
    wsl_ip = detect_wsl_host_ip()
    if wsl_ip and "localhost" in clean_base:
        candidates.append(clean_base.replace("localhost", wsl_ip))
    if "localhost" in clean_base:
        candidates.append(clean_base.replace("localhost", "127.0.0.1"))

    last_err: Exception | None = None
    for cand_url in candidates:
        models_url = f"{cand_url}/models"
        try:
            req = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [str(m.get("id", "")) for m in data.get("data", []) if m.get("id")]
                return cand_url, models
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Could not reach LM Studio server at {candidates}: {last_err}")


def sort_node_id_key(node_id: str) -> tuple[int, str]:
    """Deterministic post-order node sort key (N1, N2, ..., Nk)."""
    if node_id.startswith("N") and node_id[1:].isdigit():
        return (int(node_id[1:]), node_id)
    return (999999, node_id)


def detect_alignment_format(alignment_path: str | Path) -> str:
    """Infers Bio.AlignIO schema ('fasta' or 'phylip-relaxed') from file extension/header."""
    p = Path(alignment_path)
    ext = p.suffix.lower()
    if ext in {".phy", ".phylip", ".aln"}:
        return "phylip-relaxed"
    return "fasta"


def _file_cache_key(file_path: str | Path) -> tuple[str, int, int]:
    p = Path(file_path).resolve()
    st = p.stat()
    return (str(p), st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=64)
def _cached_alignment_data(resolved_path: str, mtime_ns: int, size: int) -> tuple[tuple[str, ...], np.ndarray, int]:
    fmt = detect_alignment_format(resolved_path)
    alignment = AlignIO.read(resolved_path, fmt)
    taxa_ids = tuple(rec.id for rec in alignment)
    seq_len = alignment.get_alignment_length()
    char_mat = np.array([list(str(rec.seq).upper()) for rec in alignment], dtype="<U1")
    char_mat.setflags(write=False)
    return taxa_ids, char_mat, seq_len


def load_alignment_data(fasta_path: str | Path) -> tuple[tuple[str, ...], np.ndarray, int]:
    """Returns cached (taxa_ids_tuple, uppercase_char_matrix [N_taxa x L], seq_len)."""
    key = _file_cache_key(fasta_path)
    return _cached_alignment_data(*key)


@lru_cache(maxsize=64)
def _cached_m_true_df(resolved_path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    df = pd.read_csv(resolved_path, sep="\t", keep_default_na=False)
    if "is_indel" in df.columns:
        df["is_indel"] = df["is_indel"].apply(
            lambda x: str(x).lower() == "true" if not isinstance(x, bool) else x
        )
    if "site" in df.columns and not df.empty:
        df["site"] = df["site"].astype(int)
    for float_col in ("posterior_prob", "delta_sls", "ci_score"):
        if float_col in df.columns and not df.empty:
            df[float_col] = df[float_col].astype(float)
    return df


def load_m_true_df(asr_state_path: str | Path) -> pd.DataFrame:
    """Returns a cached DataFrame for M_true.tsv invalidated automatically on file modification."""
    key = _file_cache_key(asr_state_path)
    return _cached_m_true_df(*key)


@lru_cache(maxsize=128)
def _cached_json_dict(resolved_path: str, mtime_ns: int, size: int) -> dict[str, Any]:
    with open(resolved_path) as f:
        return json.load(f)


def load_json_dict(json_path: str | Path) -> dict[str, Any]:
    """Returns cached parsed JSON dictionary (read-only reference; invalidated on file change)."""
    key = _file_cache_key(json_path)
    return _cached_json_dict(*key)


@lru_cache(maxsize=128)
def _cached_newick_str(resolved_path: str, mtime_ns: int, size: int) -> str:
    return Path(resolved_path).read_text().strip()


def load_ete_tree(newick_path: str | Path, fmt: int = 1) -> Tree:
    """Loads an ETE3 Tree from cached Newick text so callers get a fresh Tree instance without disk I/O."""
    key = _file_cache_key(newick_path)
    nwk_str = _cached_newick_str(*key)
    return Tree(nwk_str, format=fmt)


def load_dataset_metadata(fasta_path: str | Path) -> dict[str, Any]:
    """
    Loads per-dataset empirical or synthetic metadata if present alongside fasta_path.
    Checks dataset_metadata.json first (empirical Tiers 1-3), then simulation_metadata.json.
    """
    parent = Path(fasta_path).resolve().parent
    for fname in ("dataset_metadata.json", "simulation_metadata.json"):
        cand = parent / fname
        if cand.exists():
            return dict(load_json_dict(cand))
    return {}


def infer_tier_group_from_path(fasta_path: str | Path) -> str:
    """Infers canonical tier subdirectory ('tier1', 'tier2', 'tier3') from an alignment path if present."""
    p = Path(fasta_path).resolve()
    for part in reversed(p.parts[:-1]):
        low = part.lower()
        if low in {"tier1", "tier2", "tier3"}:
            return low
    return ""


@dataclass(frozen=True)
class DatasetPaths:
    """Canonical workspace directory layout for a dataset & target model."""
    dataset_id: str
    target_model: str = "Fusang"
    tier_group: str = ""
    root_dir: Path = Path("data")

    @property
    def ground_truth_dir(self) -> Path:
        if self.tier_group:
            return self.root_dir / "02_ground_truths" / self.tier_group / self.dataset_id
        return self.root_dir / "02_ground_truths" / self.dataset_id

    @property
    def standardized_prefix(self) -> Path:
        if self.tier_group:
            return self.root_dir / "03_standardized_trees" / self.tier_group / self.dataset_id / self.target_model
        return self.root_dir / "03_standardized_trees" / self.dataset_id / self.target_model

    @property
    def normalized_newick(self) -> Path:
        return Path(f"{self.standardized_prefix}_normalized.nwk")

    @property
    def normalized_metadata(self) -> Path:
        return Path(f"{self.standardized_prefix}_metadata.json")
