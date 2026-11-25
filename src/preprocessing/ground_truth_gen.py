"""
================================================================================
PIPELINE STEP 1: GROUND-TRUTH & EVOLUTIONARY FEATURE EXTRACTION
(Manuscript Section 4.1.1, Items 1–3 & Table 4.1, Pages 8–16)
================================================================================
Establishes the three deterministic mathematical ground truths required to audit
both target deep-learning models (Fusang, Phyloformer) and the Diagnostic Agent:

  1. Topological Ground Truth (T_true -> T_true_rooted.nwk):
     - Universal outgroup rooting via ETE 3 (tree.set_outgroup(outgroup))
     - Deterministic post-order internal node indexing (N1, N2, ..., Nk)

  2. Site-Level Causal Mutation Ground Truth (M_true -> M_true.tsv):
     - Item 2.1: Maximum-Likelihood Ancestral Sequence Reconstruction (ASR) via
       IQ-TREE 2 (--ancestral -wsl) for both DNA (GTR+G4) and Protein (LG+G4),
       retaining branch transitions (s_u -> s_v) with marginal posterior P >= 0.90
       and positive Felsenstein site log-likelihood difference delta_SLS_i(b) > 0
       computed from IQ-TREE .sitelh across branch-collapsed topologies T_{-b}.
     - Item 2.2: Deterministic Indel & Synapomorphy Mapping via two-pass Fitch
       Maximum Parsimony over binary gap-coded matrices (0 = character, 1 = gap '-')
       and Shannon Mutual Information I(X_i; b_k) in bits (Eq. 4.12 / Page 26).

  3. Homoplasy & Evolutionary Stressor Ground Truth (A_true -> A_true.json):
     - Vectorized bit-parallel Fitch parsimony Consistency Index (CI_i = m_i / s_i)
       and Retention Index (RI_i = (g_i - s_i) / (g_i - m_i)), reporting both
       all-site and informative-site means (preventing invariant-site CI dilution).
     - Felsenstein-zone Long-Branch Attraction metrics (global_rho_lba, per-leaf
       branch lengths, and identified long_branch_taxa).
     - Contiguous gap tract boundaries [i_start, i_end], cell gap density, and
       gapped_column_pct.
     - Localized recombination breakpoints [c_start, c_end] (simulated or detected
       via 200-bp sliding-window topological discordance; Table 4.1 Row 4).
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from ete3 import Tree

from src.utils.io_cache import (
    default_cpu_threads,
    load_alignment_data,
    resolve_conda_binary,
)

AMBIGUOUS_CHARS = {"-", "?", "N", "X"}


# ==============================================================================
# SECTION 1: ALPHABET DETECTION (DNA vs. PROTEIN)
# ==============================================================================

def detect_sequence_alphabet(fasta_path: str) -> str:
    """
    Automatically detects whether an alignment is Nucleotide ('DNA') or Amino Acid ('AA').
    """
    _, char_mat, _ = load_alignment_data(fasta_path)
    sample_rows = char_mat[: min(10, char_mat.shape[0])]
    sample_chars = set(np.unique(sample_rows).tolist()) - AMBIGUOUS_CHARS
    dna_alphabet = {"A", "C", "G", "T", "U", "R", "Y", "S", "W", "K", "M", "B", "D", "H", "V"}
    if sample_chars and sample_chars.issubset(dna_alphabet):
        return "DNA"
    return "AA"


# ==============================================================================
# SECTION 1B: FLEXIBLE NEWICK LOADING & IQ-TREE MODEL PARSING
# ==============================================================================

def load_published_tree_safely(raw_text_or_path: str | Path) -> tuple[Tree, bool]:
    """
    Loads a reference Newick tree from either a file path or a Newick string,
    stripping NEXUS/NHX bracket comments ([&R], [&...]), trying ETE3 formats (1, 0, 2, 5),
    stripping surrounding quotes on taxon names, and detecting whether explicit branch
    lengths (':<float>') were present in the input.
    """
    raw_str = str(raw_text_or_path)
    p_cand = Path(raw_str)
    if "\n" not in raw_str and "(" not in raw_str and p_cand.exists():
        raw_text = p_cand.read_text()
    else:
        raw_text = raw_str

    cleaned = re.sub(r"\[[^\]]*\]", "", raw_text).strip()
    has_explicit_lengths = ":" in cleaned
    for quoted in (True, False):
        for fmt in (1, 0, 2, 5):
            try:
                t = Tree(cleaned, format=fmt, quoted_node_names=quoted)
                for lf in t.iter_leaves():
                    lf.name = lf.name.strip("'\"")
                if has_explicit_lengths:
                    non_root_dists = [float(n.dist) for n in t.traverse() if not n.is_root()]
                    if non_root_dists and all(abs(d - 1.0) < 1e-9 for d in non_root_dists):
                        has_explicit_lengths = False
                return t, has_explicit_lengths
            except Exception:
                continue
    raise ValueError(f"Unable to parse Newick tree: {str(raw_text_or_path)[:120]}")


def parse_iqtree_best_model(iqtree_report_path: str | Path) -> str | None:
    """Extracts the ModelFinder best-fit substitution model from an IQ-TREE .iqtree report."""
    p = Path(iqtree_report_path)
    if not p.exists():
        return None
    text = p.read_text(errors="replace")
    for pattern in (
        r"Best-fit model according to \w+:\s*(\S+)",
        r"Model of substitution:\s*(\S+)",
    ):
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return None


def _transfer_ml_branch_lengths_to_rooted_tree(
    rooted_tree: Tree,
    ml_treefile_path: Path,
    outgroup: str,
) -> None:
    """
    Transfers ML-optimized branch lengths from IQ-TREE's .treefile onto a branchless
    outgroup-rooted reference pedigree (preserving exact rooted topology and N1..Nk labels).
    """
    if not ml_treefile_path.exists():
        return
    t_ml, _ = load_published_tree_safely(ml_treefile_path)
    all_leaves = set(t_ml.get_leaf_names())

    leaf_dists = {lf.name: max(float(lf.dist), 1e-6) for lf in t_ml.iter_leaves()}
    split_dists: dict[frozenset[str], float] = {}
    for node in t_ml.traverse():
        if node.is_root() or node.is_leaf():
            continue
        canon = _outgroup_excluding_split(set(node.get_leaf_names()), all_leaves, outgroup)
        if canon:
            split_dists[canon] = max(float(node.dist), 1e-6)

    og_total = leaf_dists.get(outgroup, 0.02)
    ingroup_children = [c for c in rooted_tree.children if not c.is_leaf()]

    for node in rooted_tree.traverse("postorder"):
        if node.is_root():
            node.dist = 0.0
        elif node.is_leaf():
            if node.name == outgroup:
                node.dist = round(max(og_total * 0.5, 1e-6), 6)
            else:
                node.dist = round(leaf_dists.get(node.name, 1e-6), 6)
        elif ingroup_children and node is ingroup_children[0]:
            node.dist = round(max(og_total * 0.5, 1e-6), 6)
        else:
            canon = _outgroup_excluding_split(set(node.get_leaf_names()), all_leaves, outgroup)
            node.dist = round(split_dists.get(canon, 1e-6), 6)


# ==============================================================================
# SECTION 2: FITCH MAXIMUM PARSIMONY (SCALAR & BIT-PARALLEL VECTORIZED)
# ==============================================================================

def fitch_parsimony_steps(tree: Tree, site_states: dict) -> tuple[int, int, int]:
    """
    Computes observed parsimony step count (s_i), minimum possible steps (m_i),
    and maximum star-tree steps (g_i) for a single alignment column (Table 4.1 Row 1).
    """
    unique_states = set(site_states.values()) - AMBIGUOUS_CHARS
    if len(unique_states) <= 1:
        return 0, 0, 0
    m_i = len(unique_states) - 1
    vals = list(site_states.values())
    state_counts = [vals.count(s) for s in unique_states]
    g_i = sum(state_counts) - max(state_counts)

    postorder_sets = {}
    steps = 0
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            ch = site_states.get(node.name, "-")
            postorder_sets[node] = {ch} if ch not in AMBIGUOUS_CHARS else unique_states.copy()
        else:
            child_sets = [postorder_sets[c] for c in node.children]
            intersect = set.intersection(*child_sets)
            if intersect:
                postorder_sets[node] = intersect
            else:
                postorder_sets[node] = set.union(*child_sets)
                steps += 1
    return steps, m_i, g_i


def fitch_binary_gap_states(
    tree: Tree, binary_states: dict[str, int]
) -> tuple[dict[str, int], int, int, int]:
    """
    Runs two-pass Fitch Maximum Parsimony (post-order bottom-up + pre-order top-down)
    on a binary gap-coded column (0 = present nucleotide/AA, 1 = alignment gap '-')
    per Section 4.1.1 Item 2.2 (Page 10).
    """
    vals = list(binary_states.values())
    unique_states = set(vals)
    if len(unique_states) <= 1:
        only = next(iter(unique_states)) if unique_states else 0
        return {n.name: only for n in tree.traverse() if n.name}, 0, 0, 0

    m_i = len(unique_states) - 1
    state_counts = [vals.count(s) for s in unique_states]
    g_i = sum(state_counts) - max(state_counts)

    postorder_sets = {}
    steps = 0
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            postorder_sets[node] = {binary_states.get(node.name, 0)}
        else:
            child_sets = [postorder_sets[c] for c in node.children]
            intersect = set.intersection(*child_sets)
            if intersect:
                postorder_sets[node] = intersect
            else:
                postorder_sets[node] = set.union(*child_sets)
                steps += 1

    assigned: dict[str, int] = {}
    for node in tree.traverse("preorder"):
        cand = postorder_sets[node]
        if node.is_root():
            state = 0 if 0 in cand else min(cand)
        else:
            parent_state = assigned[node.up.name]
            state = parent_state if parent_state in cand else min(cand)
        if node.name:
            assigned[node.name] = state

    return assigned, steps, m_i, g_i


def compute_vectorized_fitch_and_gaps(
    tree: Tree,
    taxa_ids: tuple[str, ...],
    char_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Bit-parallel vectorized Fitch Maximum Parsimony across all L columns simultaneously.
    Returns:
      - s_arr: (L,) observed substitution parsimony steps
      - m_arr: (L,) minimum substitution steps
      - g_arr: (L,) maximum star-tree substitution steps
      - gap_pct_arr: (L,) percentage of taxa containing '-' at each column
      - gap_ci_arr: (L,) Consistency Index of binary gap character at each column
      - gap_node_states: dict mapping node_name -> (L,) binary array in {0, 1}
    """
    n_taxa, seq_len = char_mat.shape
    taxon_to_row = {tid: idx for idx, tid in enumerate(taxa_ids)}

    # 1. Substitution parsimony bitmasks
    all_unique = sorted(set(np.unique(char_mat).tolist()) - AMBIGUOUS_CHARS)
    char_to_bit = {ch: np.uint32(1 << idx) for idx, ch in enumerate(all_unique)}

    leaf_raw_bits = np.zeros((n_taxa, seq_len), dtype=np.uint32)
    counts_by_state = np.zeros((max(len(all_unique), 1), seq_len), dtype=np.int32)
    for idx, ch in enumerate(all_unique):
        mask_ch = char_mat == ch
        leaf_raw_bits[mask_ch] = np.uint32(1 << idx)
        counts_by_state[idx] = np.sum(mask_ch, axis=0)

    col_union_bits = np.bitwise_or.reduce(leaf_raw_bits, axis=0)
    # Count distinct valid states per column via bit counting
    distinct_counts = np.zeros(seq_len, dtype=np.int32)
    tmp_bits = col_union_bits.copy()
    while np.any(tmp_bits):
        distinct_counts += (tmp_bits & np.uint32(1)).astype(np.int32)
        tmp_bits >>= np.uint32(1)

    variable_mask = distinct_counts >= 2
    m_arr = np.where(variable_mask, distinct_counts - 1, 0)
    total_valid_counts = np.sum(counts_by_state, axis=0)
    max_state_counts = np.max(counts_by_state, axis=0)
    g_arr = np.where(variable_mask, total_valid_counts - max_state_counts, 0)

    node_masks: dict[Tree, np.ndarray] = {}
    s_arr = np.zeros(seq_len, dtype=np.int32)
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            r_idx = taxon_to_row.get(node.name)
            if r_idx is not None:
                row_bits = leaf_raw_bits[r_idx]
                node_masks[node] = np.where(row_bits != 0, row_bits, col_union_bits)
            else:
                node_masks[node] = col_union_bits.copy()
        else:
            child_arrs = [node_masks[c] for c in node.children]
            inter = np.bitwise_and.reduce(child_arrs)
            union = np.bitwise_or.reduce(child_arrs)
            has_inter = inter != 0
            node_masks[node] = np.where(has_inter, inter, union)
            s_arr += np.where(variable_mask & (~has_inter), 1, 0)

    # 2. Binary gap-coded two-pass Fitch parsimony (bit 1 -> state 0, bit 2 -> state 1)
    is_gap_mat = char_mat == "-"
    gap_pct_arr = (np.sum(is_gap_mat, axis=0) / max(n_taxa, 1)) * 100.0
    gap_counts_1 = np.sum(is_gap_mat, axis=0)
    gap_variable = (gap_counts_1 > 0) & (gap_counts_1 < n_taxa)
    gap_m_arr = np.where(gap_variable, 1, 0)

    gap_post: dict[Tree, np.ndarray] = {}
    gap_s_arr = np.zeros(seq_len, dtype=np.int32)
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            r_idx = taxon_to_row.get(node.name)
            if r_idx is not None:
                gap_post[node] = np.where(is_gap_mat[r_idx], np.uint8(2), np.uint8(1))
            else:
                gap_post[node] = np.ones(seq_len, dtype=np.uint8)
        else:
            c_arrs = [gap_post[c] for c in node.children]
            inter = np.bitwise_and.reduce(c_arrs)
            union = np.bitwise_or.reduce(c_arrs)
            has_inter = inter != 0
            gap_post[node] = np.where(has_inter, inter, union)
            gap_s_arr += np.where(gap_variable & (~has_inter), 1, 0)

    gap_ci_arr = np.where(gap_s_arr > 0, gap_m_arr / np.maximum(gap_s_arr, 1), 1.0)

    gap_assigned_by_node: dict[Tree, np.ndarray] = {}
    gap_node_states: dict[str, np.ndarray] = {}
    for node in tree.traverse("preorder"):
        cand = gap_post[node]
        if node.is_root():
            # Prefer state 0 (bit 1) when tied, matching fitch_binary_gap_states
            state_arr = np.where((cand & np.uint8(1)) != 0, 0, 1).astype(np.int8)
        else:
            p_arr = gap_assigned_by_node[node.up]
            p_bit = np.where(p_arr == 1, np.uint8(2), np.uint8(1))
            keep_parent = (cand & p_bit) != 0
            min_cand = np.where((cand & np.uint8(1)) != 0, 0, 1).astype(np.int8)
            state_arr = np.where(keep_parent, p_arr, min_cand)
        gap_assigned_by_node[node] = state_arr
        if node.name:
            gap_node_states[node.name] = state_arr

    # Override non-variable columns with exact constant state (0 or 1)
    all_gap_cols = gap_counts_1 == n_taxa
    no_gap_cols = gap_counts_1 == 0
    for n_name, arr in gap_node_states.items():
        arr[no_gap_cols] = 0
        arr[all_gap_cols] = 1

    return s_arr, m_arr, g_arr, gap_pct_arr, gap_ci_arr, gap_node_states


def shannon_mutual_information_gap_bits(
    binary_states: dict[str, int], clade_taxa: set[str]
) -> float:
    """
    Computes Shannon mutual information I(X_i; b_k) = H(X_i) - H(X_i | b_k) in bits
    for a binary gap-coded alignment column X_i partitioned by branch b_k (Section 4.3.2.C.1, Page 26).
    """
    all_taxa = list(binary_states.keys())
    n_total = len(all_taxa)
    if n_total == 0:
        return 0.0

    def _entropy(vals: list[int]) -> float:
        if not vals:
            return 0.0
        n = len(vals)
        p1 = sum(vals) / n
        p0 = 1.0 - p1
        h = 0.0
        if p0 > 0.0:
            h -= p0 * np.log2(p0)
        if p1 > 0.0:
            h -= p1 * np.log2(p1)
        return float(h)

    all_vals = [binary_states[t] for t in all_taxa]
    in_vals = [binary_states[t] for t in all_taxa if t in clade_taxa]
    out_vals = [binary_states[t] for t in all_taxa if t not in clade_taxa]

    h_x = _entropy(all_vals)
    h_cond = (len(in_vals) / n_total) * _entropy(in_vals) + (
        len(out_vals) / n_total
    ) * _entropy(out_vals)
    return max(round(h_x - h_cond, 4), 0.0)


# ==============================================================================
# SECTION 3: EXACT FELSENSTEIN DELTA_SLS COMPUTATION VIA IQ-TREE .sitelh
# ==============================================================================

def _parse_sitelh_file(sitelh_path: Path) -> list[np.ndarray]:
    """
    Parses TREE-PUZZLE / IQ-TREE .sitelh format into a list of 1D float64 arrays
    (one array of length L per tree). Handles multi-line wrapped rows safely.
    """
    tokens = sitelh_path.read_text().split()
    if len(tokens) < 2:
        return []
    n_trees = int(tokens[0])
    n_sites = int(tokens[1])
    idx = 2
    tree_vectors: list[np.ndarray] = []
    for _ in range(n_trees):
        if idx >= len(tokens):
            break
        _tree_label = tokens[idx]
        idx += 1
        vals = np.array(tokens[idx : idx + n_sites], dtype=np.float64)
        idx += n_sites
        tree_vectors.append(vals)
    return tree_vectors


def _outgroup_excluding_split(leaf_subset: set[str], all_leaves: set[str], outgroup: str) -> frozenset[str]:
    """Returns the canonical unrooted split half that excludes the outgroup taxon."""
    if outgroup in leaf_subset:
        return frozenset(all_leaves - leaf_subset)
    return frozenset(leaf_subset)


def compute_branch_delta_sls_matrix(
    fasta_path: str,
    iqtree_bin: str,
    iq_model: str,
    asr_treefile_path: Path,
    rooted_true_tree: Tree,
    outgroup: str,
    work_dir: Path,
) -> dict[str, np.ndarray]:
    """
    Computes exact Felsenstein site-wise log-likelihood support differences:
        delta_SLS_i(b) = ln L(D_i | T_true) - ln L(D_i | T_{-b})
    by collapsing each internal branch b = (u -> v) in T_true (setting branch length to 1e-8
    under -blfix -n 0 -wsl in IQ-TREE) and evaluating all collapsed trees alongside T_true
    in a single C++/AVX2 pass (Manuscript Section 4.1.1 Item 2.1, Page 14).
    """
    t_ml = Tree(asr_treefile_path.read_text().strip(), format=1)
    all_leaves = set(t_ml.get_leaf_names())

    # Index edges in t_ml by canonical outgroup-excluding leaf-set bipartition
    split_to_edge_nodes: dict[frozenset[str], list[Tree]] = {}
    for node in t_ml.traverse():
        if node.is_root():
            continue
        sub_leaves = set(node.get_leaf_names())
        canon = _outgroup_excluding_split(sub_leaves, all_leaves, outgroup)
        if canon:
            split_to_edge_nodes.setdefault(canon, []).append(node)

    target_child_nodes: list[str] = []
    tree_lines: list[str] = [t_ml.write(format=1)]

    for r_node in rooted_true_tree.traverse("postorder"):
        if r_node.is_root() or r_node.is_leaf():
            continue
        clade_set = set(r_node.get_leaf_names())
        canon = _outgroup_excluding_split(clade_set, all_leaves, outgroup)
        t_collapsed = t_ml.copy()
        # Locate corresponding unrooted edge(s) in t_collapsed and collapse length to 1e-8
        matched = False
        for c_node in t_collapsed.traverse():
            if c_node.is_root():
                continue
            c_sub = set(c_node.get_leaf_names())
            if _outgroup_excluding_split(c_sub, all_leaves, outgroup) == canon:
                c_node.dist = 1e-8
                matched = True
        if matched:
            target_child_nodes.append(r_node.name)
            tree_lines.append(t_collapsed.write(format=1))

    if not target_child_nodes:
        return {}

    treels_path = work_dir / "iqtree_collapsed_branches.treels"
    treels_path.write_text("\n".join(tree_lines) + "\n")
    sls_prefix = work_dir / "iqtree_branch_sls"

    cmd_sls = [
        iqtree_bin,
        "-s",
        str(fasta_path),
        "-z",
        str(treels_path),
        "-m",
        iq_model,
        "-blfix",
        "-n",
        "0",
        "-wsl",
        "-nt",
        "1",
        "-pre",
        str(sls_prefix),
        "-redo",
        "-quiet",
    ]
    subprocess.run(cmd_sls, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    sitelh_vecs = _parse_sitelh_file( Path(f"{sls_prefix}.sitelh") )
    if len(sitelh_vecs) != len(target_child_nodes) + 1:
        return {}

    ref_lh = sitelh_vecs[0]
    delta_by_child: dict[str, np.ndarray] = {}
    for idx, child_name in enumerate(target_child_nodes, start=1):
        delta_by_child[child_name] = ref_lh - sitelh_vecs[idx]
    return delta_by_child


# ==============================================================================
# SECTION 4: 200-BP SLIDING-WINDOW RECOMBINATION SCANNER (TABLE 4.1 ROW 4)
# ==============================================================================

def detect_sliding_window_recombination_blocks(
    fasta_path: str,
    rooted_true_path: str,
    window_size: int = 200,
    step_size: int = 100,
    rf_discordance_threshold: float = 0.65,
) -> list[list[int]]:
    """
    Scans 200-site alignment blocks (Table 4.1 Row 4, Page 12) using FastTree 2 and
    flags localized windows exhibiting SH-supported topological incongruence against T_true
    (avoiding false positives from low-signal short-window polytomy noise on empirical MSAs).
    """
    taxa_ids, char_mat, seq_len = load_alignment_data(fasta_path)
    if seq_len < window_size * 2:
        return []

    fasttree_bin = resolve_conda_binary("fasttree")
    seq_type = detect_sequence_alphabet(fasta_path)
    t_true = Tree(str(rooted_true_path), format=1)
    all_leaves = set(t_true.get_leaf_names())
    anchor_leaf = sorted(all_leaves)[0]
    true_canon_splits = {
        _outgroup_excluding_split(set(n.get_leaf_names()), all_leaves, anchor_leaf)
        for n in t_true.traverse()
        if not n.is_root() and not n.is_leaf()
    }
    true_canon_splits.discard(frozenset())

    candidate_windows: list[tuple[int, int]] = []
    total_windows = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_fa = Path(tmpdir) / "win.fa"
        for start_0 in range(0, seq_len - window_size + 1, step_size):
            total_windows += 1
            end_0 = start_0 + window_size
            sub_recs = [
                SeqRecord(Seq("".join(char_mat[r_i, start_0:end_0])), id=taxa_ids[r_i], description="")
                for r_i in range(len(taxa_ids))
            ]
            AlignIO.write(MultipleSeqAlignment(sub_recs), str(tmp_fa), "fasta")
            cmd = [fasttree_bin, "-quiet"]
            if seq_type == "DNA":
                cmd.extend(["-nt", "-gtr"])
            else:
                cmd.extend(["-lg"])
            cmd.append(str(tmp_fa))
            try:
                res = subprocess.run(cmd, check=True, capture_output=True, text=True)
                nwk_str = res.stdout.strip()
                if nwk_str.startswith("("):
                    t_win = Tree(nwk_str, format=0)
                    supported_splits = []
                    for wn in t_win.traverse():
                        if wn.is_root() or wn.is_leaf():
                            continue
                        raw_sup = float(wn.support) if wn.support is not None else 0.0
                        sup_norm = raw_sup / 100.0 if raw_sup > 1.0 else raw_sup
                        if sup_norm >= 0.80:
                            sp = _outgroup_excluding_split(set(wn.get_leaf_names()), all_leaves, anchor_leaf)
                            if len(sp) >= 2 and len(sp) <= len(all_leaves) - 2:
                                supported_splits.append(sp)
                    is_incongruent = False
                    if len(supported_splits) >= 2:
                        incompat = sum(1 for sp in supported_splits if sp not in true_canon_splits)
                        incompat_frac = incompat / len(supported_splits)
                        is_incongruent = incompat >= 2 and incompat_frac >= rf_discordance_threshold
                    elif len(all_leaves) <= 8:
                        rf, max_rf = t_true.robinson_foulds(t_win, unrooted_trees=True)[:2]
                        norm_rf = (rf / max_rf) if max_rf > 0 else 0.0
                        is_incongruent = norm_rf >= rf_discordance_threshold

                    if is_incongruent:
                        candidate_windows.append((start_0 + 1, end_0))
            except Exception:
                continue

    # Guard against global alignment-wide noise masquerading as localized recombination
    if total_windows >= 4 and (len(candidate_windows) / total_windows) > 0.50:
        return []

    detected_intervals: list[list[int]] = []
    for c_start, c_end in candidate_windows:
        if detected_intervals and c_start <= detected_intervals[-1][1] + 1:
            detected_intervals[-1][1] = max(detected_intervals[-1][1], c_end)
        else:
            detected_intervals.append([c_start, c_end])
    return detected_intervals


# ==============================================================================
# SECTION 5: MASTER GROUND-TRUTH PIPELINE (T_true, M_true, A_true)
# ==============================================================================

def run_ground_truth_pipeline(
    fasta_path: str,
    true_tree_path: str,
    out_dir: str,
    outgroup: str = "Taxon_01",
    recombination_breakpoints: list | None = None,
    auto_scan_recombination: bool = False,
    expected_stressors: list[str] | None = None,
    verified_consensus_splits: list[list[str]] | None = None,
    substitution_model: str | None = None,
    experimental_ancestors_fasta: str | None = None,
    cataloged_mutations: list[dict] | None = None,
    kept_original_columns_1based: list[int] | None = None,
    documented_lba_taxa: list[str] | None = None,
    documented_clade_anomalies: list[dict] | None = None,
) -> dict:
    """
    Executes Pipeline Step 1 to generate:
      - T_true_rooted.nwk (Outgroup-rooted reference tree with post-order N1..Nk labels)
      - M_true.tsv        (Verified branch ASR substitutions / sequenced ancestors / cataloged mutations
                           with .sitelh delta_SLS & Fitch indel synapomorphies)
      - A_true.json       (Informative-site CI/RI, LBA metrics, gap tracts, and recombination intervals)
    Supports both DNA (GTR+G4 or ModelFinder) and Amino Acid / Protein (LG+G4 or ModelFinder) alignments.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Universal Outgroup Rooting & Deterministic Post-Order Node Labeling on T_true
    t_true, input_has_branch_lengths = load_published_tree_safely(true_tree_path)
    t_true.set_outgroup(t_true & outgroup)
    orig_internal_to_nk: dict[str, str] = {}
    clade_to_nk: dict[frozenset[str], tuple[str, str]] = {}
    k_idx = 1
    for node in t_true.traverse("postorder"):
        if not node.is_leaf():
            old_name = str(node.name).strip() if node.name else ""
            new_name = f"N{k_idx}"
            if old_name and not old_name.replace(".", "", 1).isdigit():
                orig_internal_to_nk[old_name] = new_name
            node.name = new_name
            k_idx += 1
    for node in t_true.traverse("postorder"):
        if not node.is_leaf() and not node.is_root():
            clade_to_nk[frozenset(node.get_leaf_names())] = (node.up.name, node.name)

    rooted_true_path = out_path / "T_true_rooted.nwk"
    t_true.write(format=1, outfile=str(rooted_true_path))

    # 2. Run IQ-TREE 2 Empirical Bayesian ASR (--ancestral) and site log-likelihoods (-wsl)
    taxa_ids, char_mat, seq_len = load_alignment_data(fasta_path)
    n_taxa = len(taxa_ids)
    seq_type = detect_sequence_alphabet(fasta_path)
    default_iq_model = "GTR+G4" if seq_type == "DNA" else "LG+G4"
    iq_model_arg = substitution_model if substitution_model else default_iq_model
    uniform_prior = 0.25 if seq_type == "DNA" else 0.05
    nt_arg = str(default_cpu_threads(4)) if (n_taxa * seq_len >= 50000) else "1"

    prefix = out_path / "iqtree_asr"
    iqtree_bin = resolve_conda_binary("iqtree2")
    cmd = [
        iqtree_bin,
        "-s",
        str(fasta_path),
        "-te",
        str(rooted_true_path),
        "-m",
        iq_model_arg,
        "--ancestral",
        "-wsl",
        "-nt",
        nt_arg,
        "-pre",
        str(prefix),
        "-redo",
        "-quiet",
    ]
    subprocess.run(cmd, check=True)

    resolved_iq_model = parse_iqtree_best_model(Path(f"{prefix}.iqtree")) or (
        default_iq_model if iq_model_arg.upper() == "MFP" else iq_model_arg
    )

    # 2a. If published reference tree lacked branch lengths, populate ML lengths from iqtree_asr.treefile
    asr_treefile = Path(f"{prefix}.treefile")
    if not input_has_branch_lengths and asr_treefile.exists():
        _transfer_ml_branch_lengths_to_rooted_tree(t_true, asr_treefile, outgroup)
        t_true.write(format=1, outfile=str(rooted_true_path))

    # 2b. Compute exact Felsenstein branch-collapse delta_SLS_i(b) from .sitelh
    delta_sls_by_child = compute_branch_delta_sls_matrix(
        fasta_path=fasta_path,
        iqtree_bin=iqtree_bin,
        iq_model=resolved_iq_model,
        asr_treefile_path=asr_treefile,
        rooted_true_tree=t_true,
        outgroup=outgroup,
        work_dir=out_path,
    )

    # 3. Parse .state file for marginal posterior probabilities P(X_{u,i} = s | T_true, D) >= 0.90
    state_df = pd.read_csv(f"{prefix}.state", sep="\t", comment="#")
    state_probs = [c for c in state_df.columns if c.startswith("p_")]
    state_df["max_posterior"] = state_df[state_probs].max(axis=1)

    # 4. Vectorized Bit-Parallel Fitch Parsimony CI_i, RI_i and Binary Gap States
    s_arr, m_arr, g_arr, gap_pct_arr, gap_ci_arr, gap_node_states = (
        compute_vectorized_fitch_and_gaps(t_true, taxa_ids, char_mat)
    )
    ci_arr = np.where(s_arr > 0, m_arr / np.maximum(s_arr, 1), 1.0)
    ri_arr = np.where(
        (g_arr > m_arr) & (s_arr > 0),
        (g_arr - s_arr) / np.maximum(g_arr - m_arr, 1),
        1.0,
    )

    site_metrics = {}
    for col_idx in range(seq_len):
        site_1based = col_idx + 1
        site_metrics[site_1based] = {
            "ci_i": round(float(ci_arr[col_idx]), 4),
            "ri_i": round(float(ri_arr[col_idx]), 4),
            "s_i": int(s_arr[col_idx]),
            "gap_pct": round(float(gap_pct_arr[col_idx]), 2),
            "gap_ci_i": round(float(gap_ci_arr[col_idx]), 4),
        }

    # Optional Tier 1 directly sequenced experimental ancestors lookup (mapped to N1..Nk)
    exp_ancestor_seqs: dict[str, str] = {}
    if experimental_ancestors_fasta and Path(experimental_ancestors_fasta).exists():
        for rec in SeqIO.parse(str(experimental_ancestors_fasta), "fasta"):
            raw_anc_id = rec.id.strip()
            mapped_id = orig_internal_to_nk.get(raw_anc_id, raw_anc_id)
            s_anc = str(rec.seq).upper().replace(".", "-").replace("?", "-").replace("*", "-")
            if (
                kept_original_columns_1based
                and len(s_anc) != seq_len
                and len(s_anc) >= max(kept_original_columns_1based)
            ):
                s_anc = "".join(s_anc[pos_1 - 1] for pos_1 in kept_original_columns_1based)
            if len(s_anc) != seq_len:
                raise ValueError(
                    f"Experimental ancestor '{raw_anc_id}' in {experimental_ancestors_fasta} has length "
                    f"{len(s_anc)}, expected aligned sequence length {seq_len}."
                )
            exp_ancestor_seqs[mapped_id] = s_anc

    # 5. Build M_true (Causal Mutations per branch u -> v: Sequenced Ancestors + ASR + Fitch Indels)
    node_state_lookup = state_df.set_index(["Node", "Site"]).to_dict(orient="index")
    root_node_name = t_true.name
    ingroup_children = [c for c in t_true.children if not c.is_leaf()]
    fallback_root_asr_name = ingroup_children[0].name if ingroup_children else None

    m_true_by_key: dict[tuple[str, str, int], dict] = {}
    indel_boundaries_by_branch = {}
    is_gap_int_mat = (char_mat == "-").astype(int)

    for node in t_true.traverse("postorder"):
        if node.is_root() or node.is_leaf():
            continue
        u_name = node.up.name
        v_name = node.name
        clade_taxa = set(node.get_leaf_names())

        # 5a. Contiguous indel boundaries [i_start, i_end] supporting branch u -> v
        g_u_vec = gap_node_states.get(u_name)
        g_v_vec = gap_node_states.get(v_name)
        indel_sites = []
        if g_u_vec is not None and g_v_vec is not None:
            diff_cols = np.flatnonzero(g_u_vec != g_v_vec)
            for col_0 in diff_cols:
                indel_sites.append((int(col_0 + 1), int(g_u_vec[col_0]), int(g_v_vec[col_0])))

        site_to_tract = {}
        branch_tracts = []
        idx = 0
        while idx < len(indel_sites):
            start_s, g_u0, g_v0 = indel_sites[idx]
            end_s = start_s
            while (
                idx + 1 < len(indel_sites)
                and indel_sites[idx + 1][0] == end_s + 1
                and indel_sites[idx + 1][1] == g_u0
                and indel_sites[idx + 1][2] == g_v0
            ):
                idx += 1
                end_s = indel_sites[idx][0]
            tract_label = f"{start_s}-{end_s}"
            branch_tracts.append(
                {
                    "i_start": start_s,
                    "i_end": end_s,
                    "tract_length": end_s - start_s + 1,
                    "event": "DELETION" if (g_u0 == 0 and g_v0 == 1) else "INSERTION",
                }
            )
            for s_pos in range(start_s, end_s + 1):
                site_to_tract[s_pos] = (g_u0, g_v0, tract_label)
            idx += 1
        if branch_tracts:
            indel_boundaries_by_branch[f"{u_name}->{v_name}"] = branch_tracts

        branch_sls_vec = delta_sls_by_child.get(v_name)
        asr_u_key = fallback_root_asr_name if (u_name == root_node_name and fallback_root_asr_name) else u_name
        exp_u_seq = exp_ancestor_seqs.get(u_name) or exp_ancestor_seqs.get(asr_u_key or "")
        exp_v_seq = exp_ancestor_seqs.get(v_name)

        for site_i in range(1, seq_len + 1):
            col_0 = site_i - 1
            if site_i in site_to_tract:
                g_u, g_v, tract_str = site_to_tract[site_i]
                bin_col_dict = {taxa_ids[r]: int(is_gap_int_mat[r, col_0]) for r in range(n_taxa)}
                mi_bits = shannon_mutual_information_gap_bits(bin_col_dict, clade_taxa)
                non_gap_chars = [ch for ch in char_mat[:, col_0] if ch not in AMBIGUOUS_CHARS]
                consensus_char = max(set(non_gap_chars), key=non_gap_chars.count) if non_gap_chars else "CHAR"
                m_true_by_key[(u_name, v_name, site_i)] = {
                    "parent_node": u_name,
                    "child_node": v_name,
                    "site": site_i,
                    "parent_state": "-" if g_u == 1 else consensus_char,
                    "child_state": "-" if g_v == 1 else consensus_char,
                    "posterior_prob": 1.0,
                    "delta_sls": max(mi_bits, 1e-4),
                    "is_indel": True,
                    "ci_score": site_metrics[site_i]["gap_ci_i"],
                    "indel_boundary": tract_str,
                }
                continue

            if exp_u_seq is not None and exp_v_seq is not None:
                su_exp = exp_u_seq[col_0]
                sv_exp = exp_v_seq[col_0]
                if (
                    su_exp != sv_exp
                    and su_exp not in AMBIGUOUS_CHARS
                    and sv_exp not in AMBIGUOUS_CHARS
                ):
                    raw_dsls = float(branch_sls_vec[col_0]) if branch_sls_vec is not None else 1.0
                    m_true_by_key[(u_name, v_name, site_i)] = {
                        "parent_node": u_name,
                        "child_node": v_name,
                        "site": site_i,
                        "parent_state": su_exp,
                        "child_state": sv_exp,
                        "posterior_prob": 1.0,
                        "delta_sls": round(max(raw_dsls, 0.01), 4),
                        "is_indel": False,
                        "ci_score": site_metrics[site_i]["ci_i"],
                        "indel_boundary": "",
                    }
                    continue

            u_row = node_state_lookup.get((asr_u_key, site_i))
            v_row = node_state_lookup.get((v_name, site_i))
            if u_row and v_row:
                s_u, p_u = str(u_row["State"]), float(u_row["max_posterior"])
                s_v, p_v = str(v_row["State"]), float(v_row["max_posterior"])
                if s_u != s_v and p_u >= 0.90 and p_v >= 0.90:
                    if branch_sls_vec is not None:
                        exact_dsls = float(branch_sls_vec[col_0])
                    else:
                        exact_dsls = float(np.log(min(p_u, p_v) / uniform_prior))
                    if exact_dsls > 1e-6:
                        m_true_by_key[(u_name, v_name, site_i)] = {
                            "parent_node": u_name,
                            "child_node": v_name,
                            "site": site_i,
                            "parent_state": s_u,
                            "child_state": s_v,
                            "posterior_prob": round(min(p_u, p_v), 4),
                            "delta_sls": round(exact_dsls, 4),
                            "is_indel": False,
                            "ci_score": site_metrics[site_i]["ci_i"],
                            "indel_boundary": "",
                        }

    if cataloged_mutations:
        orig_col_lookup = (
            {orig_c: idx + 1 for idx, orig_c in enumerate(kept_original_columns_1based)}
            if kept_original_columns_1based
            else None
        )
        nk_to_parent = {
            n.name: n.up.name
            for n in t_true.traverse("postorder")
            if not n.is_root() and not n.is_leaf()
        }
        for cm in cataloged_mutations:
            raw_site = int(cm.get("site", cm.get("site_coordinate", -1)))
            mapped_site = (
                orig_col_lookup.get(raw_site)
                if (orig_col_lookup is not None and not cm.get("already_cleaned_coordinates", False))
                else raw_site
            )
            if mapped_site is None or mapped_site < 1 or mapped_site > seq_len:
                continue

            target_pair: tuple[str, str] | None = None
            if cm.get("clade_taxa"):
                target_pair = clade_to_nk.get(frozenset(cm["clade_taxa"]))
            elif cm.get("child_node"):
                c_raw = str(cm["child_node"])
                c_nk = orig_internal_to_nk.get(c_raw, c_raw)
                if c_nk in nk_to_parent:
                    target_pair = (nk_to_parent[c_nk], c_nk)
            if not target_pair:
                continue

            u_cat, v_cat = target_pair
            su_cat = str(cm.get("parent_state", cm.get("ancestral_state", ""))).upper()
            sv_cat = str(cm.get("child_state", cm.get("derived_state", ""))).upper()
            if not su_cat or not sv_cat or su_cat == sv_cat:
                continue
            is_ind = bool(cm.get("is_indel", "-" in {su_cat, sv_cat}))
            b_vec = delta_sls_by_child.get(v_cat)
            dsls_val = float(b_vec[mapped_site - 1]) if b_vec is not None else 1.0
            m_true_by_key[(u_cat, v_cat, mapped_site)] = {
                "parent_node": u_cat,
                "child_node": v_cat,
                "site": mapped_site,
                "parent_state": su_cat,
                "child_state": sv_cat,
                "posterior_prob": 1.0,
                "delta_sls": round(max(dsls_val, 0.01), 4),
                "is_indel": is_ind,
                "ci_score": site_metrics[mapped_site]["gap_ci_i" if is_ind else "ci_i"],
                "indel_boundary": str(mapped_site) if is_ind else "",
            }

    m_true_records = list(m_true_by_key.values())
    m_true_df = pd.DataFrame(
        m_true_records,
        columns=[
            "parent_node",
            "child_node",
            "site",
            "parent_state",
            "child_state",
            "posterior_prob",
            "delta_sls",
            "is_indel",
            "ci_score",
            "indel_boundary",
        ],
    )
    m_true_df.to_csv(out_path / "M_true.tsv", sep="\t", index=False)

    # 6. Build A_true (Homoplasy, LBA, Indel Boundaries & Recombination Stressor Ground Truth)
    leaf_branch_lengths = {leaf.name: round(float(leaf.dist), 6) for leaf in t_true.iter_leaves()}
    ingroup_leaf_lengths = [
        float(leaf.dist) for leaf in t_true.iter_leaves() if leaf.name != outgroup
    ]
    term_dists = ingroup_leaf_lengths if ingroup_leaf_lengths else list(leaf_branch_lengths.values())
    internal_branch_lengths = {
        n.name: round(float(n.dist), 6)
        for n in t_true.traverse()
        if not n.is_leaf() and not n.is_root() and n.dist > 0
    }
    int_dists = list(internal_branch_lengths.values())
    resolved_int_dists = [d for d in int_dists if d > 1e-5]
    denom_min_int = min(resolved_int_dists) if resolved_int_dists else (min(int_dists) if int_dists else 0.05)
    rho_lba = (max(term_dists) / max(denom_min_int, 1e-4)) if term_dists else 1.0

    median_leaf_len = float(np.median(term_dists)) if term_dists else 0.05
    detected_long_taxa = {
        leaf.name
        for leaf in t_true.iter_leaves()
        if leaf.name != outgroup and float(leaf.dist) >= max(median_leaf_len * 2.5, 0.10)
    }
    # Keep quantitative long_branch_taxa (read by Tool 3) strictly separate from literature-curated
    # documented_lba_taxa (used only by post-hoc EDR scoring) to prevent oracle leakage into Tool 3.
    long_branch_taxa = sorted(detected_long_taxa)
    curated_lba_taxa = (
        sorted(t for t in documented_lba_taxa if t in leaf_branch_lengths)
        if documented_lba_taxa
        else []
    )

    var_mask = s_arr > 0
    mean_inf_ci = float(np.mean(ci_arr[var_mask])) if np.any(var_mask) else 1.0
    ensemble_ci = (
        float(np.sum(m_arr[var_mask]) / np.sum(s_arr[var_mask])) if np.any(var_mask) else 1.0
    )
    inf_ri_mask = (g_arr > m_arr) & (s_arr > 0)
    mean_inf_ri = float(np.mean(ri_arr[inf_ri_mask])) if np.any(inf_ri_mask) else 1.0

    gap_cols = [int(i + 1) for i in np.flatnonzero(gap_pct_arr > 0.0)]
    gapped_col_pct = round((len(gap_cols) / max(seq_len, 1)) * 100.0, 2)
    gap_tracts = []
    g_idx = 0
    while g_idx < len(gap_cols):
        g_start = gap_cols[g_idx]
        g_end = g_start
        while g_idx + 1 < len(gap_cols) and gap_cols[g_idx + 1] == g_end + 1:
            g_idx += 1
            g_end = gap_cols[g_idx]
        gap_tracts.append(
            {"i_start": g_start, "i_end": g_end, "l_gap": g_end - g_start + 1}
        )
        g_idx += 1

    documented_recomb = list(recombination_breakpoints) if recombination_breakpoints else []
    detected_recomb: list[list[int]] = []
    if auto_scan_recombination:
        detected_recomb = detect_sliding_window_recombination_blocks(
            fasta_path=fasta_path,
            rooted_true_path=str(rooted_true_path),
        )
    recomb_list = documented_recomb if documented_recomb else detected_recomb

    min_int_len = denom_min_int
    if expected_stressors is not None:
        gt_stressors = list(expected_stressors)
    else:
        gt_stressors = []
        if documented_clade_anomalies:
            for item in documented_clade_anomalies:
                st_lbl = str(item.get("stressor", "")).strip()
                if st_lbl and st_lbl not in gt_stressors:
                    gt_stressors.append(st_lbl)
        if rho_lba >= 5.0 and len(long_branch_taxa) >= 1:
            tag = "SHORT_BRANCH_HOMOPLASY_AND_LBA" if min_int_len < 0.005 else "LBA"
            if tag not in gt_stressors:
                gt_stressors.append(tag)
        if gapped_col_pct >= 12.0 or float(np.mean(gap_pct_arr)) >= 10.0:
            if "INDEL_GAP" not in gt_stressors:
                gt_stressors.append("INDEL_GAP")
        if recomb_list and "RECOMBINATION" not in gt_stressors:
            gt_stressors.append("RECOMBINATION")
        if min_int_len < 0.005 or mean_inf_ci < 0.65:
            if (
                "SHORT_BRANCH_HOMOPLASY_AND_LBA" not in gt_stressors
                and "SHORT_BRANCH_HOMOPLASY" not in gt_stressors
            ):
                gt_stressors.append("SHORT_BRANCH_HOMOPLASY")

    a_true = {
        "sequence_alphabet": seq_type,
        "substitution_model": resolved_iq_model,
        "input_tree_had_branch_lengths": input_has_branch_lengths,
        "outgroup_taxon": outgroup,
        "global_rho_lba": round(rho_lba, 3),
        "min_internal_branch_length": round(min_int_len, 6),
        "raw_min_internal_branch_length": round(min(int_dists), 6) if int_dists else round(min_int_len, 6),
        "leaf_branch_lengths": leaf_branch_lengths,
        "internal_branch_lengths": internal_branch_lengths,
        "long_branch_taxa": long_branch_taxa,
        "documented_lba_taxa": curated_lba_taxa,
        "mean_alignment_ci": round(float(np.mean(ci_arr)), 4),
        "mean_informative_ci": round(mean_inf_ci, 4),
        "ensemble_ci": round(ensemble_ci, 4),
        "mean_alignment_ri": round(float(np.mean(ri_arr)), 4),
        "mean_informative_ri": round(mean_inf_ri, 4),
        "mean_gap_density_pct": round(float(np.mean(gap_pct_arr)), 2),
        "gapped_column_pct": gapped_col_pct,
        "max_gap_tract_length": max((t["l_gap"] for t in gap_tracts), default=0),
        "gap_tracts": gap_tracts,
        "branch_indel_synapomorphies": indel_boundaries_by_branch,
        "recombination_scan_performed": bool(auto_scan_recombination),
        "detected_recombination_windows": detected_recomb,
        "documented_recombination_breakpoints": documented_recomb,
        "recombination_breakpoints": recomb_list,
        "expected_stressors": gt_stressors,
        "documented_clade_anomalies": documented_clade_anomalies or [],
        "verified_consensus_splits": verified_consensus_splits,
        "site_metrics": site_metrics,
    }
    with open(out_path / "A_true.json", "w") as f:
        json.dump(a_true, f, indent=2)
    return a_true


if __name__ == "__main__":
    run_ground_truth_pipeline(
        "data/00_golden_toy/toy_alignment.fa",
        "data/00_golden_toy/T_true.nwk",
        "data/02_ground_truths/toy",
    )
