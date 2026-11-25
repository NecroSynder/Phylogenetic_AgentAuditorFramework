"""
================================================================================
PIPELINE STEP 3 (PART A): DETERMINISTIC PYTHON BIOINFORMATICS TOOL SUITE
(Manuscript Section 4.1.2 & Table 4.4, Pages 14–17)
================================================================================
Provides the 5 deterministic bioinformatics functions invoked by the 4-State
LangGraph Diagnostic Agent so the LLM never performs ungrounded arithmetic or
raw coordinate guessing:

  - Tool 1: audit_tree_topology           (State 1 — Topological & Confidence Triage)
  - Tool 2: extract_branch_synapomorphies (State 2 — Causal Mutation & Indel Attribution)
  - Tool 3: diagnose_evolutionary_stressor(State 3 — Clade-Local Stressor Diagnosis)
  - Tool 4: inspect_alignment_window      (State 3 — Local Alignment Entropy & Gap Inspection)
  - Tool 5: verify_claim_coordinates      (State 4 — Branch-Scoped & Prose Verification Gate)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from src.utils.io_cache import (
    load_alignment_data,
    load_ete_tree,
    load_json_dict,
    load_m_true_df,
)

AMBIGUOUS_SET = {"-", "?", "N", "X"}


def get_bipartition_dict(tree) -> dict[frozenset[str], object]:
    """Returns a mapping of leaf-set bipartitions to internal nodes (excluding root)."""
    splits = {}
    for node in tree.traverse("postorder"):
        if not node.is_leaf() and not node.is_root():
            splits[frozenset(node.get_leaf_names())] = node
    return splits


def _infer_pred_meta_path(pred_newick_path: str) -> Path | None:
    p = Path(pred_newick_path)
    candidates = []
    if p.name.endswith("_normalized.nwk"):
        candidates.append(
            p.parent / p.name.replace("_normalized.nwk", "_metadata.json")
        )
    candidates.append(p.with_name(f"{p.stem}_metadata.json"))
    for c in candidates:
        if c.exists():
            return c
    return None


def _splits_compatible(s1: frozenset[str], s2: frozenset[str]) -> bool:
    """Two rooted clades are topologically compatible iff disjoint or nested."""
    inter = s1 & s2
    return (not inter) or s1.issubset(s2) or s2.issubset(s1)


# ==============================================================================
# TOOL 1: audit_tree_topology (State 1 — Equations 4.1, 4.2 & 4.3)
# ==============================================================================

def audit_tree_topology(
    pred_newick_path: str,
    true_newick_path: str,
    pred_meta_path: str | None = None,
    support_threshold: float = 85.0,
    stressor_gt_path: str | None = None,
    use_consensus_mask: bool = True,
) -> dict:
    """
    Compares every internal split b_k in B_pred against the ground-truth split set B_true.
    Supports both synthetic/Tier-1 absolute ground truths and Tier 2/3 empirical consensus split masks
    (verified_consensus_splits in A_true.json; Manuscript Section 4.1.1.1.2). Pass
    use_consensus_mask=False to evaluate unmasked topology metrics against the full published T_true.
    """
    t_pred = load_ete_tree(pred_newick_path, fmt=1)
    t_true = load_ete_tree(true_newick_path, fmt=1)

    resolved_meta_path = Path(pred_meta_path) if pred_meta_path else _infer_pred_meta_path(pred_newick_path)
    pred_meta = load_json_dict(resolved_meta_path) if (resolved_meta_path and resolved_meta_path.exists()) else None

    # Check if Tier 2/3 empirical consensus mask is present in A_true.json
    verified_splits_set: set[frozenset[str]] | None = None
    if use_consensus_mask:
        gt_cand = Path(stressor_gt_path) if stressor_gt_path else (Path(true_newick_path).parent / "A_true.json")
        if gt_cand.exists():
            a_true_obj = load_json_dict(gt_cand)
            raw_ver = a_true_obj.get("verified_consensus_splits")
            if raw_ver is not None:
                verified_splits_set = {frozenset(s) for s in raw_ver}

    b_pred = get_bipartition_dict(t_pred)
    b_true = get_bipartition_dict(t_true)

    pred_keys = set(b_pred.keys())
    true_keys = set(b_true.keys())

    # Eq. 4.1: Normalized Robinson-Foulds Distance (masked to verified splits on Tier 3 empirical data)
    if verified_splits_set is not None:
        effective_true = true_keys & verified_splits_set
        incompatible_pred = {
            pk
            for pk in pred_keys
            if pk not in effective_true
            and any(not _splits_compatible(pk, vk) for vk in effective_true)
        }
        missed_true = effective_true - pred_keys
        sym_diff = len(incompatible_pred) + len(missed_true)
        denom = len(incompatible_pred) + len(pred_keys & effective_true) + len(effective_true)
        rf_norm = round(sym_diff / denom, 4) if denom > 0 else 0.0
    else:
        incompatible_pred = pred_keys - true_keys
        sym_diff = len(pred_keys - true_keys) + len(true_keys - pred_keys)
        denom = len(pred_keys) + len(true_keys)
        rf_norm = round(sym_diff / denom, 4) if denom > 0 else 0.0

    node_records = []
    false_conf_count = 0
    erroneous_splits_count = len(incompatible_pred)

    for split_set, node in b_pred.items():
        node_id = node.name
        if pred_meta and node_id in pred_meta.get("nodes", {}):
            sup_pct = float(pred_meta["nodes"][node_id]["support_pct"])
        else:
            raw_s = float(node.support) if node.support is not None else 100.0
            sup_pct = round(raw_s * 100.0 if raw_s <= 1.0 else raw_s, 2)

        in_true = split_set in true_keys
        is_compatible_empirical = (
            verified_splits_set is not None and split_set not in incompatible_pred
        )

        if in_true:
            status = "CONCORDANT"
            t_match = b_true[split_set]
            true_match_node = t_match.name
            true_parent_node = (
                t_match.up.name if (t_match.up and t_match.up.name) else "N_ROOT"
            )
        elif is_compatible_empirical:
            status = "CONCORDANT"
            true_match_node = None
            true_parent_node = None
        elif sup_pct >= support_threshold:
            status = "FALSE_CONFIDENCE_ERROR"
            false_conf_count += 1
            true_match_node = None
            true_parent_node = None
        else:
            status = "DISCORDANT_LOW_SUPPORT"
            true_match_node = None
            true_parent_node = None

        parent_id = node.up.name if (node.up and node.up.name) else "N_ROOT"
        node_records.append(
            {
                "node_id": node_id,
                "parent_node_id": parent_id,
                "true_matching_node_id": true_match_node,
                "true_parent_node_id": true_parent_node,
                "clade_taxa": sorted(list(split_set)),
                "support_pct": sup_pct,
                "topological_status": status,
            }
        )

    fci_global = round(false_conf_count / len(pred_keys), 4) if pred_keys else 0.0
    fci_cond = (
        round(false_conf_count / erroneous_splits_count, 4)
        if erroneous_splits_count > 0
        else 0.0
    )

    return {
        "normalized_rf_distance": rf_norm,
        "fci_global": fci_global,
        "fci_conditional": fci_cond,
        "total_internal_nodes": len(pred_keys),
        "false_confidence_nodes_count": false_conf_count,
        "node_bipartitions": node_records,
    }


# ==============================================================================
# HELPER: DETERMINISTIC CLADE-PARTITIONING CHARACTER / INDEL EXTRACTION
# ==============================================================================

def _extract_clade_partitioning_sites(
    fasta_path: str,
    clade_taxa: list[str],
    parent_node_id: str,
    child_node_id: str,
    site_metrics: dict | None = None,
) -> list[dict]:
    """
    For discordant clades (b_k not in B_true) that have no branch in M_true.tsv,
    deterministically identifies alignment columns (homoplastic/convergent substitutions
    and indels) whose character states partition clade_taxa from non-clade taxa with
    strictly positive Shannon mutual information I(X_i; b_k) > 0.
    """
    taxa_ids, char_mat, seq_len = load_alignment_data(fasta_path)
    clade_set = set(clade_taxa)
    in_mask = np.array([tid in clade_set for tid in taxa_ids], dtype=bool)
    out_mask = ~in_mask

    n_in = int(np.sum(in_mask))
    n_out = int(np.sum(out_mask))
    n_tot = len(taxa_ids)
    if n_in == 0 or n_out == 0:
        return []

    candidates: list[dict] = []
    ambig_non_gap = {"?", "N", "X"}

    def _shannon_bits(chars_arr: np.ndarray) -> float:
        if chars_arr.size == 0:
            return 0.0
        _, cnts = np.unique(chars_arr, return_counts=True)
        probs = cnts / chars_arr.size
        return float(-np.sum(probs * np.log2(probs)))

    for col_0 in range(seq_len):
        col = char_mat[:, col_0]
        in_col = [c for c in col[in_mask] if c not in ambig_non_gap]
        out_col = [c for c in col[out_mask] if c not in ambig_non_gap]
        if not in_col or not out_col:
            continue

        in_vals, in_cnts = np.unique(in_col, return_counts=True)
        out_vals, out_cnts = np.unique(out_col, return_counts=True)
        s_v = str(in_vals[int(np.argmax(in_cnts))])
        s_u = str(out_vals[int(np.argmax(out_cnts))])
        if s_u == s_v:
            continue

        freq_v_in = float(np.max(in_cnts)) / len(in_col)
        freq_v_out = float(in_col_out := out_col.count(s_v)) / len(out_col)
        # Require genuine clade enrichment (majority in clade & minority outside clade)
        if freq_v_in < 0.60 or freq_v_out > 0.40:
            continue

        clean_mask = ~np.isin(col, ["?", "N", "X"])
        n_clean = int(np.sum(clean_mask))
        if n_clean < 2:
            continue
        h_all = _shannon_bits(col[clean_mask])
        in_clean = col[in_mask & clean_mask]
        out_clean = col[out_mask & clean_mask]
        h_cond = (in_clean.size / n_clean) * _shannon_bits(in_clean) + (
            out_clean.size / n_clean
        ) * _shannon_bits(out_clean)
        mi_bits = max(round(h_all - h_cond, 4), 0.0)
        if mi_bits <= 1e-4:
            continue

        site_1 = col_0 + 1
        is_indel = "-" in {s_u, s_v}
        sm_entry = {}
        if site_metrics:
            sm_entry = site_metrics.get(str(site_1)) or site_metrics.get(site_1) or {}
        ci_val = float(
            sm_entry.get("gap_ci_i" if is_indel else "ci_i", 0.5)
        )

        candidates.append(
            {
                "parent_node": parent_node_id,
                "child_node": child_node_id,
                "site": site_1,
                "parent_state": s_u,
                "child_state": s_v,
                "posterior_prob": round(freq_v_in, 4),
                "delta_sls": mi_bits,
                "is_indel": is_indel,
                "ci_score": round(ci_val, 4),
                "indel_boundary": str(site_1) if is_indel else "",
            }
        )

    # Annotate contiguous indel tract boundaries among extracted indel columns
    indel_cands = [c for c in candidates if c["is_indel"]]
    indel_cands.sort(key=lambda x: x["site"])
    idx = 0
    while idx < len(indel_cands):
        s0 = indel_cands[idx]["site"]
        s1 = s0
        j = idx
        while (
            j + 1 < len(indel_cands)
            and indel_cands[j + 1]["site"] == s1 + 1
            and indel_cands[j + 1]["parent_state"] == indel_cands[idx]["parent_state"]
            and indel_cands[j + 1]["child_state"] == indel_cands[idx]["child_state"]
        ):
            j += 1
            s1 = indel_cands[j]["site"]
        tract_str = f"{s0}-{s1}"
        for k in range(idx, j + 1):
            indel_cands[k]["indel_boundary"] = tract_str
        idx = j + 1

    candidates.sort(key=lambda r: (r["delta_sls"], r["posterior_prob"]), reverse=True)
    return candidates


# ==============================================================================
# TOOL 2: extract_branch_synapomorphies (State 2 — ASR & Fitch Indel Lookup)
# ==============================================================================

def extract_branch_synapomorphies(
    fasta_path: str,
    asr_state_path: str,
    parent_node_id: str,
    child_node_id: str,
    is_concordant: bool = True,
    clade_taxa: list[str] | None = None,
    stressor_gt_path: str | None = None,
    top_k: int | None = 10,
) -> list[dict]:
    """
    Retrieves verified site coordinates i in {1..L}, state transitions (s_u -> s_v),
    log-likelihood / mutual-information support (delta_SLS_i), and indel boundaries:
      - Concordant splits (is_concordant=True): Looks up the exact matched reference branch
        (true_parent_node_id -> true_matching_node_id) in M_true.tsv.
      - Discordant splits (is_concordant=False): NEVER looks up T_pred's node ID in M_true.tsv
        (preventing cross-clade contamination); instead extracts clade-partitioning convergent
        substitutions and indels directly from alignment.fa + A_true.json.
    """
    if not is_concordant:
        if not clade_taxa:
            return []
        gt_path = (
            Path(stressor_gt_path)
            if stressor_gt_path
            else (Path(asr_state_path).parent / "A_true.json")
        )
        site_metrics = (
            load_json_dict(gt_path).get("site_metrics") if gt_path.exists() else None
        )
        disc_sites = _extract_clade_partitioning_sites(
            fasta_path=fasta_path,
            clade_taxa=clade_taxa,
            parent_node_id=parent_node_id,
            child_node_id=child_node_id,
            site_metrics=site_metrics,
        )
        return disc_sites[:top_k] if top_k is not None else disc_sites

    if not child_node_id:
        return []

    df = load_m_true_df(asr_state_path)
    if df.empty:
        return []

    if parent_node_id and parent_node_id != "N_ROOT":
        sub = df[
            (df["parent_node"] == parent_node_id) & (df["child_node"] == child_node_id)
        ]
        if sub.empty:
            # In a rooted tree each child_node has a unique incoming branch in T_true
            sub = df[df["child_node"] == child_node_id]
    else:
        sub = df[df["child_node"] == child_node_id]

    sorted_sub = sub.sort_values(by="delta_sls", ascending=False)
    if top_k is not None:
        sorted_sub = sorted_sub.head(top_k)
    return sorted_sub.to_dict(orient="records")


# ==============================================================================
# TOOL 3: diagnose_evolutionary_stressor (State 3 — Clade-Local Attribution)
# ==============================================================================

def diagnose_evolutionary_stressor(
    fasta_path: str,
    pred_newick_path: str,
    node_id: str,
    stressor_gt_path: str,
) -> dict:
    """
    Determines which evolutionary stressor caused a clade split to fail or exhibit homoplasy,
    using clade-local sequence evidence and unbiased informative-site CI metrics:
      - Localized Recombination Check via clade-supporting site overlap with [c_start, c_end]
      - LBA Check via rho_LBA >= 5.0 combined with clade long-branch membership or homoplasy
      - Indel & Gap-Masking Check via clade/alignment gapped-column density & tract boundaries
      - Short-Branch Homoplasy Check via separating branch length < 0.005 or local CI < 0.65
    """
    a_true = load_json_dict(stressor_gt_path)
    t_pred = load_ete_tree(pred_newick_path, fmt=1)

    non_root_dists = [float(n.dist) for n in t_pred.traverse() if not n.is_root()]
    has_real_pred_lengths = not (
        non_root_dists and all(abs(d - 1.0) < 1e-6 for d in non_root_dists)
    )

    target_nodes = t_pred.search_nodes(name=node_id)
    clade_taxa: list[str] = []
    sister_taxa: list[str] = []
    pred_branch_len = 0.05
    if target_nodes:
        t_node = target_nodes[0]
        clade_taxa = sorted(t_node.get_leaf_names())
        pred_branch_len = float(t_node.dist)
        for sis in t_node.get_sisters():
            sister_taxa.extend(sis.get_leaf_names())

    min_true_int_len = float(a_true.get("min_internal_branch_length", 0.05))
    local_sep_len = min_true_int_len
    true_rooted_cand = Path(stressor_gt_path).parent / "T_true_rooted.nwk"
    if clade_taxa and true_rooted_cand.exists():
        try:
            t_true = load_ete_tree(true_rooted_cand, fmt=1)
            true_splits = get_bipartition_dict(t_true)
            clade_frozen = frozenset(clade_taxa)
            if clade_frozen in true_splits:
                local_sep_len = float(true_splits[clade_frozen].dist)
            else:
                true_leaf_set = set(t_true.get_leaf_names())
                valid_clade_leaves = [lf for lf in clade_taxa if lf in true_leaf_set]
                if len(valid_clade_leaves) >= 2:
                    mrca = t_true.get_common_ancestor(valid_clade_leaves)
                    spanned_dists: list[float] = []
                    for lf_name in valid_clade_leaves:
                        cur = (t_true & lf_name).up
                        while cur is not None and cur != mrca and not cur.is_root():
                            spanned_dists.append(float(cur.dist))
                            cur = cur.up
                    resolved_spanned = [d for d in spanned_dists if d > 1e-5]
                    if resolved_spanned:
                        local_sep_len = float(min(resolved_spanned))
                    elif spanned_dists:
                        local_sep_len = float(min(spanned_dists))
        except Exception:
            local_sep_len = min_true_int_len

    eff_branch_len = pred_branch_len if has_real_pred_lengths else local_sep_len

    site_metrics = a_true.get("site_metrics", {})
    clade_sites = (
        _extract_clade_partitioning_sites(
            fasta_path=fasta_path,
            clade_taxa=clade_taxa,
            parent_node_id="N_PARENT",
            child_node_id=node_id,
            site_metrics=site_metrics,
        )
        if clade_taxa
        else []
    )

    mean_inf_ci = float(
        a_true.get("mean_informative_ci", a_true.get("mean_alignment_ci", 1.0))
    )
    mean_inf_ri = float(
        a_true.get("mean_informative_ri", a_true.get("mean_alignment_ri", 1.0))
    )

    if clade_sites:
        local_ci = round(
            float(np.mean([float(m["ci_score"]) for m in clade_sites[:15]])), 4
        )
    else:
        local_ci = round(mean_inf_ci, 4)

    long_taxa = set(a_true.get("long_branch_taxa", []))
    leaf_lens = a_true.get("leaf_branch_lengths", {})
    if not long_taxa and leaf_lens:
        vals = list(leaf_lens.values())
        med = float(np.median(vals)) if vals else 0.05
        long_taxa = {k for k, v in leaf_lens.items() if float(v) >= max(med * 2.5, 0.15)}

    clade_set = set(clade_taxa)
    direct_long_taxa = clade_set & long_taxa
    direct_multi_long_branch = len(direct_long_taxa) >= 2
    sister_long_taxa = set(sister_taxa) & long_taxa
    sister_repelled_long_branch = bool(
        sister_taxa
        and len(sister_taxa) <= 3
        and sister_long_taxa
        and local_ci < 0.75
    )
    clade_has_long_branch = bool(direct_long_taxa) or sister_repelled_long_branch

    # Clade-local rho_LBA = mean(l_long) / l_internal (PDF Table 2)
    relevant_long_taxa = direct_long_taxa if direct_long_taxa else (sister_long_taxa if sister_repelled_long_branch else set())
    if relevant_long_taxa and leaf_lens:
        long_vals = [float(leaf_lens[t]) for t in relevant_long_taxa if t in leaf_lens]
        mean_long_len = float(np.mean(long_vals)) if long_vals else 0.2
        rho_lba = round(mean_long_len / max(local_sep_len, 1e-4), 3)
    elif relevant_long_taxa:
        rho_lba = float(a_true.get("global_rho_lba", 1.0))
    elif leaf_lens and clade_set:
        all_leaf_vals = [float(v) for v in leaf_lens.values()]
        med_leaf = float(np.median(all_leaf_vals)) if all_leaf_vals else 0.05
        clade_leaf_vals = [float(leaf_lens[t]) for t in clade_set if t in leaf_lens]
        mean_clade_leaf = float(np.mean(clade_leaf_vals)) if clade_leaf_vals else med_leaf
        rho_lba = round(mean_clade_leaf / max(med_leaf, 1e-4), 3)
    else:
        rho_lba = float(a_true.get("global_rho_lba", 1.0))

    # Local gap density for clade_taxa & indel tract overlap
    taxa_ids, char_mat, seq_len = load_alignment_data(fasta_path)
    if clade_set:
        in_mask = np.array([tid in clade_set for tid in taxa_ids], dtype=bool)
        clade_mat = char_mat[in_mask] if np.any(in_mask) else char_mat
    else:
        clade_mat = char_mat

    clade_cell_gap_pct = round(
        float(np.mean(clade_mat == "-") * 100.0), 2
    ) if clade_mat.size else 0.0
    clade_gapped_col_pct = round(
        float(np.mean(np.any(clade_mat == "-", axis=0)) * 100.0), 2
    ) if clade_mat.size else 0.0
    aln_gapped_col_pct = float(
        a_true.get("gapped_column_pct", a_true.get("mean_gap_density_pct", 0.0))
    )
    has_indel_support = any(bool(m["is_indel"]) for m in clade_sites[:10])
    effective_gap_pct = max(clade_cell_gap_pct, clade_gapped_col_pct)

    # Localized recombination enrichment check across [c_start, c_end]
    # Normalizes against background window width f_recomb so genome-wide uniform LBA
    # homoplasy does not falsely trigger RECOMBINATION by random Poisson overlap.
    # When blind sequence scanning ran (empirical corpus), read strictly detected_recombination_windows
    # so human-curated documented_recombination_breakpoints never leak into Tool 3.
    if a_true.get("recombination_scan_performed", False):
        recomb_intervals = a_true.get("detected_recombination_windows") or []
    else:
        recomb_intervals = a_true.get("recombination_breakpoints") or []
    recomb_overlap = False
    recomb_hit_frac = 0.0
    if recomb_intervals and seq_len > 0:
        recomb_mask = np.zeros(seq_len, dtype=bool)
        for c0, c1 in recomb_intervals:
            recomb_mask[max(0, int(c0) - 1) : min(seq_len, int(c1))] = True
        f_recomb = float(np.mean(recomb_mask))
        min_enriched_frac = max(0.20, min(0.75, f_recomb * 2.0))

        if clade_sites:
            top_clade_sites = clade_sites[:15]
            in_recomb_hits = sum(
                1
                for m in top_clade_sites
                if any(int(c0) <= int(m["site"]) <= int(c1) for c0, c1 in recomb_intervals)
            )
            recomb_hit_frac = in_recomb_hits / len(top_clade_sites)
            recomb_overlap = (recomb_hit_frac >= min_enriched_frac) and (
                in_recomb_hits >= 2
                or (len(top_clade_sites) <= 3 and in_recomb_hits >= 1 and recomb_hit_frac >= 0.50)
            )
        elif clade_set and len(clade_set) >= 2:
            in_indices = np.flatnonzero(
                np.array([tid in clade_set for tid in taxa_ids], dtype=bool)
            )
            if np.any(recomb_mask) and np.any(~recomb_mask) and len(in_indices) >= 2:
                r0, r1 = int(in_indices[0]), int(in_indices[-1])
                sim_in_blk = float(np.mean(char_mat[r0, recomb_mask] == char_mat[r1, recomb_mask]))
                sim_out_blk = float(np.mean(char_mat[r0, ~recomb_mask] == char_mat[r1, ~recomb_mask]))
                recomb_overlap = sim_in_blk > (sim_out_blk + 0.05)

    is_short_branch = eff_branch_len < 0.005
    is_lba = rho_lba >= 5.0 and (
        clade_has_long_branch or (not leaf_lens and local_ci < 0.65)
    )
    is_indel = (
        (effective_gap_pct >= 12.0 and (has_indel_support or aln_gapped_col_pct >= 12.0))
        or (has_indel_support and aln_gapped_col_pct >= 8.0)
    )

    if recomb_overlap and not (direct_multi_long_branch and recomb_hit_frac < 0.50):
        artifact = "RECOMBINATION"
    elif is_lba:
        artifact = "SHORT_BRANCH_HOMOPLASY_AND_LBA" if is_short_branch else "LBA"
    elif is_indel:
        artifact = "INDEL_GAP"
    elif is_short_branch or local_ci < 0.65:
        artifact = "SHORT_BRANCH_HOMOPLASY"
    else:
        artifact = "NONE"

    return {
        "mean_ci": local_ci,
        "mean_ri": round(mean_inf_ri, 4),
        "lba_branch_ratio": rho_lba,
        "gap_density_pct": round(effective_gap_pct, 2),
        "recombination_window_overlap": bool(recomb_overlap),
        "primary_artifact": artifact,
    }


# ==============================================================================
# TOOL 4: inspect_alignment_window (State 3 — Local Window Slicer)
# ==============================================================================

def inspect_alignment_window(
    fasta_path: str,
    taxa_subset: list[str],
    start_pos: int,
    end_pos: int,
) -> dict:
    """
    Extracts the exact character slice across [start_pos, end_pos] (1-based inclusive)
    for taxa_subset, computing per-column Shannon conservation entropy (bits),
    per-column gap percentages, and polymorphic 1-based column indices.
    """
    taxa_ids, char_mat, seq_len = load_alignment_data(fasta_path)
    s0 = max(0, int(start_pos) - 1)
    e0 = min(seq_len, int(end_pos))

    subset_set = set(taxa_subset)
    sel_indices = [idx for idx, tid in enumerate(taxa_ids) if tid in subset_set]
    if not sel_indices:
        sel_indices = list(range(len(taxa_ids)))

    win_mat = char_mat[sel_indices, s0:e0]
    sub_seqs = {
        taxa_ids[r_idx]: "".join(char_mat[r_idx, s0:e0]) for r_idx in sel_indices
    }

    n_sub = len(sel_indices)
    entropies = []
    gap_pcts = []
    polymorphic_sites = []

    for offset in range(win_mat.shape[1]):
        col_chars = win_mat[:, offset]
        gap_pct_col = round(float(np.sum(col_chars == "-") / n_sub) * 100.0, 2)
        gap_pcts.append(gap_pct_col)

        unique_non_gap = set(col_chars.tolist()) - AMBIGUOUS_SET
        if len(unique_non_gap) > 1:
            polymorphic_sites.append(s0 + offset + 1)

        _, counts = np.unique(col_chars, return_counts=True)
        probs = counts / n_sub
        entropies.append(round(float(-np.sum(probs * np.log2(probs))), 4))

    mean_gap = round(float(np.mean(gap_pcts)), 2) if gap_pcts else 0.0
    return {
        "window_slice": sub_seqs,
        "start_pos": s0 + 1,
        "end_pos": e0,
        "column_conservation_entropy_bits": entropies,
        "column_gap_percentages": gap_pcts,
        "polymorphic_sites": polymorphic_sites,
        "mean_shannon_entropy_bits": round(float(np.mean(entropies)), 4)
        if entropies
        else 0.0,
        "mean_gap_density_pct": mean_gap,
        "window_gap_pct": mean_gap,
    }


# ==============================================================================
# TOOL 5: verify_claim_coordinates (State 4 — Strict Branch & Prose Gate)
# ==============================================================================

def extract_prose_coordinate_claims(explanation_text: str) -> tuple[set[int], list[tuple[int, str, str]]]:
    """
    Extracts all 1-based site coordinates (e.g., 'site 452', 'sites 505, 464, and 128',
    'site coordinates (88, 773, and 132)', 'positions 116, 68, 120', 'tract 10-12')
    and explicit mutation tokens (e.g., 'A452V', 'G4->C') from natural-language explanation text.
    """
    if not explanation_text:
        return set(), []

    prose_sites: set[int] = set()
    num_list_pat = r"\d+(?!\.\d|\s*%)(?:\s*(?:,\s*(?:and\s+)?|\s+and\s+|\s*&\s*)\d+(?!\.\d|\s*%))*"

    # Direct 'site(s) / site coordinate(s) / position(s) / column(s) [such as / include] <list>'
    for m in re.finditer(
        rf"\b(?:site(?:\s+coordinate)?s?|positions?|columns?)\s*(?:is|are|include|includes|such\s+as|namely|:)?\s*\(?({num_list_pat})\)?",
        explanation_text,
        flags=re.IGNORECASE,
    ):
        for num_str in re.findall(r"\d+", m.group(1)):
            prose_sites.add(int(num_str))

    # Clause-bridged phrasings such as 'Verified site coordinates from Tool 2 Output include 540, 995' or 'site coordinate ... is 205'
    for m in re.finditer(
        rf"\b(?:site(?:\s+coordinate)?s?|positions?|columns?)\b(?:[^.;:\d]|Tool\s+\d+){{1,60}}?\b(?:is|are|include|includes|such\s+as|namely)\s*\(?({num_list_pat})\)?",
        explanation_text,
        flags=re.IGNORECASE,
    ):
        for num_str in re.findall(r"\d+", m.group(1)):
            prose_sites.add(int(num_str))

    # Indel tract bounds 'tract 10-12'
    for m in re.finditer(r"\btract\s+(\d+)\s*-\s*(\d+)\b", explanation_text, flags=re.IGNORECASE):
        prose_sites.add(int(m.group(1)))
        prose_sites.add(int(m.group(2)))

    # Mutation tokens such as G4C, A452V, G4->C (excluding node IDs like N12 which lack trailing state)
    prose_mutations: list[tuple[int, str, str]] = []
    valid_bio_chars = set("ACGTUARNDCQEGHILKMFPSTWYV-")
    for m in re.finditer(r"\b([A-Z-])(?:-?(\d+)(?:->|>)?)([A-Z-])\b", explanation_text):
        s_u, pos_str, s_v = m.group(1), m.group(2), m.group(3)
        if s_u in valid_bio_chars and s_v in valid_bio_chars:
            pos = int(pos_str)
            prose_sites.add(pos)
            prose_mutations.append((pos, s_u, s_v))

    return prose_sites, prose_mutations


def verify_claim_coordinates(
    draft_claims: list[dict],
    fasta_path: str,
    asr_state_path: str,
    valid_branch_mutations: list[dict] | None = None,
    explanation_text: str | None = None,
) -> dict:
    """
    Intercepts every discrete mutation claim in draft_claims AND in explanation_text
    and verifies it strictly against the target branch/clade ground truth (Eq. 4.8).
    Eliminates the global leaf-presence bypass (is_valid_leaf) and prevents duplicate-claim gaming.
    """
    _, char_mat, seq_len = load_alignment_data(fasta_path)

    if valid_branch_mutations is not None:
        source_records = valid_branch_mutations
    else:
        asr_df = load_m_true_df(asr_state_path)
        source_records = asr_df.to_dict(orient="records")

    valid_tuples: set[tuple[int, str, str]] = set()
    valid_sites: set[int] = set()
    indel_sites: set[int] = set()

    for rec in source_records:
        s_pos = int(rec["site"])
        p_st = str(rec["parent_state"]).strip().upper()
        c_st = str(rec["child_state"]).strip().upper()
        valid_sites.add(s_pos)
        valid_tuples.add((s_pos, p_st, c_st))
        if bool(rec.get("is_indel", False)):
            indel_sites.add(s_pos)
            # Also accept generic 'CHAR' <-> '-' synonym when consensus char was recorded
            if p_st == "-":
                valid_tuples.add((s_pos, "-", "CHAR"))
            elif c_st == "-":
                valid_tuples.add((s_pos, "CHAR", "-"))

    # Deduplicate draft_claims by (site, s_u, s_v) so repeated identical claims cannot inflate CCVR
    seen_keys: set[tuple[int, str, str]] = set()
    unique_claims: list[dict] = []
    for claim in draft_claims:
        site = int(claim.get("site_coordinate", claim.get("site", -1)))
        s_u = str(claim.get("ancestral_state", claim.get("parent_state", ""))).strip().upper()
        s_v = str(claim.get("derived_state", claim.get("child_state", ""))).strip().upper()
        key = (site, s_u, s_v)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_claims.append(claim)

    mismatched: list[dict] = []
    total_evaluated = 0

    for claim in unique_claims:
        total_evaluated += 1
        site = int(claim.get("site_coordinate", claim.get("site", -1)))
        s_u = str(claim.get("ancestral_state", claim.get("parent_state", ""))).strip().upper()
        s_v = str(claim.get("derived_state", claim.get("child_state", ""))).strip().upper()

        if site < 1 or site > seq_len:
            mismatched.append({**claim, "reason": "Coordinate out of alignment bounds"})
            continue

        col_states = set(char_mat[:, site - 1].tolist())
        is_exact_match = (site, s_u, s_v) in valid_tuples
        is_valid_indel = (
            site in indel_sites
            and "-" in col_states
            and ("-" in {s_u, s_v})
            and s_u != s_v
        )

        if not (is_exact_match or is_valid_indel):
            mismatched.append(
                {
                    **claim,
                    "reason": "Mutation claim not verified in branch ASR/parsimony ground truth",
                }
            )

    # Verify coordinates & mutations cited in natural-language explanation_text
    if explanation_text:
        prose_sites, prose_muts = extract_prose_coordinate_claims(explanation_text)
        checked_prose_sites: set[int] = set()
        for p_site, p_u_raw, p_v_raw in prose_muts:
            p_u, p_v = str(p_u_raw).strip().upper(), str(p_v_raw).strip().upper()
            checked_prose_sites.add(p_site)
            if (p_site, p_u, p_v) not in seen_keys:
                total_evaluated += 1
                if (p_site < 1 or p_site > seq_len) or (
                    (p_site, p_u, p_v) not in valid_tuples
                    and not (p_site in indel_sites and "-" in {p_u, p_v} and p_u != p_v)
                ):
                    mismatched.append(
                        {
                            "source": "explanation_text",
                            "site_coordinate": p_site,
                            "ancestral_state": p_u,
                            "derived_state": p_v,
                            "reason": "Unverified mutation token in explanation prose",
                        }
                    )

        for p_site in prose_sites - checked_prose_sites:
            if not any(int(c.get("site_coordinate", c.get("site", -1))) == p_site for c in unique_claims):
                total_evaluated += 1
                if p_site < 1 or p_site > seq_len or p_site not in valid_sites:
                    mismatched.append(
                        {
                            "source": "explanation_text",
                            "site_coordinate": p_site,
                            "reason": "Unverified site coordinate cited in explanation prose",
                        }
                    )

    # Fair empty-claim handling: if valid_branch_mutations is explicitly non-empty and 0 claims were made,
    # CCVR = 0.0 (cannot game CCVR=1.0 by refusing to cite available evidence). If branch has 0 mutations, CCVR = 1.0.
    if total_evaluated == 0:
        if valid_branch_mutations is not None and len(valid_branch_mutations) > 0:
            return {
                "all_verified": False,
                "mismatched_claims": [
                    {"reason": "No coordinate claims cited despite available branch mutations"}
                ],
                "ccvr_score": 0.0,
                "total_claims_evaluated": 0,
                "verified_claims_count": 0,
            }
        return {
            "all_verified": True,
            "mismatched_claims": [],
            "ccvr_score": 1.0,
            "total_claims_evaluated": 0,
            "verified_claims_count": 0,
        }

    verified_count = max(total_evaluated - len(mismatched), 0)
    return {
        "all_verified": len(mismatched) == 0,
        "mismatched_claims": mismatched,
        "ccvr_score": round(verified_count / total_evaluated, 4),
        "total_claims_evaluated": total_evaluated,
        "verified_claims_count": verified_count,
    }
