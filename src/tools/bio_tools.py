"""
================================================================================
PIPELINE STEP 3 (PART A): DETERMINISTIC PYTHON BIOINFORMATICS TOOL SUITE
(Manuscript Section 4.1.2 & Table 4.4, Pages 14–17)
================================================================================
Provides the 5 deterministic bioinformatics functions invoked by the 4-State
LangGraph Diagnostic Agent so the LLM never performs ungrounded arithmetic or
raw coordinate guessing:

  - Tool 1: audit_tree_topology         (State 1 — Topological & Confidence Triage)
  - Tool 2: extract_branch_synapomorphies (State 2 — Causal Mutation & Indel Attribution)
  - Tool 3: diagnose_evolutionary_stressor (State 3 — Adversarial Error & Stressor Diagnosis)
  - Tool 4: inspect_alignment_window    (State 3 — Local Alignment Entropy & Gap Inspection)
  - Tool 5: verify_claim_coordinates    (State 4 — Deterministic Self-Verification Gate)
"""
import json
import numpy as np
import pandas as pd
from Bio import AlignIO
from ete3 import Tree


def get_bipartition_dict(tree: Tree) -> dict[frozenset, Tree]:
    """Returns a mapping of sorted leaf-set bipartitions to internal nodes (excluding root)."""
    splits = {}
    for node in tree.traverse("postorder"):
        if not node.is_leaf() and not node.is_root():
            splits[frozenset(node.get_leaf_names())] = node
    return splits


# ==============================================================================
# TOOL 1: audit_tree_topology (State 1 — Equations 4.1, 4.2 & 4.3)
# ==============================================================================

def audit_tree_topology(
    pred_newick_path: str,
    true_newick_path: str,
    pred_meta_path: str,
    support_threshold: float = 85.0,
) -> dict:
    """
    Compares every internal split b_k in B_pred against the ground-truth split set B_true.
    Computes:
      - Normalized Robinson-Foulds distance RF_norm (Eq. 4.1)
      - Global False-Confidence Index FCI_global (Eq. 4.2)
      - Conditional False-Confidence Index FCI_cond (Eq. 4.3)
      - Per-node topological status: CONCORDANT, DISCORDANT_LOW_SUPPORT, or FALSE_CONFIDENCE_ERROR.
    """
    t_pred = Tree(pred_newick_path, format=1)
    t_true = Tree(true_newick_path, format=1)
    with open(pred_meta_path) as f:
        pred_meta = json.load(f)

    b_pred = get_bipartition_dict(t_pred)
    b_true = get_bipartition_dict(t_true)

    pred_keys = set(b_pred.keys())
    true_keys = set(b_true.keys())

    # Eq. 4.1: Normalized Robinson-Foulds Distance
    sym_diff = len(pred_keys - true_keys) + len(true_keys - pred_keys)
    denom = len(pred_keys) + len(true_keys)
    rf_norm = round(sym_diff / denom, 4) if denom > 0 else 0.0

    node_records = []
    false_conf_count = 0
    erroneous_splits_count = len(pred_keys - true_keys)

    for split_set, node in b_pred.items():
        node_id = node.name
        sup_pct = pred_meta["nodes"][node_id]["support_pct"]
        in_true = split_set in true_keys

        if in_true:
            status = "CONCORDANT"
            true_match_node = b_true[split_set].name
        elif sup_pct >= support_threshold:
            status = "FALSE_CONFIDENCE_ERROR"
            false_conf_count += 1
            true_match_node = None
        else:
            status = "DISCORDANT_LOW_SUPPORT"
            true_match_node = None

        parent_id = node.up.name if (node.up and node.up.name) else "N_ROOT"
        node_records.append(
            {
                "node_id": node_id,
                "parent_node_id": parent_id,
                "true_matching_node_id": true_match_node,
                "clade_taxa": sorted(list(split_set)),
                "support_pct": sup_pct,
                "topological_status": status,
            }
        )

    # Eqs. 4.2 & 4.3: Global and Conditional FCI
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
# TOOL 2: extract_branch_synapomorphies (State 2 — ASR & Fitch Indel Lookup)
# ==============================================================================

def extract_branch_synapomorphies(
    fasta_path: str,
    asr_state_path: str,
    parent_node_id: str,
    child_node_id: str,
) -> list[dict]:
    """
    Retrieves the verified site coordinates i in {1..L}, ancestral-to-derived state
    transitions (s_u -> s_v), site-wise log-likelihood contributions (delta_SLS_i),
    and contiguous Fitch indel boundaries supporting branch (parent_node_id -> child_node_id).
    """
    df = pd.read_csv(asr_state_path, sep="\t", keep_default_na=False)
    if "is_indel" in df.columns:
        df["is_indel"] = df["is_indel"].apply(
            lambda x: str(x).lower() == "true" if not isinstance(x, bool) else x
        )
    sub = df[
        (df["parent_node"] == parent_node_id) & (df["child_node"] == child_node_id)
    ]
    if sub.empty:
        sub = df[df["child_node"] == child_node_id]
    return (
        sub.sort_values(by="delta_sls", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )


# ==============================================================================
# TOOL 3: diagnose_evolutionary_stressor (State 3 — Stressor Attribution)
# ==============================================================================

def diagnose_evolutionary_stressor(
    fasta_path: str,
    pred_newick_path: str,
    node_id: str,
    stressor_gt_path: str,
) -> dict:
    """
    Determines which evolutionary stressor caused a split to fail or exhibit low CI:
      - LBA Check via rho_LBA >= 5.0 and CI < 0.5
      - Indel & Gap-Masking Check via gap_density_pct >= 15.0%
      - Localized Recombination Check via recombination_breakpoints
      - Short-Branch Homoplasy Check via branch_len < 0.005 or mean_CI < 0.5
    """
    with open(stressor_gt_path) as f:
        a_true = json.load(f)

    t_pred = Tree(pred_newick_path, format=1)
    target_nodes = t_pred.search_nodes(name=node_id)
    branch_len = target_nodes[0].dist if target_nodes else 0.01

    mean_ci = a_true["mean_alignment_ci"]
    mean_ri = a_true["mean_alignment_ri"]
    rho_lba = a_true["global_rho_lba"]
    gap_pct = a_true["mean_gap_density_pct"]
    recomb_overlap = bool(a_true.get("recombination_breakpoints"))

    if rho_lba >= 5.0 and mean_ci < 0.5:
        artifact = "SHORT_BRANCH_HOMOPLASY_AND_LBA" if branch_len < 0.005 else "LBA"
    elif gap_pct >= 15.0:
        artifact = "INDEL_GAP"
    elif recomb_overlap:
        artifact = "RECOMBINATION"
    elif branch_len < 0.005 or mean_ci < 0.5:
        artifact = "SHORT_BRANCH_HOMOPLASY"
    else:
        artifact = "NONE"

    return {
        "mean_ci": mean_ci,
        "mean_ri": mean_ri,
        "lba_branch_ratio": rho_lba,
        "gap_density_pct": gap_pct,
        "recombination_window_overlap": recomb_overlap,
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
    alignment = AlignIO.read(fasta_path, "fasta")
    sub_records = [rec for rec in alignment if rec.id in taxa_subset]
    if not sub_records:
        sub_records = list(alignment)

    sub_seqs = {
        rec.id: str(rec.seq[start_pos - 1 : end_pos]) for rec in sub_records
    }
    entropies = []
    gap_pcts = []
    polymorphic_sites = []
    n_sub = len(sub_records)
    for col_i in range(start_pos - 1, end_pos):
        col_chars = [rec.seq[col_i] for rec in sub_records]
        gap_pct_col = round((col_chars.count("-") / n_sub) * 100.0, 2)
        gap_pcts.append(gap_pct_col)

        unique_non_gap = set(col_chars) - {"-", "N", "?"}
        if len(unique_non_gap) > 1:
            polymorphic_sites.append(col_i + 1)

        _, counts = np.unique(col_chars, return_counts=True)
        probs = counts / n_sub
        entropies.append(round(float(-np.sum(probs * np.log2(probs))), 4))

    mean_gap = round(float(np.mean(gap_pcts)), 2) if gap_pcts else 0.0
    return {
        "window_slice": sub_seqs,
        "start_pos": start_pos,
        "end_pos": end_pos,
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
# TOOL 5: verify_claim_coordinates (State 4 — Deterministic Verification Gate)
# ==============================================================================

def verify_claim_coordinates(
    draft_claims: list[dict],
    fasta_path: str,
    asr_state_path: str,
) -> dict:
    """
    Intercepts every discrete mutation claim (site_coordinate, ancestral_state, derived_state)
    drafted by the LLM and cross-checks it against the raw FASTA alignment and M_true.tsv
    ASR/parsimony table (Eq. 4.8). Blocks any hallucinated coordinates before JSON emission.
    """
    alignment = AlignIO.read(fasta_path, "fasta")
    seq_len = alignment.get_alignment_length()
    asr_df = pd.read_csv(asr_state_path, sep="\t", keep_default_na=False)
    valid_tuples = set(
        zip(asr_df["site"], asr_df["parent_state"], asr_df["child_state"])
    )
    indel_sites = (
        set(asr_df[asr_df["is_indel"].astype(str).str.lower() == "true"]["site"])
        if "is_indel" in asr_df.columns
        else set()
    )

    mismatched = []
    for claim in draft_claims:
        site = int(claim.get("site_coordinate", claim.get("site", -1)))
        s_u = str(claim.get("ancestral_state", claim.get("parent_state", "")))
        s_v = str(claim.get("derived_state", claim.get("child_state", "")))

        if site < 1 or site > seq_len:
            mismatched.append({**claim, "reason": "Coordinate out of alignment bounds"})
            continue

        col_states = set(alignment[:, site - 1])
        is_valid_asr = (site, s_u, s_v) in valid_tuples
        is_valid_leaf = s_u in col_states and s_v in col_states
        is_valid_indel = (
            site in indel_sites and "-" in col_states and ("-" in {s_u, s_v})
        )

        if not (is_valid_asr or is_valid_leaf or is_valid_indel):
            mismatched.append(
                {**claim, "reason": "Character states mismatch FASTA/ASR ground truth"}
            )

    return {
        "all_verified": len(mismatched) == 0,
        "mismatched_claims": mismatched,
        "ccvr_score": round(1.0 - (len(mismatched) / len(draft_claims)), 4)
        if draft_claims
        else 1.0,
    }
