import json
import pandas as pd
import numpy as np
from Bio import AlignIO
from ete3 import Tree

def get_bipartition_dict(tree: Tree) -> dict[frozenset, Tree]:
    """Returns mapping of sorted leaf-set bipartitions to internal nodes."""
    splits = {}
    for node in tree.traverse("postorder"):
        if not node.is_leaf() and not node.is_root():
            splits[frozenset(node.get_leaf_names())] = node
    return splits

# Tool 1: audit_tree_topology
def audit_tree_topology(
    pred_newick_path: str,
    true_newick_path: str,
    pred_meta_path: str,
    support_threshold: float = 85.0
) -> dict:
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
        node_records.append({
            "node_id": node_id,
            "parent_node_id": parent_id,
            "true_matching_node_id": true_match_node,
            "clade_taxa": sorted(list(split_set)),
            "support_pct": sup_pct,
            "topological_status": status
        })

    # Eqs. 4.2 & 4.3: Global and Conditional FCI
    fci_global = round(false_conf_count / len(pred_keys), 4) if pred_keys else 0.0
    fci_cond = round(false_conf_count / erroneous_splits_count, 4) if erroneous_splits_count > 0 else 0.0

    return {
        "normalized_rf_distance": rf_norm,
        "fci_global": fci_global,
        "fci_conditional": fci_cond,
        "total_internal_nodes": len(pred_keys),
        "false_confidence_nodes_count": false_conf_count,
        "node_bipartitions": node_records
    }

# Tool 2: extract_branch_synapomorphies
def extract_branch_synapomorphies(
    fasta_path: str,
    asr_state_path: str,
    parent_node_id: str,
    child_node_id: str
) -> list[dict]:
    df = pd.read_csv(asr_state_path, sep="\t")
    sub = df[(df["parent_node"] == parent_node_id) & (df["child_node"] == child_node_id)]
    if sub.empty:
        sub = df[df["child_node"] == child_node_id]
    return sub.sort_values(by="delta_sls", ascending=False).head(10).to_dict(orient="records")

# Tool 3: diagnose_evolutionary_stressor
def diagnose_evolutionary_stressor(
    fasta_path: str,
    pred_newick_path: str,
    node_id: str,
    stressor_gt_path: str
) -> dict:
    with open(stressor_gt_path) as f:
        a_true = json.load(f)

    t_pred = Tree(pred_newick_path, format=1)
    target_nodes = t_pred.search_nodes(name=node_id)
    branch_len = target_nodes[0].dist if target_nodes else 0.01

    mean_ci = a_true["mean_alignment_ci"]
    mean_ri = a_true["mean_alignment_ri"]
    rho_lba = a_true["global_rho_lba"]
    gap_pct = a_true["mean_gap_density_pct"]

    if rho_lba >= 5.0 and mean_ci < 0.5:
        artifact = "SHORT_BRANCH_HOMOPLASY_AND_LBA" if branch_len < 0.005 else "LBA"
    elif gap_pct >= 15.0:
        artifact = "INDEL_GAP"
    elif branch_len < 0.005 or mean_ci < 0.5:
        artifact = "SHORT_BRANCH_HOMOPLASY"
    else:
        artifact = "NONE"

    return {
        "mean_ci": mean_ci,
        "mean_ri": mean_ri,
        "lba_branch_ratio": rho_lba,
        "gap_density_pct": gap_pct,
        "recombination_window_overlap": False,
        "primary_artifact": artifact
    }

# Tool 4: inspect_alignment_window
def inspect_alignment_window(
    fasta_path: str,
    taxa_subset: list[str],
    start_pos: int,
    end_pos: int
) -> dict:
    alignment = AlignIO.read(fasta_path, "fasta")
    sub_seqs = {
        rec.id: str(rec.seq[start_pos - 1 : end_pos])
        for rec in alignment if rec.id in taxa_subset
    }
    entropies = []
    for col_i in range(start_pos - 1, end_pos):
        col_chars = [rec.seq[col_i] for rec in alignment if rec.id in taxa_subset]
        _, counts = np.unique(col_chars, return_counts=True)
        probs = counts / len(col_chars)
        entropies.append(-np.sum(probs * np.log2(probs)))

    return {
        "window_slice": sub_seqs,
        "mean_shannon_entropy_bits": round(float(np.mean(entropies)), 4) if entropies else 0.0
    }

# Tool 5: verify_claim_coordinates
def verify_claim_coordinates(
    draft_claims: list[dict],
    fasta_path: str,
    asr_state_path: str
) -> dict:
    alignment = AlignIO.read(fasta_path, "fasta")
    seq_len = alignment.get_alignment_length()
    asr_df = pd.read_csv(asr_state_path, sep="\t")
    valid_tuples = set(zip(asr_df["site"], asr_df["parent_state"], asr_df["child_state"]))

    mismatched = []
    for claim in draft_claims:
        site = int(claim.get("site_coordinate", -1))
        s_u = str(claim.get("ancestral_state", ""))
        s_v = str(claim.get("derived_state", ""))

        if site < 1 or site > seq_len:
            mismatched.append({**claim, "reason": "Coordinate out of alignment bounds"})
            continue

        col_states = set(alignment[:, site - 1])
        if (site, s_u, s_v) not in valid_tuples and (s_u not in col_states or s_v not in col_states):
            mismatched.append({**claim, "reason": "Character states mismatch FASTA/ASR ground truth"})

    return {
        "all_verified": len(mismatched) == 0,
        "mismatched_claims": mismatched,
        "ccvr_score": round(1.0 - (len(mismatched) / len(draft_claims)), 4) if draft_claims else 1.0
    }