"""
================================================================================
PIPELINE STEP 2: PRE-PROCESSING & OUTPUT NORMALIZATION WRAPPER
(Manuscript Section 4.1.1, Item 4 & Pages 11, 14)
================================================================================
Unifies the heterogeneous native outputs of Fusang (<= 40 taxa, branchless
topologies) and Phyloformer (> 40 taxa, NxN pairwise distance matrices) into a
single standardized Newick + JSON representation before passing to the Agent:

  1. Non-Parametric Bootstrap Resampler (generate_bootstrap_replicates):
     - Samples L alignment columns with replacement B = 100 times via NumPy.

  2. Phyloformer BIONJ Agglomerative Converter (distance_matrix_to_bionj_newick):
     - Validates NxN distance matrices via scikit-bio DistanceMatrix (Aton et al., 2025).
     - Applies vectorized Gascuel (1997) variance-minimizing BIONJ updates (validated
       against FastME; Lefort et al., 2015) and clamps negative branches to epsilon = 1e-6.

  3. Tree & Support Normalizer — Rules 1, 2 & 3 (standardize_and_root_tree):
     - Rule 1: Universal Outgroup Rooting via ETE 3 (tree.set_outgroup(outgroup)).
     - Rule 2: Deterministic Post-Order Node Indexing (N1, N2, ..., Nk) and
               sorted leaf-set bipartition signatures.
     - Rule 3: Maps empirical bootstrap split frequencies Support(b) in [0%, 100%]
               via DendroPy (Sukumaran & Holder, 2010) and assigns placeholder unit
               branch lengths (l_k = 1.0) with "branch_lengths_present": false for Fusang.
"""
from __future__ import annotations

import json
from pathlib import Path

import dendropy
import numpy as np
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from ete3 import Tree
from skbio import DistanceMatrix

from src.utils.io_cache import load_alignment_data


# ==============================================================================
# SECTION 1: NON-PARAMETRIC BOOTSTRAP RESAMPLER (B = 100 REPLICATES)
# ==============================================================================

def generate_bootstrap_replicates(
    fasta_path: str,
    out_dir: str,
    b_replicates: int = 100,
    seed: int = 42,
) -> list[str]:
    """Samples L alignment columns with replacement using NumPy (B = 100 replicates)."""
    rng = np.random.default_rng(seed)
    taxa_ids, char_matrix, seq_len = load_alignment_data(fasta_path)

    rep_dir = Path(out_dir) / "bootstrap_reps"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_paths = []

    for b in range(b_replicates):
        col_indices = rng.integers(0, seq_len, size=seq_len)
        resampled_mat = char_matrix[:, col_indices]
        records = [
            SeqRecord(Seq("".join(resampled_mat[idx])), id=taxa_ids[idx], description="")
            for idx in range(len(taxa_ids))
        ]
        rep_file = rep_dir / f"rep_{b:03d}.fasta"
        AlignIO.write(MultipleSeqAlignment(records), str(rep_file), "fasta")
        rep_paths.append(str(rep_file))
    return rep_paths


# ==============================================================================
# SECTION 2: PHYLOFORMER DISTANCE MATRIX VALIDATION & VECTORIZED BIONJ
# ==============================================================================

def to_skbio_distance_matrix(
    dist_matrix,
    taxa_labels: list[str] | None = None,
) -> DistanceMatrix:
    """
    Wraps a Phyloformer NxN distance matrix in a validated scikit-bio DistanceMatrix
    (Aton et al., 2025), enforcing symmetry, a zero diagonal, and unique taxon IDs.
    """
    if isinstance(dist_matrix, DistanceMatrix):
        return dist_matrix
    D = np.asarray(dist_matrix, dtype=float)
    if D.ndim == 2 and D.shape[0] == D.shape[1] and np.allclose(D, D.T, rtol=0.0, atol=1e-9):
        D = (D + D.T) / 2.0
    ids = [str(t) for t in taxa_labels] if taxa_labels is not None else None
    return DistanceMatrix(D, ids=ids)


def distance_matrix_to_bionj_newick(
    dist_matrix: "DistanceMatrix | np.ndarray",
    taxa_labels: list[str] | None = None,
    epsilon: float = 1e-6,
) -> str:
    """
    Converts a Phyloformer NxN pairwise distance matrix into a clamped BIONJ Newick string
    using vectorized Gascuel (1997) variance-minimizing agglomerative updates.
    Validated against reference FastME BIONJ (Lefort et al., 2015); see tests/test_bionj.py.
    """
    dm = to_skbio_distance_matrix(dist_matrix, taxa_labels)
    if dm.shape[0] < 3:
        raise ValueError("BIONJ requires at least 3 taxa.")
    D = np.array(dm.data, dtype=float, copy=True)
    V = np.array(dm.data, dtype=float, copy=True)
    active_nodes = list(dm.ids)

    while len(active_nodes) > 3:
        r = len(active_nodes)
        row_sums = np.sum(D, axis=1)

        # Compute Q-matrix: Q_{ij} = (r - 2) * D_{ij} - sum_k D_{ik} - sum_k D_{jk}
        Q = (r - 2) * D - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(Q, np.inf)

        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i > j:
            i, j = j, i

        d_ij = float(D[i, j])
        v_ij = float(V[i, j])
        delta_ij = float((row_sums[i] - row_sums[j]) / (r - 2))

        l_i_raw = 0.5 * (d_ij + delta_ij)
        l_j_raw = 0.5 * (d_ij - delta_ij)
        l_i = max(l_i_raw, epsilon)
        l_j = max(l_j_raw, epsilon)

        rem_mask = np.ones(r, dtype=bool)
        rem_mask[i] = False
        rem_mask[j] = False

        if v_ij == 0.0:
            lam = 0.5
        else:
            lam_num = float(np.sum(V[j, rem_mask] - V[i, rem_mask]))
            lam = 0.5 + lam_num / (2.0 * (r - 2) * v_ij)
            lam = float(np.clip(lam, 0.0, 1.0))

        new_node_str = f"({active_nodes[i]}:{l_i:.6f},{active_nodes[j]}:{l_j:.6f})"

        new_d = lam * (D[i, rem_mask] - l_i_raw) + (1.0 - lam) * (D[j, rem_mask] - l_j_raw)
        new_v = lam * V[i, rem_mask] + (1.0 - lam) * V[j, rem_mask] - lam * (1.0 - lam) * v_ij

        m = r - 2
        D_next = np.zeros((m + 1, m + 1), dtype=float)
        V_next = np.zeros((m + 1, m + 1), dtype=float)
        D_next[:m, :m] = D[rem_mask][:, rem_mask]
        D_next[m, :m] = D_next[:m, m] = new_d
        V_next[:m, :m] = V[rem_mask][:, rem_mask]
        V_next[m, :m] = V_next[:m, m] = new_v
        D = D_next
        V = V_next

        active_nodes = [
            active_nodes[k] for k in range(r) if k != i and k != j
        ] + [new_node_str]

    d01, d02, d12 = float(D[0, 1]), float(D[0, 2]), float(D[1, 2])
    l0 = max(0.5 * (d01 + d02 - d12), epsilon)
    l1 = max(0.5 * (d01 + d12 - d02), epsilon)
    l2 = max(0.5 * (d02 + d12 - d01), epsilon)
    nwk_str = f"({active_nodes[0]}:{l0:.6f},{active_nodes[1]}:{l1:.6f},{active_nodes[2]}:{l2:.6f});"

    ete_tree = Tree(nwk_str, format=1)
    return ete_tree.write(format=1)


# ==============================================================================
# SECTION 3: TREE & SUPPORT NORMALIZER (RULES 1, 2 & 3)
# ==============================================================================

def standardize_and_root_tree(
    primary_newick_str: str,
    bootstrap_newick_strs: list[str],
    outgroup_taxon: str,
    target_model: str,
    output_prefix: str,
) -> dict:
    """
    Enforces the 3-Rule Standardization Protocol (Manuscript Section 4.1.1 Item 4):
      - Rule 1: Universal Outgroup Rooting at outgroup_taxon via ETE 3.
      - Rule 2: Post-order internal node indexing (N1, N2, ..., Nk) & sorted leaf-set bipartitions.
      - Rule 3: Empirical bootstrap split-frequency mapping [0%, 100%] via DendroPy, plus
                placeholder unit branch lengths (1.0) with branch_lengths_present=False for Fusang.
    """
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    taxa = dendropy.TaxonNamespace()
    primary_dp = dendropy.Tree.get(
        data=primary_newick_str, schema="newick", taxon_namespace=taxa
    )
    boot_trees = dendropy.TreeList(taxon_namespace=taxa)
    for b_str in bootstrap_newick_strs:
        boot_trees.append(
            dendropy.Tree.get(data=b_str, schema="newick", taxon_namespace=taxa)
        )

    split_dist = boot_trees.split_distribution(is_bipartitions_updated=False)
    split_freqs = split_dist.split_frequencies
    primary_dp.encode_bipartitions()
    for dp_node in primary_dp.postorder_internal_node_iter():
        if dp_node is primary_dp.seed_node:
            dp_node.label = None
            continue
        bitmask = dp_node.edge.bipartition.split_bitmask
        dp_node.label = f"{split_freqs.get(bitmask, 0.0):.4f}"

    dp_newick = primary_dp.as_string(schema="newick", suppress_rooting=True).strip()
    ete_tree = Tree(dp_newick, format=0)
    ete_tree.set_outgroup(ete_tree & outgroup_taxon)

    branch_lengths_present = target_model != "Fusang"
    node_index_map = {}
    k_counter = 1

    for node in ete_tree.traverse("postorder"):
        if not branch_lengths_present:
            node.dist = 1.0  # Placeholder unit branch length for Fusang
        elif node.dist < 0:
            node.dist = 1e-6

        if not node.is_leaf():
            node_id = f"N{k_counter}"
            node.name = node_id
            raw_sup = float(node.support) if node.support is not None else 1.0
            sup_pct = round(raw_sup * 100.0 if raw_sup <= 1.0 else raw_sup, 2)
            node.support = sup_pct
            clade_leaves = sorted(node.get_leaf_names())
            node_index_map[node_id] = {
                "node_id": node_id,
                "support_pct": sup_pct,
                "branch_length": round(float(node.dist), 6),
                "clade_taxa": clade_leaves,
            }
            k_counter += 1

    out_nwk = f"{output_prefix}_normalized.nwk"
    out_meta = f"{output_prefix}_metadata.json"
    ete_tree.write(format=1, outfile=out_nwk)

    metadata = {
        "target_model": target_model,
        "branch_lengths_present": branch_lengths_present,
        "outgroup_taxon": outgroup_taxon,
        "total_internal_nodes": k_counter - 1,
        "nodes": node_index_map,
    }
    with open(out_meta, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata
