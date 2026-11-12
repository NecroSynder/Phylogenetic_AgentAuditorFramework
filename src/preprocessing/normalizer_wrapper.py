import json
from pathlib import Path
import numpy as np
import dendropy
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from ete3 import Tree
from skbio import DistanceMatrix

def generate_bootstrap_replicates(fasta_path: str, out_dir: str, b_replicates: int = 100, seed: int = 42) -> list[str]:
    """Samples L alignment columns with replacement using NumPy (B=100)."""
    rng = np.random.default_rng(seed)
    alignment = AlignIO.read(fasta_path, "fasta")
    seq_len = alignment.get_alignment_length()
    char_matrix = np.array([list(str(rec.seq)) for rec in alignment])
    taxa_ids = [rec.id for rec in alignment]

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

def to_skbio_distance_matrix(dist_matrix, taxa_labels: list[str] | None = None) -> DistanceMatrix:
    """
    Wraps a Phyloformer NxN distance matrix in a validated scikit-bio DistanceMatrix
    (Aton et al., 2025). scikit-bio enforces symmetry, a zero diagonal and unique taxon IDs,
    raising DistanceMatrixError otherwise. Float round-off asymmetry (np.allclose) is
    symmetrized first; genuine asymmetry is still rejected.
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
    using Gascuel (1997) variance-minimizing agglomerative updates.

    Implemented in NumPy over scikit-bio DistanceMatrix objects (Aton et al., 2025) and
    validated against the reference FastME BIONJ implementation (Lefort et al., 2015);
    see tests/test_bionj.py.
    """
    dm = to_skbio_distance_matrix(dist_matrix, taxa_labels)
    if dm.shape[0] < 3:
        raise ValueError("BIONJ requires at least 3 taxa.")
    D = np.array(dm.data, dtype=float, copy=True)
    V = np.array(dm.data, dtype=float, copy=True)  # Initial variances proportional to distance
    active_nodes = list(dm.ids)

    while len(active_nodes) > 3:
        r = len(active_nodes)
        row_sums = np.sum(D, axis=1)

        # Compute Q-matrix: Q_{ij} = (r - 2) * D_{ij} - sum_k D_{ik} - sum_k D_{jk}
        Q = (r - 2) * D - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(Q, np.inf)

        # Find pair (i, j) minimizing Q_{ij}
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i > j:
            i, j = j, i

        d_ij = D[i, j]
        v_ij = V[i, j]
        delta_ij = (row_sums[i] - row_sums[j]) / (r - 2)

        # Raw BIONJ branch-length estimates (used unmodified in the Gascuel recurrences)
        l_i_raw = 0.5 * (d_ij + delta_ij)
        l_j_raw = 0.5 * (d_ij - delta_ij)
        # Output branch lengths: negative estimates clamped to epsilon (1e-6)
        l_i = max(l_i_raw, epsilon)
        l_j = max(l_j_raw, epsilon)

        # Gascuel (1997) variance-minimizing lambda weight
        if v_ij == 0.0:
            lam = 0.5
        else:
            lam_num = sum(V[j, k] - V[i, k] for k in range(r) if k != i and k != j)
            lam = 0.5 + lam_num / (2.0 * (r - 2) * v_ij)
            lam = float(np.clip(lam, 0.0, 1.0))

        new_node_str = f"({active_nodes[i]}:{l_i:.6f},{active_nodes[j]}:{l_j:.6f})"

        # Update distances and variances for the new composite node u (Gascuel, 1997):
        #   d_uk = lam*(d_ik - l_i) + (1-lam)*(d_jk - l_j)
        #   v_uk = lam*v_ik + (1-lam)*v_jk - lam*(1-lam)*v_ij
        remaining = [k for k in range(r) if k != i and k != j]
        new_d = [
            lam * (D[i, k] - l_i_raw) + (1.0 - lam) * (D[j, k] - l_j_raw)
            for k in remaining
        ]
        new_v = [
            lam * V[i, k] + (1.0 - lam) * V[j, k] - lam * (1.0 - lam) * v_ij
            for k in remaining
        ]

        # Rebuild shrunk D, V, and active_nodes
        D_rem = D[np.ix_(remaining, remaining)]
        V_rem = V[np.ix_(remaining, remaining)]
        m = len(remaining)
        D = np.zeros((m + 1, m + 1), dtype=float)
        V = np.zeros((m + 1, m + 1), dtype=float)
        D[:m, :m] = D_rem
        D[m, :m] = D[:m, m] = new_d
        V[:m, :m] = V_rem
        V[m, :m] = V[:m, m] = new_v

        active_nodes = [active_nodes[k] for k in remaining] + [new_node_str]

    # Final 3-taxon star resolution
    d01, d02, d12 = D[0, 1], D[0, 2], D[1, 2]
    l0 = max(0.5 * (d01 + d02 - d12), epsilon)
    l1 = max(0.5 * (d01 + d12 - d02), epsilon)
    l2 = max(0.5 * (d02 + d12 - d01), epsilon)
    nwk_str = f"({active_nodes[0]}:{l0:.6f},{active_nodes[1]}:{l1:.6f},{active_nodes[2]}:{l2:.6f});"

    ete_tree = Tree(nwk_str, format=1)
    return ete_tree.write(format=1)

def standardize_and_root_tree(
    primary_newick_str: str,
    bootstrap_newick_strs: list[str],
    outgroup_taxon: str,
    target_model: str,
    output_prefix: str
) -> dict:
    """
    Maps B=100 bootstrap split frequencies via DendroPy, enforces unit lengths for Fusang,
    roots at outgroup_taxon via ETE 3, and assigns deterministic post-order labels (N1..Nk).
    """
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    taxa = dendropy.TaxonNamespace()
    primary_dp = dendropy.Tree.get(data=primary_newick_str, schema="newick", taxon_namespace=taxa)
    boot_trees = dendropy.TreeList(taxon_namespace=taxa)
    for b_str in bootstrap_newick_strs:
        boot_trees.append(dendropy.Tree.get(data=b_str, schema="newick", taxon_namespace=taxa))

    # Map empirical bipartition support [0%, 100%] onto primary topology
    # DendroPy 4: build a SplitDistribution from the bootstrap trees, then write each
    # internal node's split frequency (fraction 0-1) into its label explicitly.
    # ETE 3 (format=0) reads these internal labels back as node.support below.
    split_dist = boot_trees.split_distribution(is_bipartitions_updated=False)
    split_freqs = split_dist.split_frequencies
    primary_dp.encode_bipartitions()
    for dp_node in primary_dp.postorder_internal_node_iter():
        if dp_node is primary_dp.seed_node:
            dp_node.label = None
            continue
        bitmask = dp_node.edge.bipartition.split_bitmask
        dp_node.label = f"{split_freqs.get(bitmask, 0.0):.4f}"

    # suppress_rooting=True drops the "[&U]"/"[&R]" token that ETE 3 cannot parse
    dp_newick = primary_dp.as_string(schema="newick", suppress_rooting=True).strip()
    ete_tree = Tree(dp_newick, format=0)
    ete_tree.set_outgroup(ete_tree & outgroup_taxon)

    branch_lengths_present = (target_model != "Fusang")
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
                "clade_taxa": clade_leaves
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
        "nodes": node_index_map
    }
    with open(out_meta, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata