import json
from pathlib import Path
import numpy as np
import dendropy
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from skbio import DistanceMatrix
from skbio.tree import bionj
from ete3 import Tree

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

def distance_matrix_to_bionj_newick(dist_matrix: np.ndarray, taxa_labels: list[str], epsilon: float = 1e-6) -> str:
    """Converts Phyloformer NxN pairwise distance matrix into a clamped BIONJ Newick string."""
    dm = DistanceMatrix(dist_matrix, ids=taxa_labels)
    sk_tree = bionj(dm)
    ete_tree = Tree(str(sk_tree), format=1)
    for node in ete_tree.traverse():
        if node.dist is None or node.dist < 0.0:
            node.dist = epsilon
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
    taxa = dendropy.TaxonNamespace()
    primary_dp = dendropy.Tree.get(data=primary_newick_str, schema="newick", taxon_namespace=taxa)
    boot_trees = dendropy.TreeList(taxon_namespace=taxa)
    for b_str in bootstrap_newick_strs:
        boot_trees.append(dendropy.Tree.get(data=b_str, schema="newick", taxon_namespace=taxa))

    # Map empirical bipartition support [0%, 100%] onto primary topology
    boot_trees.summarize_splits_on_tree(
        primary_dp, is_bipartitions_updated=False, set_edge_support_from_split_support=True
    )

    ete_tree = Tree(primary_dp.as_string(schema="newick").strip(), format=0)
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
            # Scale support to percentage [0, 100]
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