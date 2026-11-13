"""
Synthetic Benchmark Generator for Tiers 1 and 2 (Manuscript Section 4.2.2, Table 4.1 & Table 4.5).
Uses IQ-TREE 2 integrated AliSim (--alisim; Ly-Trong et al., 2022) paired with:
  - Tier 1: Zipfian indel injection (L_gap in [1, 30]; Fletcher & Yang, 2009; Suvorov et al., 2019),
            gap column proportions G in {5%, 15%, 30%}, and terminal LBA multipliers k in {2, 10, 40}
            across N_taxa in {15, 30, 40, 50}, L = 1000.
  - Tier 2: Radiation-burst short internal branches (l_int < 0.005 substitutions/site),
            extreme gamma rate heterogeneity (alpha = 0.2), LBA, and localized recombinant blocks
            (15% of alignment length L; Castro et al., 2023) across N_taxa in {50, 100, 200}, L in {1000, 2000}.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from ete3 import Tree

from src.preprocessing.ground_truth_gen import run_ground_truth_pipeline


def _resolve_iqtree_bin() -> str:
    return (
        shutil.which("iqtree2", path=str(Path(sys.executable).parent))
        or shutil.which("iqtree2")
        or "iqtree2"
    )


def generate_stressor_tree(
    n_taxa: int,
    tier: int = 1,
    lba_multiplier: float = 2.0,
    base_branch_len: float = 0.03,
    short_internal_len: float = 0.002,
    seed: int = 42,
) -> tuple[Tree, dict]:
    """
    Constructs a rooted binary generative tree T_true with Taxon_01 as the designated outgroup:
      - Tier 1: Internal branches ~ Uniform(0.02, 0.05), and two non-sister terminal lineages
                scaled by lba_multiplier * base_branch_len to induce Felsenstein-zone LBA.
      - Tier 2: Radiation-burst internal branches with l_int = short_internal_len (< 0.005)
                and non-sister long terminal branches scaled by lba_multiplier.
    """
    if n_taxa < 4:
        raise ValueError("n_taxa must be >= 4.")
    rng = np.random.default_rng(seed)

    # Build a balanced/random binary tree over Taxon_02 .. Taxon_{n_taxa}
    ingroup_names = [f"Taxon_{i:02d}" if n_taxa < 100 else f"Taxon_{i:03d}" for i in range(2, n_taxa + 1)]
    outgroup_name = "Taxon_01" if n_taxa < 100 else "Taxon_001"

    def _build_subtree(names: list[str]) -> Tree:
        node = Tree()
        if len(names) == 1:
            node.name = names[0]
            node.dist = float(rng.uniform(base_branch_len * 0.8, base_branch_len * 1.2))
            return node
        mid = len(names) // 2
        left = _build_subtree(names[:mid])
        right = _build_subtree(names[mid:])
        node.add_child(left)
        node.add_child(right)
        if tier == 2:
            node.dist = float(short_internal_len)
        else:
            node.dist = float(rng.uniform(base_branch_len * 0.7, base_branch_len * 1.3))
        return node

    ingroup_Copy = list(ingroup_names)
    rng.shuffle(ingroup_Copy)
    ingroup_tree = _build_subtree(ingroup_Copy)

    root = Tree()
    out_leaf = Tree(name=outgroup_name, dist=float(base_branch_len * 2.0))
    root.add_child(out_leaf)
    root.add_child(ingroup_tree)

    # Root at outgroup_name and assign post-order internal node IDs N1..Nk
    root.set_outgroup(root & outgroup_name)
    k_idx = 1
    internal_nodes = []
    for node in root.traverse("postorder"):
        if not node.is_leaf():
            node.name = f"N{k_idx}"
            k_idx += 1
            if not node.is_root():
                internal_nodes.append(node)

    # Select two non-sister ingroup leaves from distinct subtrees to act as LBA long branches
    ingroup_leaves = [lf for lf in root.iter_leaves() if lf.name != outgroup_name]
    long_leaf_1 = ingroup_leaves[0]
    long_leaf_2 = ingroup_leaves[-1]
    long_len = round(float(base_branch_len * lba_multiplier), 6)
    long_leaf_1.dist = long_len
    long_leaf_2.dist = long_len

    # Ensure separating internal branch has known length for rho_LBA calculation
    if tier == 2:
        for inode in internal_nodes:
            inode.dist = round(float(short_internal_len), 6)
        min_int = float(short_internal_len)
    else:
        # Set at least one separating internal branch to base_branch_len / 2 so rho_LBA scales with k
        sep_len = round(float(base_branch_len * 0.5), 6)
        if internal_nodes:
            internal_nodes[0].dist = sep_len
        min_int = min(float(n.dist) for n in internal_nodes) if internal_nodes else base_branch_len

    rho_lba = round(long_len / max(min_int, 1e-6), 3)
    tree_meta = {
        "outgroup_taxon": outgroup_name,
        "lba_long_taxa": [long_leaf_1.name, long_leaf_2.name],
        "lba_long_branch_length": long_len,
        "min_internal_branch_length": round(min_int, 6),
        "theoretical_rho_lba": rho_lba,
    }
    return root, tree_meta


def run_alisim_simulation(
    tree_nwk_path: str,
    out_fasta_path: str,
    seq_len: int = 1000,
    seq_type: str = "DNA",
    tier: int = 1,
    gamma_alpha: float | None = None,
    seed: int = 42,
) -> None:
    """
    Executes IQ-TREE 2 AliSim (--alisim) to simulate sequence evolution along tree_nwk_path:
      - Tier 1: DNA GTR+G4+I or Protein LG+G4+I
      - Tier 2: DNA GTR+G4{0.2} or Protein LG+G4{0.2} (extreme site-to-site rate heterogeneity)
    """
    iqtree_bin = _resolve_iqtree_bin()
    alpha = gamma_alpha if gamma_alpha is not None else (0.2 if tier == 2 else 0.5)

    if seq_type.upper() == "DNA":
        model_str = (
            f"GTR{{1.0/2.5/1.0/1.0/2.5}}+F{{0.25/0.25/0.25/0.25}}+G4{{{alpha}}}+I{{0.1}}"
            if tier == 1
            else f"GTR{{1.0/2.5/1.0/1.0/2.5}}+F{{0.25/0.25/0.25/0.25}}+G4{{{alpha}}}"
        )
    else:
        model_str = f"LG+G4{{{alpha}}}+I{{0.1}}" if tier == 1 else f"LG+G4{{{alpha}}}"

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "alisim_out"
        cmd = [
            iqtree_bin,
            "--alisim",
            str(prefix),
            "-t",
            str(tree_nwk_path),
            "-m",
            model_str,
            "--length",
            str(seq_len),
            "-af",
            "fasta",
            "-seed",
            str(seed),
            "-quiet",
        ]
        subprocess.run(cmd, check=True)
        sim_fa = Path(f"{prefix}.fa")
        if not sim_fa.exists():
            sim_fa = Path(f"{prefix}.fasta")
        Path(out_fasta_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(sim_fa), str(out_fasta_path))


def inject_zipfian_indels(
    fasta_path: str,
    tree: Tree,
    target_gap_proportion: float = 0.15,
    max_tract_len: int = 30,
    zipf_exponent: float = 1.6,
    seed: int = 42,
) -> list[dict]:
    """
    Injects clade-specific shared derived deletions (indel synapomorphies) along internal branches
    of T_true following a Zipfian tract-length distribution (L_gap in [1, 30]; Fletcher & Yang, 2009;
    Suvorov et al., 2019) until approximately target_gap_proportion of alignment columns contain gaps.
    """
    if target_gap_proportion <= 0.0:
        return []

    rng = np.random.default_rng(seed)
    records = list(SeqIO.parse(fasta_path, "fasta"))
    seq_len = len(records[0].seq)
    char_mat = {rec.id: list(str(rec.seq)) for rec in records}

    # Precompute Zipfian probability distribution over L_gap in [1, max_tract_len]
    lengths = np.arange(1, max_tract_len + 1, dtype=float)
    weights = lengths ** (-zipf_exponent)
    probs = weights / np.sum(weights)

    # Candidate internal clades (excluding root and whole-ingroup clade)
    candidate_nodes = [
        n
        for n in tree.traverse("postorder")
        if (not n.is_leaf()) and (not n.is_root()) and (2 <= len(n.get_leaf_names()) <= len(records) - 2)
    ]
    if not candidate_nodes:
        return []

    target_gap_cols = int(round(target_gap_proportion * seq_len))
    occupied_cols = set()
    injected_events = []

    max_attempts = 200
    attempt = 0
    while len(occupied_cols) < target_gap_cols and attempt < max_attempts:
        attempt += 1
        tract_len = int(rng.choice(np.arange(1, max_tract_len + 1), p=probs))
        remaining_needed = target_gap_cols - len(occupied_cols)
        tract_len = min(tract_len, remaining_needed, max_tract_len)
        if tract_len <= 0:
            break

        # Choose a start site (1-based) with a 2-bp buffer around existing indel tracts
        start_site = int(rng.integers(2, max(3, seq_len - tract_len)))
        end_site = start_site + tract_len - 1
        window_with_buffer = set(range(max(1, start_site - 1), min(seq_len, end_site + 1) + 1))
        if occupied_cols & window_with_buffer:
            continue

        clade_node = candidate_nodes[int(rng.integers(0, len(candidate_nodes)))]
        clade_taxa = sorted(clade_node.get_leaf_names())
        parent_name = clade_node.up.name if (clade_node.up and clade_node.up.name) else "N_ROOT"

        for taxon in clade_taxa:
            for pos_1based in range(start_site, end_site + 1):
                char_mat[taxon][pos_1based - 1] = "-"
                occupied_cols.add(pos_1based)

        injected_events.append(
            {
                "parent_node": parent_name,
                "child_node": clade_node.name,
                "i_start": start_site,
                "i_end": end_site,
                "l_gap": tract_len,
                "clade_taxa": clade_taxa,
            }
        )

    # Write updated FASTA
    updated_records = [
        SeqRecord(Seq("".join(char_mat[rec.id])), id=rec.id, description="")
        for rec in records
    ]
    AlignIO.write(MultipleSeqAlignment(updated_records), fasta_path, "fasta")
    injected_events.sort(key=lambda x: x["i_start"])
    return injected_events


def inject_recombinant_block(
    fasta_path: str,
    tree: Tree,
    recomb_fraction: float = 0.15,
    seq_type: str = "DNA",
    gamma_alpha: float = 0.2,
    seed: int = 42,
) -> list[list[int]]:
    """
    Simulates localized horizontal gene transfer / recombination (15% of alignment length L;
    Table 4.1 & Table 4.5) by creating an SPR-rearranged topology, simulating the recombinant
    block of length int(recomb_fraction * L) via AliSim, and splicing it into [c_start, c_end].
    """
    if recomb_fraction <= 0.0:
        return []

    rng = np.random.default_rng(seed)
    records = list(SeqIO.parse(fasta_path, "fasta"))
    seq_len = len(records[0].seq)
    block_len = max(10, int(round(recomb_fraction * seq_len)))

    c_start = int(rng.integers(int(seq_len * 0.2), int(seq_len * 0.75)))
    c_end = min(seq_len, c_start + block_len - 1)
    actual_len = c_end - c_start + 1

    # Construct an alternate topology by swapping two distant ingroup clades/leaves
    recomb_tree = tree.copy()
    leaves = [lf for lf in recomb_tree.iter_leaves() if not lf.name.endswith("_01") and not lf.name.endswith("_001")]
    if len(leaves) >= 4:
        lf_a, lf_b = leaves[1], leaves[-2]
        lf_a.name, lf_b.name = lf_b.name, lf_a.name

    with tempfile.TemporaryDirectory() as tmpdir:
        recomb_nwk = Path(tmpdir) / "recomb_tree.nwk"
        recomb_fa = Path(tmpdir) / "recomb_block.fa"
        recomb_tree.write(format=1, outfile=str(recomb_nwk))
        run_alisim_simulation(
            tree_nwk_path=str(recomb_nwk),
            out_fasta_path=str(recomb_fa),
            seq_len=actual_len,
            seq_type=seq_type,
            tier=2,
            gamma_alpha=gamma_alpha,
            seed=seed + 1000,
        )
        recomb_recs = {r.id: str(r.seq) for r in SeqIO.parse(str(recomb_fa), "fasta")}

    spliced_records = []
    for rec in records:
        orig_seq = str(rec.seq)
        donor_block = recomb_recs.get(rec.id, orig_seq[c_start - 1 : c_end])
        new_seq = orig_seq[: c_start - 1] + donor_block + orig_seq[c_end:]
        spliced_records.append(SeqRecord(Seq(new_seq), id=rec.id, description=""))

    AlignIO.write(MultipleSeqAlignment(spliced_records), fasta_path, "fasta")
    return [[c_start, c_end]]


def generate_single_synthetic_dataset(
    dataset_id: str,
    tier: int,
    n_taxa: int,
    seq_len: int,
    out_raw_dir: str,
    seq_type: str = "DNA",
    gap_proportion: float = 0.0,
    lba_multiplier: float = 2.0,
    short_internal_len: float = 0.002,
    recomb_fraction: float = 0.0,
    seed: int = 42,
    compute_ground_truth: bool = False,
    out_gt_dir: str | None = None,
) -> dict:
    """
    Generates one complete synthetic stressor dataset (alignment.fa, T_true.nwk, simulation_metadata.json)
    and optionally executes Pipeline Step 1 (run_ground_truth_pipeline) to produce T_true_rooted.nwk,
    M_true.tsv, and A_true.json.
    """
    raw_path = Path(out_raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    tree, tree_meta = generate_stressor_tree(
        n_taxa=n_taxa,
        tier=tier,
        lba_multiplier=lba_multiplier,
        short_internal_len=short_internal_len,
        seed=seed,
    )
    true_nwk_path = raw_path / "T_true.nwk"
    fasta_path = raw_path / "alignment.fa"
    tree.write(format=1, outfile=str(true_nwk_path))

    run_alisim_simulation(
        tree_nwk_path=str(true_nwk_path),
        out_fasta_path=str(fasta_path),
        seq_len=seq_len,
        seq_type=seq_type,
        tier=tier,
        seed=seed,
    )

    indel_events = []
    if gap_proportion > 0.0:
        indel_events = inject_zipfian_indels(
            fasta_path=str(fasta_path),
            tree=tree,
            target_gap_proportion=gap_proportion,
            seed=seed + 100,
        )

    recomb_breakpoints = []
    if recomb_fraction > 0.0:
        recomb_breakpoints = inject_recombinant_block(
            fasta_path=str(fasta_path),
            tree=tree,
            recomb_fraction=recomb_fraction,
            seq_type=seq_type,
            seed=seed + 200,
        )

    sim_meta = {
        "dataset_id": dataset_id,
        "tier": tier,
        "n_taxa": n_taxa,
        "sequence_length": seq_len,
        "sequence_type": seq_type,
        "target_gap_proportion": gap_proportion,
        "lba_multiplier": lba_multiplier,
        "short_internal_branch_length": short_internal_len if tier == 2 else None,
        "recombination_fraction": recomb_fraction,
        "recombination_breakpoints": recomb_breakpoints,
        "simulated_indel_events": indel_events,
        "seed": seed,
        **tree_meta,
    }
    meta_path = raw_path / "simulation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(sim_meta, f, indent=2)

    if compute_ground_truth and seq_type.upper() == "DNA":
        gt_dir = out_gt_dir or str(Path("data/02_ground_truths") / f"tier{tier}" / dataset_id)
        run_ground_truth_pipeline(
            fasta_path=str(fasta_path),
            true_tree_path=str(true_nwk_path),
            out_dir=gt_dir,
            outgroup=tree_meta["outgroup_taxon"],
            recombination_breakpoints=recomb_breakpoints,
        )
        sim_meta["ground_truth_dir"] = gt_dir

    return sim_meta
