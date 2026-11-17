"""
Synthetic Benchmark Generator for Tiers 1 and 2 (Manuscript Section 4.2.2, Table 4.1 & Table 4.5).
Uses IQ-TREE 2 integrated AliSim (--alisim; Ly-Trong et al., 2022) paired with:
  - Tier 1: Native CTMC Zipfian indel evolution (--indel + --indel-size POW{1.6/30},POW{1.6/30};
            Fletcher & Yang, 2009; Ly-Trong et al., 2022), gap column proportions G in {5%, 15%, 30%},
            and terminal LBA multipliers k in {2, 10, 40} across N_taxa in {15, 30, 40, 50}, L = 1000.
  - Tier 2: Radiation-burst short internal branches (l_int < 0.005 substitutions/site),
            extreme gamma rate heterogeneity (alpha = 0.2), LBA, and localized recombinant blocks
            (15% of alignment length L; Castro et al., 2023) across N_taxa in {50, 100, 200}, L in {1000, 2000}.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from ete3 import Tree

from src.preprocessing.ground_truth_gen import run_ground_truth_pipeline
from src.utils.io_cache import resolve_conda_binary


def _expected_zipf_tract_length(zipf_exponent: float = 1.6, max_tract_len: int = 30) -> float:
    lengths = np.arange(1, max_tract_len + 1, dtype=float)
    weights = lengths ** (-zipf_exponent)
    probs = weights / np.sum(weights)
    return float(np.sum(lengths * probs))


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

    ingroup_names = [
        f"Taxon_{i:02d}" if n_taxa < 100 else f"Taxon_{i:03d}"
        for i in range(2, n_taxa + 1)
    ]
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

    ingroup_copy = list(ingroup_names)
    rng.shuffle(ingroup_copy)
    ingroup_tree = _build_subtree(ingroup_copy)

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
        sep_len = round(float(base_branch_len * 0.5), 6)
        if internal_nodes:
            internal_nodes[0].dist = sep_len
        min_int = (
            min(float(n.dist) for n in internal_nodes)
            if internal_nodes
            else base_branch_len
        )

    rho_lba = round(long_len / max(min_int, 1e-6), 3)
    total_tree_len = sum(float(n.dist) for n in root.traverse() if not n.is_root())
    tree_meta = {
        "outgroup_taxon": outgroup_name,
        "lba_long_taxa": [long_leaf_1.name, long_leaf_2.name],
        "lba_long_branch_length": long_len,
        "min_internal_branch_length": round(min_int, 6),
        "theoretical_rho_lba": rho_lba,
        "total_tree_branch_length": round(total_tree_len, 6),
    }
    return root, tree_meta


def _extract_observed_indel_tracts(
    char_mat: np.ndarray, max_tract_len: int = 30
) -> list[dict]:
    """Extracts contiguous gapped column intervals [i_start, i_end] from a simulated MSA."""
    has_gap = np.any(char_mat == "-", axis=0)
    gap_cols = [int(i + 1) for i in np.flatnonzero(has_gap)]
    events = []
    idx = 0
    while idx < len(gap_cols):
        start_s = gap_cols[idx]
        end_s = start_s
        while (
            idx + 1 < len(gap_cols)
            and gap_cols[idx + 1] == end_s + 1
            and (end_s - start_s + 1) < max_tract_len
        ):
            idx += 1
            end_s = gap_cols[idx]
        events.append(
            {
                "i_start": start_s,
                "i_end": end_s,
                "l_gap": end_s - start_s + 1,
            }
        )
        idx += 1
    return events


def run_alisim_simulation(
    tree_nwk_path: str,
    out_fasta_path: str,
    seq_len: int = 1000,
    seq_type: str = "DNA",
    tier: int = 1,
    gamma_alpha: float | None = None,
    target_gap_proportion: float = 0.0,
    max_tract_len: int = 30,
    zipf_exponent: float = 1.6,
    seed: int = 42,
) -> list[dict]:
    """
    Executes IQ-TREE 2 AliSim (--alisim) to simulate sequence evolution along tree_nwk_path:
      - Tier 1: DNA GTR+G4+I or Protein LG+G4+I, with native CTMC Zipfian insertions & deletions
                (--indel <ins>,<del> --indel-size POW{1.6/30},POW{1.6/30}) when target_gap_proportion > 0.
      - Tier 2: DNA GTR+G4{0.2} or Protein LG+G4{0.2} (extreme site-to-site rate heterogeneity).
    Returns observed contiguous indel tracts (empty list when target_gap_proportion == 0).
    """
    iqtree_bin = resolve_conda_binary("iqtree2")
    alpha = gamma_alpha if gamma_alpha is not None else (0.2 if tier == 2 else 0.5)

    if seq_type.upper() == "DNA":
        # Note: IQ-TREE AliSim disallows invariant-site +I simultaneously with --indel;
        # use G4{alpha} when --indel is active and G4{alpha}+I{0.1} when gapless in Tier 1.
        use_inv = (tier == 1) and (target_gap_proportion <= 0.0)
        model_str = (
            f"GTR{{1.0/2.5/1.0/1.0/2.5}}+F{{0.25/0.25/0.25/0.25}}+G4{{{alpha}}}"
            + ("+I{0.1}" if use_inv else "")
        )
    else:
        use_inv = (tier == 1) and (target_gap_proportion <= 0.0)
        model_str = f"LG+G4{{{alpha}}}" + ("+I{0.1}" if use_inv else "")

    t_obj = Tree(str(tree_nwk_path), format=1)
    total_tree_len = max(
        sum(float(n.dist) for n in t_obj.traverse() if not n.is_root()), 0.05
    )

    # Oversimulate root length slightly when CTMC indels are enabled so post-all-gap-stripping
    # yields at least seq_len columns, then slice cleanly to exact seq_len for Table 4.5 parity.
    sim_root_len = seq_len
    indel_args: list[str] = []
    if target_gap_proportion > 0.0:
        g_clamped = min(max(target_gap_proportion, 0.01), 0.60)
        mean_tract = _expected_zipf_tract_length(zipf_exponent, max_tract_len)
        x_load = g_clamped / (2.0 - g_clamped)
        indel_rate = max(x_load / (mean_tract * total_tree_len), 0.005)
        sim_root_len = max(seq_len + 60, int(round(seq_len * 1.15)))
        indel_args = [
            "--indel",
            f"{indel_rate:.6f},{indel_rate:.6f}",
            "--indel-size",
            f"POW{{{zipf_exponent}/{max_tract_len}}},POW{{{zipf_exponent}/{max_tract_len}}}",
        ]

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
            str(sim_root_len),
            *indel_args,
            "-af",
            "fasta",
            "-seed",
            str(seed),
            "-quiet",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sim_fa = Path(f"{prefix}.fa")
        if not sim_fa.exists():
            sim_fa = Path(f"{prefix}.fasta")

        Path(out_fasta_path).parent.mkdir(parents=True, exist_ok=True)
        if target_gap_proportion <= 0.0:
            shutil.copyfile(str(sim_fa), str(out_fasta_path))
            return []

        raw_recs = list(SeqIO.parse(str(sim_fa), "fasta"))
        taxa_ids = [r.id for r in raw_recs]
        char_mat = np.array([list(str(r.seq).upper()) for r in raw_recs], dtype="<U1")
        # Strip columns that became 100% gaps across all extant leaves
        non_empty = ~np.all(char_mat == "-", axis=0)
        char_mat = char_mat[:, non_empty]

        if char_mat.shape[1] >= seq_len:
            char_mat = char_mat[:, :seq_len]
        else:
            # Pad with simulated ungapped columns if extreme deletion shortened MSA below seq_len
            pad_needed = seq_len - char_mat.shape[1]
            pad_cols = char_mat[:, :pad_needed]
            char_mat = np.concatenate([char_mat, pad_cols], axis=1)

        # Guarantee at least one indel event if stochastic draw produced zero gaps on a very short MSA
        if not np.any(char_mat == "-"):
            rng_fb = np.random.default_rng(seed)
            clade_leaves = [lf.name for lf in t_obj.iter_leaves() if lf.name != taxa_ids[0]][:2]
            pos = int(rng_fb.integers(5, max(6, seq_len - 5)))
            for r_i, tid in enumerate(taxa_ids):
                if tid in clade_leaves:
                    char_mat[r_i, pos : min(pos + 2, seq_len)] = "-"

        final_recs = [
            SeqRecord(Seq("".join(char_mat[i])), id=taxa_ids[i], description="")
            for i in range(len(taxa_ids))
        ]
        AlignIO.write(MultipleSeqAlignment(final_recs), str(out_fasta_path), "fasta")
        return _extract_observed_indel_tracts(char_mat, max_tract_len=max_tract_len)


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

    recomb_tree = tree.copy()
    leaves = [
        lf
        for lf in recomb_tree.iter_leaves()
        if not lf.name.endswith("_01") and not lf.name.endswith("_001")
    ]
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
            target_gap_proportion=0.0,
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


def derive_expected_stressors(
    tier: int,
    gap_proportion: float,
    theoretical_rho_lba: float,
    short_internal_len: float,
    recomb_fraction: float,
) -> list[str]:
    """Deterministically records ground-truth active evolutionary stressors for simulation metadata."""
    stressors: list[str] = []
    is_short = (tier == 2) and (short_internal_len < 0.005)
    is_lba = theoretical_rho_lba >= 5.0

    if is_lba and is_short:
        stressors.append("SHORT_BRANCH_HOMOPLASY_AND_LBA")
    elif is_lba:
        stressors.append("LBA")
    elif is_short:
        stressors.append("SHORT_BRANCH_HOMOPLASY")

    if gap_proportion >= 0.10:
        stressors.append("INDEL_GAP")
    if recomb_fraction > 0.0:
        stressors.append("RECOMBINATION")
    return stressors


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

    indel_events = run_alisim_simulation(
        tree_nwk_path=str(true_nwk_path),
        out_fasta_path=str(fasta_path),
        seq_len=seq_len,
        seq_type=seq_type,
        tier=tier,
        target_gap_proportion=gap_proportion,
        seed=seed,
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

    expected_stressors = derive_expected_stressors(
        tier=tier,
        gap_proportion=gap_proportion,
        theoretical_rho_lba=tree_meta["theoretical_rho_lba"],
        short_internal_len=short_internal_len,
        recomb_fraction=recomb_fraction,
    )

    sim_meta = {
        "dataset_id": dataset_id,
        "tier": tier,
        "n_taxa": n_taxa,
        "sequence_length": seq_len,
        "sequence_type": seq_type.upper(),
        "target_gap_proportion": gap_proportion,
        "lba_multiplier": lba_multiplier,
        "short_internal_branch_length": short_internal_len if tier == 2 else None,
        "recombination_fraction": recomb_fraction,
        "recombination_breakpoints": recomb_breakpoints,
        "simulated_indel_events": indel_events,
        "expected_stressors": expected_stressors,
        "seed": seed,
        **tree_meta,
    }
    meta_path = raw_path / "simulation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(sim_meta, f, indent=2)

    if compute_ground_truth:
        gt_dir = out_gt_dir or str(
            Path("data/02_ground_truths") / f"tier{tier}" / dataset_id
        )
        run_ground_truth_pipeline(
            fasta_path=str(fasta_path),
            true_tree_path=str(true_nwk_path),
            out_dir=gt_dir,
            outgroup=tree_meta["outgroup_taxon"],
            recombination_breakpoints=recomb_breakpoints,
            expected_stressors=expected_stressors,
        )
        sim_meta["ground_truth_dir"] = gt_dir

    return sim_meta
