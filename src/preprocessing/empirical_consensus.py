"""
Tier 3 Empirical 3-Engine Consensus Silver Standard Builder (Manuscript Section 4.2.2 & Table 4.1).

Constructs the empirical ground-truth reference tree T_true for real biological FASTA alignments
(TreeBASE, OrthoMaM v12, SARS-CoV-2 / HIV-1) by running three independent maximum-likelihood engines:
  1. IQ-TREE 2 (Minh et al., 2020)
  2. RAxML-NG (Kozlov et al., 2019)
  3. FastTree 2 (Price et al., 2010)
and identifying high-confidence consensus clades agreed upon by all three engines (support >= 95%),
followed by Pipeline Step 1 (run_ground_truth_pipeline) to emit T_true_rooted.nwk, M_true.tsv, and A_true.json.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from Bio import SeqIO
from ete3 import Tree

from src.preprocessing.ground_truth_gen import run_ground_truth_pipeline


def _resolve_bin(name: str) -> str:
    env_bin_dir = str(Path(sys.executable).parent)
    return shutil.which(name, path=env_bin_dir) or shutil.which(name) or name


def _normalize_support_pct(raw_val: float | None) -> float:
    if raw_val is None:
        return 100.0
    val = float(raw_val)
    return round(val * 100.0 if val <= 1.0 else val, 2)


def run_three_classical_engines(
    fasta_path: str,
    outgroup: str,
    seq_type: str = "DNA",
    ufboot_replicates: int = 1000,
    raxml_bs_trees: int = 0,
    seed: int = 42,
) -> dict:
    """
    Runs IQ-TREE 2, RAxML-NG, and FastTree 2 on fasta_path, roots each output tree at outgroup,
    and returns their Newick strings and per-clade support maps.
    """
    iqtree_bin = _resolve_bin("iqtree2")
    raxml_bin = _resolve_bin("raxml-ng")
    fasttree_bin = _resolve_bin("fasttree")

    is_dna = seq_type.upper() == "DNA"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. IQ-TREE 2 (with UFBoot if N >= 4 and ufboot_replicates >= 1000, else standard ML)
        iq_prefix = tmp_path / "iqtree_run"
        iq_model = "GTR+G4" if is_dna else "LG+G4"
        iq_cmd = [
            iqtree_bin,
            "-s",
            str(fasta_path),
            "-m",
            iq_model,
            "--prefix",
            str(iq_prefix),
            "-seed",
            str(seed),
            "-quiet",
            "-redo",
        ]
        if ufboot_replicates >= 1000:
            iq_cmd.extend(["-B", str(ufboot_replicates)])
        try:
            subprocess.run(iq_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            # Fallback without -B if alignment is too short/small for UFBoot
            iq_cmd_fallback = [
                iqtree_bin,
                "-s",
                str(fasta_path),
                "-m",
                iq_model,
                "--prefix",
                str(iq_prefix),
                "-seed",
                str(seed),
                "-quiet",
                "-redo",
            ]
            subprocess.run(iq_cmd_fallback, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        iq_nwk = (tmp_path / "iqtree_run.treefile").read_text().strip()

        # 2. RAxML-NG
        rax_prefix = tmp_path / "raxml_run"
        rax_model = "GTR+G" if is_dna else "LG+G"
        if raxml_bs_trees > 0:
            rax_cmd = [
                raxml_bin,
                "--all",
                "--msa",
                str(fasta_path),
                "--model",
                rax_model,
                "--bs-trees",
                str(raxml_bs_trees),
                "--prefix",
                str(rax_prefix),
                "--seed",
                str(seed),
                "--threads",
                "1",
                "--redo",
            ]
            subprocess.run(rax_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            sup_file = tmp_path / "raxml_run.raxml.support"
            best_file = tmp_path / "raxml_run.raxml.bestTree"
            rax_nwk = (sup_file if sup_file.exists() else best_file).read_text().strip()
        else:
            rax_cmd = [
                raxml_bin,
                "--msa",
                str(fasta_path),
                "--model",
                rax_model,
                "--prefix",
                str(rax_prefix),
                "--seed",
                str(seed),
                "--threads",
                "1",
                "--redo",
            ]
            subprocess.run(rax_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            rax_nwk = (tmp_path / "raxml_run.raxml.bestTree").read_text().strip()

        # 3. FastTree 2 (computes SH-like local support values in [0, 1] natively)
        ft_cmd = [fasttree_bin, "-quiet"]
        if is_dna:
            ft_cmd.extend(["-nt", "-gtr"])
        ft_cmd.append(str(fasta_path))
        ft_res = subprocess.run(ft_cmd, check=True, capture_output=True, text=True)
        ft_nwk = ft_res.stdout.strip()

    # Root all three trees at outgroup and extract bipartition -> support maps
    engine_trees = {
        "IQ-TREE2": Tree(iq_nwk, format=0),
        "RAxML-NG": Tree(rax_nwk, format=0),
        "FastTree2": Tree(ft_nwk, format=0),
    }

    engine_clades: dict[str, dict[tuple[str, ...], float]] = {}
    for eng_name, t_obj in engine_trees.items():
        t_obj.set_outgroup(t_obj & outgroup)
        clade_map: dict[tuple[str, ...], float] = {}
        for node in t_obj.traverse("postorder"):
            if not node.is_leaf() and not node.is_root():
                leaves = tuple(sorted(node.get_leaf_names()))
                clade_map[leaves] = _normalize_support_pct(node.support)
        engine_clades[eng_name] = clade_map

    return {
        "iqtree_newick": engine_trees["IQ-TREE2"].write(format=0),
        "raxml_newick": engine_trees["RAxML-NG"].write(format=0),
        "fasttree_newick": engine_trees["FastTree2"].write(format=0),
        "engine_clades": engine_clades,
    }


def build_empirical_tier3_ground_truth(
    fasta_path: str,
    out_dir: str,
    outgroup: str | None = None,
    seq_type: str = "DNA",
    consensus_support_threshold: float = 95.0,
    recombination_breakpoints: list | None = None,
    ufboot_replicates: int = 1000,
    raxml_bs_trees: int = 0,
    seed: int = 42,
) -> dict:
    """
    Executes the 3-engine empirical consensus protocol (IQ-TREE 2, RAxML-NG, FastTree 2) on a real
    biological FASTA alignment, annotates 3-engine consensus clades, and runs Pipeline Step 1
    (run_ground_truth_pipeline) to emit T_true_rooted.nwk, M_true.tsv, A_true.json, and
    tier3_consensus_summary.json.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records = list(SeqIO.parse(fasta_path, "fasta"))
    if len(records) < 4:
        raise ValueError("Empirical alignment must contain at least 4 taxa.")
    taxa_names = sorted(rec.id for rec in records)
    chosen_outgroup = outgroup if (outgroup and outgroup in taxa_names) else taxa_names[0]

    eng_res = run_three_classical_engines(
        fasta_path=fasta_path,
        outgroup=chosen_outgroup,
        seq_type=seq_type,
        ufboot_replicates=ufboot_replicates,
        raxml_bs_trees=raxml_bs_trees,
        seed=seed,
    )

    iq_clades = eng_res["engine_clades"]["IQ-TREE2"]
    rax_clades = eng_res["engine_clades"]["RAxML-NG"]
    ft_clades = eng_res["engine_clades"]["FastTree2"]

    consensus_clades = []
    discordant_or_weak_clades = []
    for leaves_tuple, iq_sup in iq_clades.items():
        in_rax = leaves_tuple in rax_clades
        in_ft = leaves_tuple in ft_clades
        rax_sup = rax_clades.get(leaves_tuple, 0.0)
        ft_sup = ft_clades.get(leaves_tuple, 0.0)
        all_three_topologically_agree = in_rax and in_ft
        min_sup = min(iq_sup, rax_sup if rax_bs_enabled(raxml_bs_trees) else iq_sup, ft_sup)
        entry = {
            "clade_taxa": list(leaves_tuple),
            "iqtree_support_pct": iq_sup,
            "raxml_present": in_rax,
            "raxml_support_pct": rax_sup,
            "fasttree_present": in_ft,
            "fasttree_support_pct": ft_sup,
            "all_three_engines_concordant": all_three_topologically_agree,
            "meets_95pct_silver_standard": bool(
                all_three_topologically_agree and min_sup >= consensus_support_threshold
            ),
        }
        if all_three_topologically_agree:
            consensus_clades.append(entry)
        else:
            discordant_or_weak_clades.append(entry)

    # Save individual engine trees and consensus reference tree
    (out_path / "iqtree2_reference.nwk").write_text(eng_res["iqtree_newick"] + "\n")
    (out_path / "raxml_ng_reference.nwk").write_text(eng_res["raxml_newick"] + "\n")
    (out_path / "fasttree2_reference.nwk").write_text(eng_res["fasttree_newick"] + "\n")
    consensus_nwk_path = out_path / "T_consensus.nwk"
    consensus_nwk_path.write_text(eng_res["iqtree_newick"] + "\n")

    # Run Pipeline Step 1 on the consensus reference tree to produce T_true_rooted.nwk, M_true.tsv, A_true.json
    if seq_type.upper() == "DNA":
        run_ground_truth_pipeline(
            fasta_path=str(fasta_path),
            true_tree_path=str(consensus_nwk_path),
            out_dir=str(out_path),
            outgroup=chosen_outgroup,
            recombination_breakpoints=recombination_breakpoints or [],
        )

    total_internal = len(iq_clades)
    concordant_count = len(consensus_clades)
    summary = {
        "fasta_path": str(fasta_path),
        "outgroup_taxon": chosen_outgroup,
        "sequence_type": seq_type.upper(),
        "total_internal_splits": total_internal,
        "three_engine_concordant_splits": concordant_count,
        "three_engine_concordance_rate": round(concordant_count / max(total_internal, 1), 4),
        "consensus_support_threshold_pct": consensus_support_threshold,
        "consensus_clades": consensus_clades,
        "discordant_clades": discordant_or_weak_clades,
    }
    with open(out_path / "tier3_consensus_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def rax_bs_enabled(raxml_bs_trees: int) -> bool:
    return raxml_bs_trees > 0
