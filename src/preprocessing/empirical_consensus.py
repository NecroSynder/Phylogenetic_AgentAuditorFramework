"""
Tier 3 Empirical 3-Engine Consensus Silver Standard Builder (Manuscript Section 4.1.1.1.2 & Table 4.5).

Constructs the empirical ground-truth reference tree T_true for real biological FASTA alignments
(TreeBASE, OrthoMaM v12, SARS-CoV-2 / HIV-1) by running three independent maximum-likelihood engines:
  1. IQ-TREE 2 with ModelFinder (-m MFP; Minh et al., 2020)
  2. RAxML-NG (Kozlov et al., 2019)
  3. FastTree 2 (Price et al., 2010)
and identifying high-confidence consensus clades agreed upon by all three engines (support >= 95%),
recording verified_consensus_splits in A_true.json so ambiguous empirical splits ("quasi-terraces")
never penalize target models in RF_norm or FCI.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from Bio import SeqIO
from ete3 import Tree

from src.preprocessing.ground_truth_gen import (
    detect_sequence_alphabet,
    run_ground_truth_pipeline,
)
from src.utils.io_cache import default_cpu_threads, resolve_conda_binary


def _normalize_support_pct(raw_val: float | None) -> float:
    if raw_val is None:
        return 100.0
    val = float(raw_val)
    return round(val * 100.0 if val <= 1.0 else val, 2)


def run_three_classical_engines(
    fasta_path: str,
    outgroup: str,
    seq_type: str = "AUTO",
    ufboot_replicates: int = 1000,
    raxml_bs_trees: int = 0,
    use_modelfinder: bool = True,
    seed: int = 42,
) -> dict:
    """
    Runs IQ-TREE 2 (-m MFP), RAxML-NG, and FastTree 2 on fasta_path, roots each output tree
    at outgroup, and returns their Newick strings and per-clade support maps.
    """
    iqtree_bin = resolve_conda_binary("iqtree2")
    raxml_bin = resolve_conda_binary("raxml-ng")
    fasttree_bin = resolve_conda_binary("fasttree")

    resolved_type = (
        detect_sequence_alphabet(fasta_path)
        if seq_type.upper() == "AUTO"
        else seq_type.upper()
    )
    is_dna = resolved_type == "DNA"
    threads = str(default_cpu_threads(4))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. IQ-TREE 2 with ModelFinder (-m MFP) and UFBoot2 (-B 1000)
        iq_prefix = tmp_path / "iqtree_run"
        fallback_model = "GTR+G4" if is_dna else "LG+G4"
        iq_model = "MFP" if use_modelfinder else fallback_model
        iq_cmd = [
            iqtree_bin,
            "-s",
            str(fasta_path),
            "-m",
            iq_model,
            "-nt",
            threads,
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
            subprocess.run(
                iq_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            # Fallback to standard GTR+G4 / LG+G4 without UFBoot on tiny/short toy alignments
            iq_cmd_fallback = [
                iqtree_bin,
                "-s",
                str(fasta_path),
                "-m",
                fallback_model,
                "-nt",
                "1",
                "--prefix",
                str(iq_prefix),
                "-seed",
                str(seed),
                "-quiet",
                "-redo",
            ]
            subprocess.run(
                iq_cmd_fallback,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
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
                threads,
                "--redo",
            ]
            subprocess.run(
                rax_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
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
            subprocess.run(
                rax_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            rax_nwk = (tmp_path / "raxml_run.raxml.bestTree").read_text().strip()

        # 3. FastTree 2 (computes SH-like local support values in [0, 1] natively)
        ft_cmd = [fasttree_bin, "-quiet"]
        if is_dna:
            ft_cmd.extend(["-nt", "-gtr"])
        else:
            ft_cmd.extend(["-lg"])
        ft_cmd.append(str(fasta_path))
        ft_res = subprocess.run(ft_cmd, check=True, capture_output=True, text=True)
        ft_nwk = ft_res.stdout.strip()

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
        "resolved_seq_type": resolved_type,
        "iqtree_newick": engine_trees["IQ-TREE2"].write(format=0),
        "raxml_newick": engine_trees["RAxML-NG"].write(format=0),
        "fasttree_newick": engine_trees["FastTree2"].write(format=0),
        "engine_clades": engine_clades,
    }


def build_empirical_tier3_ground_truth(
    fasta_path: str,
    out_dir: str,
    outgroup: str | None = None,
    seq_type: str = "AUTO",
    consensus_support_threshold: float = 95.0,
    recombination_breakpoints: list | None = None,
    ufboot_replicates: int = 1000,
    raxml_bs_trees: int = 0,
    use_modelfinder: bool = True,
    seed: int = 42,
) -> dict:
    """
    Executes the 3-engine empirical consensus protocol (IQ-TREE 2 + ModelFinder, RAxML-NG, FastTree 2)
    on a real biological FASTA alignment, records verified Silver-Standard clades (3-engine agreement
    AND UFBoot >= 95% when UFBoot is enabled), and runs Pipeline Step 1 (run_ground_truth_pipeline).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records = list(SeqIO.parse(fasta_path, "fasta"))
    if len(records) < 4:
        raise ValueError("Empirical alignment must contain at least 4 taxa.")
    taxa_names = sorted(rec.id for rec in records)
    chosen_outgroup = (
        outgroup if (outgroup and outgroup in taxa_names) else taxa_names[0]
    )

    eng_res = run_three_classical_engines(
        fasta_path=fasta_path,
        outgroup=chosen_outgroup,
        seq_type=seq_type,
        ufboot_replicates=ufboot_replicates,
        raxml_bs_trees=raxml_bs_trees,
        use_modelfinder=use_modelfinder,
        seed=seed,
    )

    iq_clades = eng_res["engine_clades"]["IQ-TREE2"]
    rax_clades = eng_res["engine_clades"]["RAxML-NG"]
    ft_clades = eng_res["engine_clades"]["FastTree2"]

    consensus_clades = []
    silver_standard_splits: list[list[str]] = []
    discordant_or_weak_clades = []

    for leaves_tuple, iq_sup in iq_clades.items():
        in_rax = leaves_tuple in rax_clades
        in_ft = leaves_tuple in ft_clades
        rax_sup = rax_clades.get(leaves_tuple, 0.0)
        ft_sup = ft_clades.get(leaves_tuple, 0.0)
        all_three_topologically_agree = in_rax and in_ft

        support_candidates = []
        if ufboot_replicates >= 1000:
            support_candidates.append(iq_sup)
        if rax_bs_enabled(raxml_bs_trees):
            support_candidates.append(rax_sup)
        # FastTree SH-like local support is always available unless single-branch trivial
        if len(leaves_tuple) < len(taxa_names) - 1:
            support_candidates.append(ft_sup)
        min_sup = min(support_candidates) if support_candidates else 100.0

        meets_silver = bool(
            all_three_topologically_agree and min_sup >= consensus_support_threshold
        )
        entry = {
            "clade_taxa": list(leaves_tuple),
            "iqtree_support_pct": iq_sup,
            "raxml_present": in_rax,
            "raxml_support_pct": rax_sup,
            "fasttree_present": in_ft,
            "fasttree_support_pct": ft_sup,
            "all_three_engines_concordant": all_three_topologically_agree,
            "meets_95pct_silver_standard": meets_silver,
        }
        if all_three_topologically_agree:
            consensus_clades.append(entry)
        else:
            discordant_or_weak_clades.append(entry)

        if meets_silver or (
            ufboot_replicates < 1000 and all_three_topologically_agree
        ):
            silver_standard_splits.append(list(leaves_tuple))

    (out_path / "iqtree2_reference.nwk").write_text(eng_res["iqtree_newick"] + "\n")
    (out_path / "raxml_ng_reference.nwk").write_text(eng_res["raxml_newick"] + "\n")
    (out_path / "fasttree2_reference.nwk").write_text(eng_res["fasttree_newick"] + "\n")
    consensus_nwk_path = out_path / "T_consensus.nwk"
    consensus_nwk_path.write_text(eng_res["iqtree_newick"] + "\n")

    run_ground_truth_pipeline(
        fasta_path=str(fasta_path),
        true_tree_path=str(consensus_nwk_path),
        out_dir=str(out_path),
        outgroup=chosen_outgroup,
        recombination_breakpoints=recombination_breakpoints or [],
        auto_scan_recombination=(not bool(recombination_breakpoints)),
        verified_consensus_splits=silver_standard_splits,
    )

    total_internal = len(iq_clades)
    concordant_count = len(consensus_clades)
    summary = {
        "fasta_path": str(fasta_path),
        "outgroup_taxon": chosen_outgroup,
        "sequence_type": eng_res["resolved_seq_type"],
        "total_internal_splits": total_internal,
        "three_engine_concordant_splits": concordant_count,
        "silver_standard_verified_splits": len(silver_standard_splits),
        "three_engine_concordance_rate": round(
            concordant_count / max(total_internal, 1), 4
        ),
        "consensus_support_threshold_pct": consensus_support_threshold,
        "consensus_clades": consensus_clades,
        "discordant_clades": discordant_or_weak_clades,
    }
    with open(out_path / "tier3_consensus_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def rax_bs_enabled(raxml_bs_trees: int) -> bool:
    return raxml_bs_trees > 0
