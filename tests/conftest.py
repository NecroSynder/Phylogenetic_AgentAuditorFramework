"""
Session-scoped isolated fixtures so pytest never mutates production files under data/.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from src.preprocessing.ground_truth_gen import run_ground_truth_pipeline
from src.preprocessing.normalizer_wrapper import standardize_and_root_tree


@pytest.fixture(scope="session")
def golden_toy_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """
    Runs Step 1 (run_ground_truth_pipeline) and Step 2 (standardize_and_root_tree) once per
    pytest session inside an isolated temporary directory using data/00_golden_toy/ inputs.
    """
    base = tmp_path_factory.mktemp("golden_toy_session")
    gt_dir = base / "02_ground_truths" / "toy"
    std_prefix = base / "03_standardized_trees" / "toy" / "Fusang"

    fasta_path = str(Path("data/00_golden_toy/toy_alignment.fa").resolve())
    raw_tree_path = str(Path("data/00_golden_toy/T_true.nwk").resolve())

    run_ground_truth_pipeline(
        fasta_path=fasta_path,
        true_tree_path=raw_tree_path,
        out_dir=str(gt_dir),
        outgroup="Taxon_01",
    )

    rooted_nwk = (gt_dir / "T_true_rooted.nwk").read_text().strip()
    standardize_and_root_tree(
        primary_newick_str=rooted_nwk,
        bootstrap_newick_strs=[rooted_nwk] * 10,
        outgroup_taxon="Taxon_01",
        target_model="Fusang",
        output_prefix=str(std_prefix),
    )

    return {
        "fasta_path": fasta_path,
        "raw_tree_path": raw_tree_path,
        "gt_dir": str(gt_dir),
        "true_newick_path": str(gt_dir / "T_true_rooted.nwk"),
        "asr_state_path": str(gt_dir / "M_true.tsv"),
        "stressor_gt_path": str(gt_dir / "A_true.json"),
        "pred_newick_path": f"{std_prefix}_normalized.nwk",
        "pred_meta_path": f"{std_prefix}_metadata.json",
    }
