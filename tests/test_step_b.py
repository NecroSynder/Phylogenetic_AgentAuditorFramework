"""
Unit tests for Step B:
  1. Tier 3 Empirical 3-Engine Consensus Silver Standard (IQ-TREE 2 + RAxML-NG + FastTree 2)
  2. Target Model & Classical Baseline Batch Inference + Rules 1-3 Standardization (Fusang, Phyloformer, FastTree2)
"""
import json
from pathlib import Path

from scripts.sanitize_tier3_fasta import sanitize_tier3_empirical_fasta
from src.preprocessing.empirical_consensus import build_empirical_tier3_ground_truth
from src.preprocessing.synthetic_generator import generate_single_synthetic_dataset
from src.preprocessing.target_model_runner import run_target_model_pipeline


def test_tier3_fasta_compatibility_sanitizer(tmp_path: Path) -> None:
    raw_messy = tmp_path / "messy_ncbi.fasta"
    clean_out = tmp_path / "tier3_clean" / "alignment.fa"

    # Raw FASTA with Newick-breaking punctuation, lowercase, U bases, ?/. gaps, and a 100%-gap column (col 4)
    raw_messy.write_text(
        ">Outgroup:Bat_CoV (RaTG13) isolate=1\n"
        "acgu-A\n"
        ">SARS-CoV-2 (Wuhan:2019), strain A\n"
        "acgt-A\n"
        ">SARS-CoV-2 (XBB.1.5); recombinant\n"
        "acg.-C\n"
        ">SARS-CoV-2 [BA.2] variant\n"
        "acg?-C\n"
    )

    summary = sanitize_tier3_empirical_fasta(
        input_fasta_path=str(raw_messy),
        output_fasta_path=str(clean_out),
        outgroup="Outgroup:Bat_CoV",
    )

    assert clean_out.exists()
    assert summary["purpose"] == "Tier 3 Empirical Biological FASTA Compatibility Pre-Processing Only"
    assert summary["n_taxa"] == 4
    assert summary["original_alignment_length"] == 6
    assert summary["removed_all_gap_columns"] == 1
    assert summary["final_alignment_length"] == 5
    assert summary["requested_outgroup_found"] is True
    assert summary["recommended_outgroup"] == "Outgroup_Bat_CoV"


def test_empirical_tier3_three_engine_consensus(tmp_path: Path) -> None:
    raw_dir = tmp_path / "empirical_sample"
    gt_dir = tmp_path / "gt_tier3"

    # Simulate a clean 8-taxon alignment to act as a sample FASTA input for the 3-engine consensus protocol
    generate_single_synthetic_dataset(
        dataset_id="SAMPLE_EMPIRICAL_8TAXA",
        tier=1,
        n_taxa=8,
        seq_len=300,
        out_raw_dir=str(raw_dir),
        seq_type="DNA",
        gap_proportion=0.05,
        lba_multiplier=2.0,
        seed=123,
        compute_ground_truth=False,
    )

    summary = build_empirical_tier3_ground_truth(
        fasta_path=str(raw_dir / "alignment.fa"),
        out_dir=str(gt_dir),
        outgroup="Taxon_01",
        seq_type="DNA",
        ufboot_replicates=0,  # fast ML search for unit test speed
        raxml_bs_trees=0,
        recombination_breakpoints=[[100, 140]],
    )

    assert (gt_dir / "iqtree2_reference.nwk").exists()
    assert (gt_dir / "raxml_ng_reference.nwk").exists()
    assert (gt_dir / "fasttree2_reference.nwk").exists()
    assert (gt_dir / "T_consensus.nwk").exists()
    assert (gt_dir / "T_true_rooted.nwk").exists()
    assert (gt_dir / "M_true.tsv").exists()
    assert (gt_dir / "A_true.json").exists()
    assert (gt_dir / "tier3_consensus_summary.json").exists()

    assert summary["total_internal_splits"] >= 5
    assert 0.0 <= summary["three_engine_concordance_rate"] <= 1.0


def test_target_model_runner_fusang_phyloformer_and_fasttree(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_10taxa"
    std_dir = tmp_path / "std_out"

    generate_single_synthetic_dataset(
        dataset_id="TEST_10TAXA",
        tier=1,
        n_taxa=10,
        seq_len=250,
        out_raw_dir=str(raw_dir),
        seq_type="DNA",
        gap_proportion=0.10,
        lba_multiplier=5.0,
        seed=77,
        compute_ground_truth=False,
    )
    fasta_path = str(raw_dir / "alignment.fa")

    # 1. Fusang (discrete topology, unit branch lengths, B=5 replicates for fast test)
    fusang_prefix = str(std_dir / "Fusang")
    fusang_meta = run_target_model_pipeline(
        fasta_path=fasta_path,
        target_model="Fusang",
        output_prefix=fusang_prefix,
        outgroup_taxon="Taxon_01",
        b_replicates=5,
        seed=42,
    )
    assert fusang_meta is not None
    assert fusang_meta["branch_lengths_present"] is False
    assert Path(f"{fusang_prefix}_normalized.nwk").exists()
    assert Path(f"{fusang_prefix}_metadata.json").exists()
    for node_info in fusang_meta["nodes"].values():
        assert node_info["branch_length"] == 1.0
        assert 0.0 <= node_info["support_pct"] <= 100.0

    # 2. Phyloformer (NxN distance matrix -> BioNJ -> Rules 1-3 standardization)
    pf_prefix = str(std_dir / "Phyloformer")
    pf_meta = run_target_model_pipeline(
        fasta_path=fasta_path,
        target_model="Phyloformer",
        output_prefix=pf_prefix,
        outgroup_taxon="Taxon_01",
        b_replicates=5,
        seed=42,
    )
    assert pf_meta is not None
    assert pf_meta["branch_lengths_present"] is True
    assert Path(f"{pf_prefix}_distance_matrix.tsv").exists()
    assert Path(f"{pf_prefix}_normalized.nwk").exists()
    assert Path(f"{pf_prefix}_metadata.json").exists()

    # 3. FastTree2 classical baseline
    ft_prefix = str(std_dir / "FastTree2")
    ft_meta = run_target_model_pipeline(
        fasta_path=fasta_path,
        target_model="FastTree2",
        output_prefix=ft_prefix,
        outgroup_taxon="Taxon_01",
        b_replicates=3,
        seed=42,
    )
    assert ft_meta is not None
    assert ft_meta["branch_lengths_present"] is True
    assert Path(f"{ft_prefix}_normalized.nwk").exists()


def test_fusang_skips_when_taxa_exceed_40(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_45taxa"
    generate_single_synthetic_dataset(
        dataset_id="TEST_45TAXA",
        tier=2,
        n_taxa=45,
        seq_len=150,
        out_raw_dir=str(raw_dir),
        seq_type="DNA",
        seed=88,
        compute_ground_truth=False,
    )
    res = run_target_model_pipeline(
        fasta_path=str(raw_dir / "alignment.fa"),
        target_model="Fusang",
        output_prefix=str(tmp_path / "Fusang_45"),
        outgroup_taxon="Taxon_01",
        b_replicates=2,
    )
    assert res is None
