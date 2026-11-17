"""
Unit tests for Step B:
  1. Tier 3 Empirical FASTA Sanitizer (DNA & Protein stop-codon '*' sanitization)
  2. Tier 3 3-Engine Consensus Silver Standard (-m MFP & verified_consensus_splits masking)
  3. Target Model Runner (--use-surrogate guardrail, 20-state AA Phyloformer math, parallel FastTree2)
"""
from pathlib import Path
import pytest

from scripts.sanitize_tier3_fasta import sanitize_tier3_empirical_fasta
from src.preprocessing.empirical_consensus import build_empirical_tier3_ground_truth
from src.preprocessing.synthetic_generator import generate_single_synthetic_dataset
from src.preprocessing.target_model_runner import run_target_model_pipeline


def test_tier3_fasta_sanitizer_dna_and_protein_stop_codons(tmp_path: Path) -> None:
    # 1. DNA FASTA
    raw_dna = tmp_path / "messy_ncbi.fasta"
    clean_dna = tmp_path / "tier3_dna" / "alignment.fa"
    raw_dna.write_text(
        ">Outgroup:Bat_CoV (RaTG13) isolate=1\nacgu-A\n"
        ">SARS-CoV-2 (Wuhan:2019), strain A\nacgt-A\n"
        ">SARS-CoV-2 (XBB.1.5); recombinant\nacg.-C\n"
        ">SARS-CoV-2 [BA.2] variant\nacg?-C\n"
    )
    sum_dna = sanitize_tier3_empirical_fasta(
        str(raw_dna), str(clean_dna), outgroup="Outgroup:Bat_CoV"
    )
    assert sum_dna["sequence_type"] == "DNA"
    assert sum_dna["removed_all_gap_columns"] == 1
    assert sum_dna["final_alignment_length"] == 5

    # 2. Protein (AA) FASTA containing terminal stop codons '*' and non-IUPAC 'J'
    raw_aa = tmp_path / "messy_orthomam.fa"
    clean_aa = tmp_path / "tier3_aa" / "alignment.fa"
    raw_aa.write_text(
        ">Human (BRCA1)\nMKWVTFISLL*\n"
        ">Chimp (BRCA1)\nMKWVTFISLL*\n"
        ">Mouse (Brca1)\nMKWVTFIJLL*\n"
        ">Platypus:Outgroup\nMKWVAFIPLL*\n"
    )
    sum_aa = sanitize_tier3_empirical_fasta(
        str(raw_aa), str(clean_aa), outgroup="Platypus:Outgroup"
    )
    assert sum_aa["sequence_type"] == "AA"
    # Trailing '*' column became 100% '-' and was stripped cleanly; 'J' became 'X'
    assert sum_aa["removed_all_gap_columns"] == 1
    assert sum_aa["final_alignment_length"] == 10
    text_aa = clean_aa.read_text()
    assert "*" not in text_aa and "J" not in text_aa and "X" in text_aa


def test_empirical_tier3_consensus_and_split_masking(tmp_path: Path) -> None:
    raw_dir = tmp_path / "empirical_sample"
    gt_dir = tmp_path / "gt_tier3"

    generate_single_synthetic_dataset(
        dataset_id="SAMPLE_EMPIRICAL_8TAXA",
        tier=1,
        n_taxa=8,
        seq_len=250,
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
        seq_type="AUTO",
        ufboot_replicates=0,
        raxml_bs_trees=0,
        use_modelfinder=False,
        recombination_breakpoints=[[100, 140]],
    )
    assert summary["sequence_type"] == "DNA"
    assert (gt_dir / "A_true.json").exists()


def test_target_model_runner_guardrail_and_aa_phyloformer(tmp_path: Path) -> None:
    raw_aa_dir = tmp_path / "raw_aa_10taxa"
    std_dir = tmp_path / "std_out"

    generate_single_synthetic_dataset(
        dataset_id="TEST_AA_10TAXA",
        tier=1,
        n_taxa=10,
        seq_len=160,
        out_raw_dir=str(raw_aa_dir),
        seq_type="AA",
        gap_proportion=0.05,
        lba_multiplier=4.0,
        seed=77,
        compute_ground_truth=False,
    )
    fasta_path = str(raw_aa_dir / "alignment.fa")

    # 1. Must block silent surrogate execution when allow_surrogate=False
    with pytest.raises(RuntimeError, match="allow_surrogate"):
        run_target_model_pipeline(
            fasta_path=fasta_path,
            target_model="Phyloformer",
            output_prefix=str(std_dir / "PF_Blocked"),
            b_replicates=2,
            allow_surrogate=False,
        )

    # 2. Explicit allow_surrogate=True runs 20-state AA Phyloformer & Fusang cleanly
    fusang_meta = run_target_model_pipeline(
        fasta_path=fasta_path,
        target_model="Fusang",
        output_prefix=str(std_dir / "Fusang"),
        outgroup_taxon="Taxon_01",
        b_replicates=4,
        allow_surrogate=True,
        seed=42,
    )
    assert fusang_meta is not None
    assert fusang_meta["branch_lengths_present"] is False
    assert fusang_meta["inference_backend"] == "surrogate"

    pf_meta = run_target_model_pipeline(
        fasta_path=fasta_path,
        target_model="Phyloformer",
        output_prefix=str(std_dir / "Phyloformer"),
        outgroup_taxon="Taxon_01",
        b_replicates=4,
        allow_surrogate=True,
        seed=42,
    )
    assert pf_meta is not None
    assert pf_meta["branch_lengths_present"] is True

    # 3. Parallel FastTree2 classical baseline on AA alignment
    ft_meta = run_target_model_pipeline(
        fasta_path=fasta_path,
        target_model="FastTree2",
        output_prefix=str(std_dir / "FastTree2"),
        outgroup_taxon="Taxon_01",
        b_replicates=4,
        max_workers=2,
        seed=42,
    )
    assert ft_meta is not None
    assert ft_meta["inference_backend"] == "external_cli"
