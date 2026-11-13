"""
Unit tests for Step A: Tier 1 & Tier 2 Synthetic Stressor Generator (Manuscript Section 4.2.2, Table 4.1 & Table 4.5).
"""
import json
from pathlib import Path
from Bio import SeqIO
from ete3 import Tree

from src.preprocessing.synthetic_generator import generate_single_synthetic_dataset


def test_tier1_zipfian_indel_and_lba_generation(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_t1"
    gt_dir = tmp_path / "gt_t1"

    meta = generate_single_synthetic_dataset(
        dataset_id="TEST_T1_DNA_N15_L300_G15_K10",
        tier=1,
        n_taxa=15,
        seq_len=300,
        out_raw_dir=str(raw_dir),
        seq_type="DNA",
        gap_proportion=0.15,
        lba_multiplier=10.0,
        seed=42,
        compute_ground_truth=True,
        out_gt_dir=str(gt_dir),
    )

    assert (raw_dir / "alignment.fa").exists()
    assert (raw_dir / "T_true.nwk").exists()
    assert (raw_dir / "simulation_metadata.json").exists()
    assert (gt_dir / "T_true_rooted.nwk").exists()
    assert (gt_dir / "M_true.tsv").exists()
    assert (gt_dir / "A_true.json").exists()

    records = list(SeqIO.parse(str(raw_dir / "alignment.fa"), "fasta"))
    assert len(records) == 15
    assert len(records[0].seq) == 300

    # Verify Zipfian indel injection produced gap tracts in [1, 30]
    assert len(meta["simulated_indel_events"]) >= 1
    for ev in meta["simulated_indel_events"]:
        assert 1 <= ev["l_gap"] <= 30

    # Verify LBA ratio >= 5.0 when k=10
    assert meta["theoretical_rho_lba"] >= 5.0

    with open(gt_dir / "A_true.json") as f:
        a_true = json.load(f)
    assert len(a_true["gap_tracts"]) >= 1
    assert a_true["max_gap_tract_length"] >= 1


def test_tier2_short_branch_and_recombination_generation(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_t2"
    gt_dir = tmp_path / "gt_t2"

    meta = generate_single_synthetic_dataset(
        dataset_id="TEST_T2_DNA_N12_L300_K10_RECOMB15",
        tier=2,
        n_taxa=12,
        seq_len=300,
        out_raw_dir=str(raw_dir),
        seq_type="DNA",
        gap_proportion=0.0,
        lba_multiplier=10.0,
        short_internal_len=0.002,
        recomb_fraction=0.15,
        seed=99,
        compute_ground_truth=True,
        out_gt_dir=str(gt_dir),
    )

    assert (raw_dir / "alignment.fa").exists()
    assert (gt_dir / "A_true.json").exists()

    # Verify short internal branches < 0.005 substitutions/site
    tree = Tree(str(raw_dir / "T_true.nwk"), format=1)
    internal_lengths = [
        float(n.dist) for n in tree.traverse() if (not n.is_leaf()) and (not n.is_root())
    ]
    assert all(l < 0.005 for l in internal_lengths)

    # Verify 15% recombinant block breakpoints are recorded in simulation metadata and A_true.json
    assert len(meta["recombination_breakpoints"]) == 1
    bp_start, bp_end = meta["recombination_breakpoints"][0]
    assert (bp_end - bp_start + 1) == 45  # 15% of 300 bp

    with open(gt_dir / "A_true.json") as f:
        a_true = json.load(f)
    assert a_true["recombination_breakpoints"] == [[bp_start, bp_end]]
