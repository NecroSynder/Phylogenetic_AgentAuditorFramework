"""
Comprehensive Regression Test Suite for All Empirical Readiness Fixes (Bugs #1 - #15):
  1. DendroPy preserve_underscores=True with hyphenated pathogen IDs (SARS-CoV-2_Wuhan_2019)
  2. Empirical FASTA sanitizer: paired T_true.nwk remapping, empty-sequence row removal,
     duplicate detection, and kept_original_columns_1based coordinate mapping
  3. Branchless reference pedigree ML branch-length transfer + direct in vitro sequenced
     ancestors & cataloged mutations in M_true.tsv
  4. Clade-local rho_LBA & local separating branch length (no polytomy explosion)
  5. State 2 -> State 3 routing threshold (mean_ci < 0.65)
  6. Live LLM retry_count <= max_retries (1 + 3 = 4 calls) & NO silent Python-template overwrite
  7. Non-circular EDR against independent A_true.json stressors
  8. Prose BSF ('low bootstrap support'), FGS anchors, and State 4 coordinate extraction
     ('positions 116, 68', 'site coordinates (88, 773)')
  9. Baseline 2 dynamic polymorphic window selection beyond column 80 & unassisted status
  10. Unbootstrapped Newick support guard in empirical_consensus.py
  11. Mock-row isolation in replicate_level_statistical_suite.json + full Stage A & Stage B suites
  12. Resumable Stage A + Stage B empirical corpus runner (scripts/run_empirical_corpus.py)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.run_ai_baselines import run_three_way_comparison
from scripts.run_empirical_corpus import discover_empirical_datasets
from scripts.sanitize_tier3_fasta import (
    remap_site_coordinate,
    sanitize_tier3_empirical_fasta,
)
from src.agent.ai_baselines import _select_most_informative_raw_window
from src.agent.diagnostic_agent import route_after_state_2, run_diagnostic_audit
from src.evaluation.metrics import (
    compute_bsf_local,
    compute_fgs_local,
    compute_report_edr,
    run_stage_a_statistical_suite,
)
from src.preprocessing.empirical_consensus import (
    _newick_has_internal_support_labels,
    _normalize_support_pct,
)
from src.preprocessing.ground_truth_gen import run_ground_truth_pipeline
from src.preprocessing.normalizer_wrapper import standardize_and_root_tree
from src.tools.bio_tools import (
    diagnose_evolutionary_stressor,
    extract_prose_coordinate_claims,
    verify_claim_coordinates,
)


def test_dendropy_preserves_underscores_with_hyphenated_pathogen_names(tmp_path: Path) -> None:
    """Bug #8: Hyphen + underscore taxon names must never turn into single-quoted space names."""
    nwk = (
        "((SARS-CoV-2_Wuhan_2019:0.01,SARS-CoV-2_XBB_1_5:0.02):0.03,"
        "(HIV-1_Group_M:0.04,Outgroup-Bat_CoV_RaTG13:0.10):0.05);"
    )
    out_prefix = str(tmp_path / "hyphen_norm")
    meta = standardize_and_root_tree(
        primary_newick_str=nwk,
        bootstrap_newick_strs=[nwk] * 5,
        outgroup_taxon="Outgroup-Bat_CoV_RaTG13",
        target_model="Phyloformer",
        output_prefix=out_prefix,
    )
    assert meta["outgroup_taxon"] == "Outgroup-Bat_CoV_RaTG13"
    all_leaves = sorted(meta["nodes"][f"N{meta['total_internal_nodes']}"]["clade_taxa"])
    assert "SARS-CoV-2_Wuhan_2019" in all_leaves
    assert "SARS-CoV-2 Wuhan 2019" not in all_leaves


def test_sanitizer_remaps_published_tree_screens_empty_rows_and_maps_columns(tmp_path: Path) -> None:
    """Bugs #4 & #10 + Extra #1: Paired tree remapping, all-gap row filtering, duplicate detection, col map."""
    raw_fa = tmp_path / "raw_panel.fasta"
    raw_tree = tmp_path / "raw_published.nwk"
    clean_fa = tmp_path / "clean" / "alignment.fa"
    clean_tree = tmp_path / "clean" / "T_true.nwk"

    # Column 2 is 100% '-', so original site 5 shifts to cleaned site 4; 'Ghost:Empty' is 100% gaps/N
    raw_fa.write_text(
        ">Outgroup:Bat-CoV\nA-CGTA\n"
        ">Strain:Alpha.1\nA-CGTG\n"
        ">Strain:Alpha.2\nA-CGTG\n"
        ">Strain:Beta/2024\nA-TTTG\n"
        ">Ghost:Empty\n--NNN-\n"
    )
    raw_tree.write_text(
        "[&R] (('Outgroup:Bat-CoV':0.2,'Ghost:Empty':0.01):0.05,"
        "(('Strain:Alpha.1':0.01,'Strain:Alpha.2':0.01)98:0.04,'Strain:Beta/2024':0.03)95:0.05);\n"
    )

    summary = sanitize_tier3_empirical_fasta(
        input_fasta_path=str(raw_fa),
        output_fasta_path=str(clean_fa),
        outgroup="Outgroup:Bat-CoV",
        input_tree_path=str(raw_tree),
        output_tree_path=str(clean_tree),
    )
    assert summary["n_taxa"] == 4
    assert summary["removed_empty_taxa"] == ["Ghost_Empty"]
    assert summary["has_identical_sequences"] is True
    assert ["Strain_Alpha_1", "Strain_Alpha_2"] in summary["duplicate_sequence_groups"]
    assert summary["kept_original_columns_1based"] == [1, 3, 4, 5, 6]
    assert remap_site_coordinate(5, summary["kept_original_columns_1based"]) == 4
    assert remap_site_coordinate(2, summary["kept_original_columns_1based"]) is None
    assert clean_tree.exists()
    tree_txt = clean_tree.read_text()
    assert "Ghost" not in tree_txt
    assert "Outgroup_Bat-CoV" in tree_txt
    assert "Strain_Beta_2024" in tree_txt


def test_branchless_pedigree_ml_lengths_and_sequenced_ancestors_and_cataloged_mutations(
    tmp_path: Path,
) -> None:
    """Bugs #3a & #11: Branchless T_true populates ML branch lengths; sequenced ancestors & cataloged muts enter M_true."""
    fa = tmp_path / "pedigree.fa"
    nwk = tmp_path / "pedigree_branchless.nwk"
    anc_fa = tmp_path / "ancestors.fa"
    gt_dir = tmp_path / "gt_pedigree"

    # 5-taxon 60-bp alignment with clear synapomorphies
    s_out = "ACGT" * 15
    s_a1 = "TCGT" + ("ACGT" * 14)
    s_a2 = "TCGA" + ("ACGT" * 14)
    s_b1 = "ACGG" + ("ACGT" * 14)
    s_b2 = "ACGG" + ("ACGT" * 13) + "ACGA"
    fa.write_text(
        f">Outgroup\n{s_out}\n>Leaf_A1\n{s_a1}\n>Leaf_A2\n{s_a2}\n>Leaf_B1\n{s_b1}\n>Leaf_B2\n{s_b2}\n"
    )
    # Branchless pedigree with named internal nodes Anc_A and Anc_AB
    nwk.write_text("(Outgroup,((Leaf_A1,Leaf_A2)Anc_A,(Leaf_B1,Leaf_B2)Anc_B)Anc_AB);\n")
    # Sequenced internal ancestors showing exact observed transition C->T at site 20 on Anc_AB -> Anc_A
    anc_ab_seq = list(s_out)
    anc_a_seq = list(s_out)
    anc_ab_seq[19] = "C"  # 1-based site 20
    anc_a_seq[19] = "T"
    anc_fa.write_text(
        f">Anc_AB\n{''.join(anc_ab_seq)}\n>Anc_A\n{''.join(anc_a_seq)}\n"
    )

    a_true = run_ground_truth_pipeline(
        fasta_path=str(fa),
        true_tree_path=str(nwk),
        out_dir=str(gt_dir),
        outgroup="Outgroup",
        experimental_ancestors_fasta=str(anc_fa),
        cataloged_mutations=[
            {
                "clade_taxa": ["Leaf_B1", "Leaf_B2"],
                "site": 31,  # original site 31 -> cleaned site 30 when col 2 was stripped
                "parent_state": "G",
                "child_state": "A",
            }
        ],
        kept_original_columns_1based=[1] + list(range(3, 62)),
    )

    assert a_true["input_tree_had_branch_lengths"] is False
    # ML branch lengths transferred from IQ-TREE must NOT all be placeholder 1.0
    assert any(abs(v - 1.0) > 1e-3 for v in a_true["leaf_branch_lengths"].values())

    m_df = pd.read_csv(gt_dir / "M_true.tsv", sep="\t")
    # Sequenced ancestor mutation at site 20 (C -> T) must be present with posterior_prob == 1.0
    row_20 = m_df[(m_df["site"] == 20) & (m_df["parent_state"] == "C") & (m_df["child_state"] == "T")]
    assert len(row_20) == 1
    assert float(row_20.iloc[0]["posterior_prob"]) == 1.0
    # Cataloged mutation at original site 31 -> mapped site 30 (G -> A) must be present
    row_30 = m_df[(m_df["site"] == 30) & (m_df["parent_state"] == "G") & (m_df["child_state"] == "A")]
    assert len(row_30) == 1


def test_clade_local_rho_lba_does_not_misfire_on_non_lba_clades(golden_toy_env, tmp_path: Path) -> None:
    """Bug #3b: Non-LBA clades must not inherit tree-wide global_rho_lba = 12.0 or trigger false LBA."""
    diag_non_lba = diagnose_evolutionary_stressor(
        fasta_path=golden_toy_env["fasta_path"],
        pred_newick_path=golden_toy_env["pred_newick_path"],
        node_id="N8",  # (Taxon_03, Taxon_04) — normal 0.05 leaves, far from long-branch Taxon_08
        stressor_gt_path=golden_toy_env["stressor_gt_path"],
    )
    assert diag_non_lba["lba_branch_ratio"] < 5.0
    assert diag_non_lba["primary_artifact"] != "LBA"


def test_state_2_routing_threshold_and_live_llm_no_silent_template_overwrite(
    golden_toy_env, monkeypatch
) -> None:
    """Bugs #1, #2, #5: mean_ci < 0.65 routes to State 3; live LLM gets 1+3=4 calls and NO template overwrite."""
    # Bug #5: mean_ci = 0.60 (< 0.65) on a concordant node MUST route to State 3
    route = route_after_state_2(
        {"current_node": {"topological_status": "CONCORDANT"}, "current_mean_ci": 0.60}
    )
    assert route == "state_3_stressor_interrogation"
    route_high = route_after_state_2(
        {"current_node": {"topological_status": "CONCORDANT"}, "current_mean_ci": 0.70}
    )
    assert route_high == "state_4_synthesize_and_verify"

    # Bug #1: Mock live LM Studio + ChatOpenAI persistently hallucinating site 9999 across max_retries=3
    monkeypatch.setattr(
        "src.agent.diagnostic_agent.resolve_lm_studio_url",
        lambda url, timeout=3.0: ("http://mock-lmstudio:1234/v1", ["qwen/qwen3-14b"]),
    )
    llm_call_count = {"n": 0}

    class DummyHallucinatingChatOpenAI:
        def __init__(self, **kwargs):
            pass

        def invoke(self, prompt: str):
            llm_call_count["n"] += 1
            payload = {
                "draft_claims": [
                    {
                        "site_coordinate": 9999,
                        "ancestral_state": "A",
                        "derived_state": "G",
                        "posterior_probability": 0.99,
                        "consistency_index_ci": 1.0,
                        "is_indel": False,
                    }
                ],
                "primary_stressor": "NONE",
                "explanation": "Hallucinated coordinate at positions 9999 with 100% bootstrap support (definitive high confidence).",
            }
            return SimpleNamespace(content=json.dumps(payload))

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", DummyHallucinatingChatOpenAI)

    rep = run_diagnostic_audit(
        dataset_id="live_no_overwrite_check",
        target_model="Fusang",
        fasta_path=golden_toy_env["fasta_path"],
        pred_newick_path=golden_toy_env["pred_newick_path"],
        pred_meta_path=golden_toy_env["pred_meta_path"],
        true_newick_path=golden_toy_env["true_newick_path"],
        asr_state_path=golden_toy_env["asr_state_path"],
        stressor_gt_path=golden_toy_env["stressor_gt_path"],
        orchestrator_llm="qwen/qwen3-14b",
        lm_studio_base_url="http://mock-lmstudio:1234/v1",
        require_live_llm=True,
        allow_offline_mock=False,
        max_retries=3,
    )
    # 13 internal nodes * (1 initial + 3 retries) = 52 live LLM calls!
    assert llm_call_count["n"] == 52
    assert rep["audit_metadata"]["orchestrator_llm"] == "qwen/qwen3-14b"
    # Must NOT be silently overwritten to CCVR = 1.0 by Python template!
    assert rep["global_metrics"]["claim_to_coordinate_verification_rate"] == 0.0
    assert rep["global_metrics"]["overall_tree_goodness_score"] == 0.0


def test_non_circular_edr_penalizes_wrong_stressor_and_prose_regexes(golden_toy_env) -> None:
    """Bugs #2 & #4: EDR checks A_true.json without Tool 3 circularity; BSF/FGS/coordinate regexes work."""
    # 1. Prose regex checks
    sites, _ = extract_prose_coordinate_claims(
        "Verified site coordinates from Tool 2 Output include positions 116, 68, and 120 (100.0% support, CI = 0.55)."
    )
    assert sites == {116, 68, 120}

    assert compute_bsf_local(20.0, "Split has low bootstrap support (20.0%).") == 1
    assert compute_bsf_local(95.0, "Split has high bootstrap support (95.0%).") == 1
    assert compute_bsf_local(20.0, "Split does not have high bootstrap support; shows low bootstrap support.") == 1

    syns = [
        {"site": 88, "parent_state": "C", "child_state": "A", "is_indel": False},
        {"site": 773, "parent_state": "T", "child_state": "C", "is_indel": False},
    ]
    fgs_val = compute_fgs_local(
        "Node N4 is concordant with 100% bootstrap support. The verified site coordinates (88 and 773) show mutations.",
        "N4",
        100.0,
        syns,
        {"primary_artifact": "NONE"},
    )
    assert fgs_val == 1.0

    # Verify State 4 blocks 'positions 9999' in prose even when draft_claims is valid
    ver_bad_pos = verify_claim_coordinates(
        draft_claims=[{"site_coordinate": 88, "ancestral_state": "C", "derived_state": "A"}],
        fasta_path=golden_toy_env["fasta_path"],
        asr_state_path=golden_toy_env["asr_state_path"],
        valid_branch_mutations=syns,
        explanation_text="Verified site coordinates include positions 88 and 9999.",
    )
    assert ver_bad_pos["all_verified"] is False

    # 2. Non-circular EDR check: Golden toy A_true has expected_stressors=["LBA"], long_branch_taxa=["Taxon_08"]
    wrong_audits = [
        {
            "node_id": "N1",
            "predicted_clade_taxa": ["Taxon_08", "Taxon_15"],
            "model_support_pct": 90.0,
            "topological_status": "FALSE_CONFIDENCE_ERROR",
            "diagnosed_failure_mode": {"primary_stressor": "INDEL_GAP"},
        }
    ]
    assert (
        compute_report_edr(
            wrong_audits,
            golden_toy_env["fasta_path"],
            golden_toy_env["pred_newick_path"],
            golden_toy_env["stressor_gt_path"],
        )
        == 0.0
    )


def test_baseline2_dynamic_window_and_consensus_support_guard_and_stage_a_b_stats(
    golden_toy_env, tmp_path: Path
) -> None:
    """Bugs #6, #7, #9, #12, #13: Dynamic window >80, UFBoot support guard, mock isolation, Stage A & corpus runner."""
    # Bug #7: Polymorphic signal placed at columns 220..235 (columns 1..200 invariant)
    far_fa = tmp_path / "far_signal.fa"
    for idx, tid in enumerate(["T1", "T2", "T3", "T4"]):
        prefix_200 = "A" * 200
        poly_blk = ("C" * 20) if tid in ("T1", "T2") else ("G" * 20)
        suffix = "T" * 40
        with open(far_fa, "a") as f:
            f.write(f">{tid}\n{prefix_200}{poly_blk}{suffix}\n")
    w_s, w_e = _select_most_informative_raw_window(str(far_fa), ["T1", "T2"], window_width=80)
    assert w_s <= 205 and w_e >= 220

    # Bug #9: Unbootstrapped Newick must NOT fake 100% support
    unboot_nwk = "((A:0.1,B:0.1):0.05,(C:0.1,D:0.1):0.05);"
    assert _newick_has_internal_support_labels(unboot_nwk) is False
    assert _normalize_support_pct(1.0, has_explicit_support=False) == 0.0
    boot_nwk = "((A:0.1,B:0.1)98:0.05,(C:0.1,D:0.1)96:0.05);"
    assert _newick_has_internal_support_labels(boot_nwk) is True
    assert _normalize_support_pct(98.0, has_explicit_support=True) == 98.0

    # Bug #12: Stage A Wilcoxon + BH-FDR suite (Neural vs Classical ML)
    stage_a_mock_rows = [
        {"dataset_id": f"D{i}", "target_model": "Fusang", "rf_norm": 0.35 + 0.01 * i, "fci_global": 0.20, "fci_conditional": 0.50}
        for i in range(6)
    ] + [
        {"dataset_id": f"D{i}", "target_model": "IQ-TREE2", "rf_norm": 0.05 + 0.01 * i, "fci_global": 0.00, "fci_conditional": 0.00}
        for i in range(6)
    ]
    stage_a_res = run_stage_a_statistical_suite(stage_a_mock_rows)
    assert "Fusang" in stage_a_res
    assert "Fusang_vs_IQ-TREE2" in stage_a_res["Fusang"]
    assert stage_a_res["Fusang"]["Fusang_vs_IQ-TREE2"]["wilcoxon_rf_norm"]["significant"] is True

    # Bug #6: Pre-seed master TSV with a fake live row and ensure offline mock runs don't pool with it
    out_rep = tmp_path / "reports_isolated"
    stat_dir = out_rep / "statistical_comparisons"
    stat_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = stat_dir / "master_benchmark_metrics_table.tsv"
    pd.DataFrame(
        [
            {
                "dataset_id": "LIVE_ONLY_DS",
                "target_model": "Fusang",
                "auditor_method": m_lbl,
                "llm_model_id": "qwen/qwen3-14b",
                "rf_norm": 0.2,
                "fci_global": 0.1,
                "conditional_fci": 0.5,
                "ccvr": 1.0,
                "bsf": 1.0,
                "lcr": 1.0,
                "mean_g_node": 0.9,
                "g_tree": 0.9,
                "mean_feature_f1": 1.0,
                "mean_ig": 1.0,
                "mean_fgs": 1.0,
                "edr": 1.0,
                "total_internal_nodes_audited": 10,
                "false_confidence_nodes_count": 1,
                "total_hallucination_retries_triggered": 0,
                "report_json_path": "nonexistent.json",
            }
            for m_lbl in ("Proposed_4State_Agent", "Baseline1_ZeroShot_LLM", "Baseline2_Generic_Agent")
        ]
    ).to_csv(tsv_path, sep="\t", index=False)

    for ds_tag in ("MOCK_DS_1", "MOCK_DS_2"):
        comp = run_three_way_comparison(
            dataset_id=ds_tag,
            target_model="Fusang",
            fasta_path=golden_toy_env["fasta_path"],
            pred_newick_path=golden_toy_env["pred_newick_path"],
            pred_meta_path=golden_toy_env["pred_meta_path"],
            true_newick_path=golden_toy_env["true_newick_path"],
            asr_state_path=golden_toy_env["asr_state_path"],
            stressor_gt_path=golden_toy_env["stressor_gt_path"],
            out_reports_root=str(out_rep),
            lm_studio_base_url=None,
            allow_offline_mock=True,
        )
    suite_b1 = comp["replicate_level_suite"]["vs_baseline1_zeroshot_across_replicates"]
    # Must count ONLY the 2 offline mock datasets (NOT pooling with LIVE_ONLY_DS!)
    assert suite_b1["n_replicates"] == 2
    assert "wilcoxon_mean_ig" in suite_b1
    assert "wilcoxon_mean_g_node" in suite_b1
    assert "mcnemar_bsf_pooled_nodes" in suite_b1
    assert "mcnemar_ccvr_pooled_nodes" in suite_b1

    # Bug #13: Verify empirical dataset discovery excludes simulation_metadata.json fixtures by default
    raw_emp_root = tmp_path / "01_raw"
    sim_ds = raw_emp_root / "tier1" / "SYNTH_FIXTURE"
    emp_ds = raw_emp_root / "tier1" / "EMP_RANDALL_FP"
    emp_t2 = raw_emp_root / "tier2" / "EMP_NEXTSTRAIN_PANEL"
    sim_ds.mkdir(parents=True)
    emp_ds.mkdir(parents=True)
    emp_t2.mkdir(parents=True)
    (sim_ds / "alignment.fa").write_text(Path(golden_toy_env["fasta_path"]).read_text())
    (sim_ds / "simulation_metadata.json").write_text('{"dataset_id": "SYNTH_FIXTURE"}')
    (emp_ds / "alignment.fa").write_text(Path(golden_toy_env["fasta_path"]).read_text())
    (emp_ds / "T_true.nwk").write_text(Path(golden_toy_env["raw_tree_path"]).read_text())
    (emp_ds / "dataset_metadata.json").write_text(
        json.dumps({"dataset_id": "EMP_RANDALL_FP", "outgroup_taxon": "Taxon_01"})
    )
    (emp_t2 / "alignment.fa").write_text(Path(golden_toy_env["fasta_path"]).read_text())
    (emp_t2 / "T_true.nwk").write_text(Path(golden_toy_env["raw_tree_path"]).read_text())
    (emp_t2 / "dataset_metadata.json").write_text(
        json.dumps({"dataset_id": "EMP_NEXTSTRAIN_PANEL", "outgroup_taxon": "Taxon_01"})
    )

    found = discover_empirical_datasets(raw_emp_root, include_synthetic=False)
    assert [d["dataset_id"] for d in found] == ["EMP_RANDALL_FP", "EMP_NEXTSTRAIN_PANEL"]

    # Execute resumable corpus runner end-to-end (Stage A + Stage B) on isolated tmp directory
    out_data_root = tmp_path / "corpus_out"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_empirical_corpus.py",
            "--raw-root",
            str(raw_emp_root),
            "--out-data-root",
            str(out_data_root),
            "--bootstraps",
            "4",
            "--skip-classical-ml",
            "--use-surrogate",
            "--offline-mock",
        ],
        check=True,
    )
    # Tier 2 must default to 3-engine consensus (tier3_consensus_summary.json + verified_consensus_splits present), while Tier 1 uses direct T_true
    assert not (
        out_data_root / "02_ground_truths" / "tier1" / "EMP_RANDALL_FP" / "tier3_consensus_summary.json"
    ).exists()
    assert (
        out_data_root / "02_ground_truths" / "tier2" / "EMP_NEXTSTRAIN_PANEL" / "tier3_consensus_summary.json"
    ).exists()
    t2_a_true = json.loads(
        (out_data_root / "02_ground_truths" / "tier2" / "EMP_NEXTSTRAIN_PANEL" / "A_true.json").read_text()
    )
    assert t2_a_true["verified_consensus_splits"] is not None
    assert (out_data_root / "04_audit_reports" / "statistical_comparisons" / "stage_a_model_metrics.tsv").exists()
    master_tsv = out_data_root / "04_audit_reports" / "statistical_comparisons" / "master_benchmark_metrics_table.tsv"
    assert master_tsv.exists()

    # Verify checkpoint resume re-syncs master_benchmark_metrics_table.tsv when checkpoints already exist
    master_tsv.unlink()
    subprocess.run(
        [
            sys.executable,
            "scripts/run_empirical_corpus.py",
            "--raw-root",
            str(raw_emp_root),
            "--out-data-root",
            str(out_data_root),
            "--bootstraps",
            "4",
            "--skip-classical-ml",
            "--use-surrogate",
            "--offline-mock",
        ],
        check=True,
    )
    assert master_tsv.exists()


def test_review_12_lingering_edge_cases(golden_toy_env, tmp_path: Path, monkeypatch) -> None:
    """
    Verifies all 7 edge cases identified in second-pass review ('12'):
      1. Tier 2 3-engine consensus forwards cataloged_mutations into M_true.tsv
      2. Sanitizer preserves internal ancestor names on branchless pedigrees (format=8), and
         mismatched ancestor FASTA length raises ValueError
      3. compute_report_edr handles empty long_branch_taxa with expected_stressors=['LBA'] (no FAB:LBA cap)
         and credits SHORT_BRANCH_HOMOPLASY_AND_LBA toward GT:LBA
      4. Prose coordinate regex parses 'Tool 2 Output include 540, 995, 251' despite digit '2'
      5. Malformed LLM JSON on attempt 0 triggers State 4 retry and recovers cleanly without crashing
      6. Sparse alignment bootstrap resampling never emits a 100%-gap taxon row
    """
    from src.preprocessing.empirical_consensus import build_empirical_tier3_ground_truth
    from src.preprocessing.normalizer_wrapper import generate_bootstrap_replicates

    # 1 & 2. Branchless pedigree sanitization preserves internal ancestor names (Anc_A) + length guard
    raw_fa = tmp_path / "t1_raw.fa"
    raw_nwk = tmp_path / "t1_branchless.nwk"
    clean_fa = tmp_path / "t1_clean" / "alignment.fa"
    clean_nwk = tmp_path / "t1_clean" / "T_true.nwk"
    seq_base = "ACGTACGTAC" * 4
    raw_fa.write_text(
        f">Outgroup\n{seq_base}\n>A1\nT{seq_base[1:]}\n>A2\nT{seq_base[1:]}\n>B1\nG{seq_base[1:]}\n"
    )
    raw_nwk.write_text("(Outgroup,((A1,A2)Anc_A,B1)Anc_In);\n")
    sanitize_tier3_empirical_fasta(
        str(raw_fa),
        str(clean_fa),
        outgroup="Outgroup",
        input_tree_path=str(raw_nwk),
        output_tree_path=str(clean_nwk),
    )
    assert "Anc_A" in clean_nwk.read_text()

    bad_anc = tmp_path / "bad_anc.fa"
    bad_anc.write_text(">Anc_A\nACGT\n")  # wrong length 4 vs 40
    with pytest.raises(ValueError, match="expected aligned sequence length"):
        run_ground_truth_pipeline(
            fasta_path=str(clean_fa),
            true_tree_path=str(clean_nwk),
            out_dir=str(tmp_path / "gt_bad_anc"),
            outgroup="Outgroup",
            experimental_ancestors_fasta=str(bad_anc),
        )

    # Tier 2 3-engine consensus + cataloged_mutations forwarding
    gt_t2 = tmp_path / "gt_tier2_consensus"
    build_empirical_tier3_ground_truth(
        fasta_path=str(clean_fa),
        out_dir=str(gt_t2),
        outgroup="Outgroup",
        ufboot_replicates=0,
        use_modelfinder=False,
        published_tree_path=str(clean_nwk),
        cataloged_mutations=[
            {"clade_taxa": ["A1", "A2"], "site": 15, "parent_state": "G", "child_state": "T"}
        ],
    )
    m_t2 = pd.read_csv(gt_t2 / "M_true.tsv", sep="\t")
    assert len(m_t2[(m_t2["site"] == 15) & (m_t2["child_state"] == "T")]) == 1

    # 3. compute_report_edr: no FAB:LBA self-contradiction & SHORT_BRANCH_HOMOPLASY_AND_LBA -> GT:LBA credit
    custom_gt = tmp_path / "custom_A_true.json"
    custom_gt.write_text(
        json.dumps(
            {
                "expected_stressors": ["LBA"],
                "long_branch_taxa": [],
                "recombination_breakpoints": [],
                "gapped_column_pct": 0.0,
                "min_internal_branch_length": 0.02,
                "mean_informative_ci": 0.55,
            }
        )
    )
    audits_compound = [
        {
            "node_id": "N1",
            "predicted_clade_taxa": ["Taxon_08", "Taxon_15"],
            "model_support_pct": 90.0,
            "topological_status": "FALSE_CONFIDENCE_ERROR",
            "diagnosed_failure_mode": {"primary_stressor": "SHORT_BRANCH_HOMOPLASY_AND_LBA"},
        }
    ]
    assert (
        compute_report_edr(
            audits_compound,
            golden_toy_env["fasta_path"],
            golden_toy_env["pred_newick_path"],
            str(custom_gt),
        )
        == 1.0
    )

    # 4. Prose regex handles 'Tool 2 Output include 540, 995, 251' and 'sites such as 152, 399'
    s_tool2, _ = extract_prose_coordinate_claims(
        "Verified site coordinates from Tool 2 Output include 540, 995, 251."
    )
    assert s_tool2 == {540, 995, 251}
    s_such, _ = extract_prose_coordinate_claims(
        "Supported at verified sites such as 152, 399, and 518."
    )
    assert s_such == {152, 399, 518}

    # 5. Malformed LLM JSON on attempt 0 self-corrects via State 4 retry instead of crashing
    monkeypatch.setattr(
        "src.agent.diagnostic_agent.resolve_lm_studio_url",
        lambda url, timeout=3.0: ("http://mock-lmstudio:1234/v1", ["qwen/qwen3-14b"]),
    )
    per_node_calls: dict[str, int] = {}
    m_true_toy = pd.read_csv(golden_toy_env["asr_state_path"], sep="\t")

    class RecoveringFromMalformedJsonChat:
        def __init__(self, **kwargs):
            pass

        def invoke(self, prompt: str):
            import re as _re

            m_nid = _re.search(r"node (N\d+)", prompt)
            nid = m_nid.group(1) if m_nid else "N1"
            cnt = per_node_calls.get(nid, 0)
            per_node_calls[nid] = cnt + 1
            if cnt == 0:
                return SimpleNamespace(content="{truncated malformed json...")
            # On retry 1, return valid verified claim from prompt's Tool 2 Output
            m_t2_json = _re.search(r"Verified Clade Mutations \(Tool 2 Output\): (\[.*?\])\n", prompt)
            syns = json.loads(m_t2_json.group(1)) if m_t2_json else []
            claims = [
                {
                    "site_coordinate": int(s["site"]),
                    "ancestral_state": s["parent_state"],
                    "derived_state": s["child_state"],
                    "posterior_probability": float(s["posterior_prob"]),
                    "consistency_index_ci": float(s["ci_score"]),
                    "is_indel": bool(s["is_indel"]),
                }
                for s in syns[:1]
            ]
            site_note = f" at site {claims[0]['site_coordinate']}" if claims else ""
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "draft_claims": claims,
                        "primary_stressor": "NONE",
                        "explanation": f"Concordant split at {nid} with 100.0% support (definitive high confidence){site_note}.",
                    }
                )
            )

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", RecoveringFromMalformedJsonChat)
    rep_rec = run_diagnostic_audit(
        dataset_id="malformed_json_recovery",
        target_model="Fusang",
        fasta_path=golden_toy_env["fasta_path"],
        pred_newick_path=golden_toy_env["pred_newick_path"],
        pred_meta_path=golden_toy_env["pred_meta_path"],
        true_newick_path=golden_toy_env["true_newick_path"],
        asr_state_path=golden_toy_env["asr_state_path"],
        stressor_gt_path=golden_toy_env["stressor_gt_path"],
        lm_studio_base_url="http://mock-lmstudio:1234/v1",
        require_live_llm=True,
        max_retries=3,
    )
    assert rep_rec["global_metrics"]["claim_to_coordinate_verification_rate"] == 1.0
    assert rep_rec["global_metrics"]["total_hallucination_retries_triggered"] == 13

    # 6. Sparse taxon bootstrap safeguard: 1 non-gap base in 60 columns never draws a 100%-gap replicate
    sparse_fa = tmp_path / "sparse.fa"
    sparse_row = ("-" * 59) + "A"
    full_row = "ACGT" * 15
    sparse_fa.write_text(
        f">T1\n{full_row}\n>T2\n{full_row}\n>T3\n{full_row}\n>SparseTaxon\n{sparse_row}\n"
    )
    rep_files = generate_bootstrap_replicates(str(sparse_fa), str(tmp_path / "reps"), b_replicates=25, seed=7)
    for rf in rep_files:
        txt = Path(rf).read_text()
        sparse_seq = txt.split(">SparseTaxon\n")[1].splitlines()[0]
        assert "A" in sparse_seq

