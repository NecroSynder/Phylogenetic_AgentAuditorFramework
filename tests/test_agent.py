"""
End-to-End & Anti-Loophole Tests for Pipeline Step 3 (LangGraph Diagnostic Agent)
and Pipeline Step 4 (Evaluation Metrics, FGS, BSF, LCR, EDR, and Statistical Tests).
"""
from pathlib import Path
import pytest

from src.agent.diagnostic_agent import run_diagnostic_audit
from src.evaluation.metrics import (
    compute_bsf_local,
    compute_edr,
    compute_feature_precision_recall_f1,
    compute_fgs_local,
    compute_lcr,
    compute_node_goodness,
    run_statistical_validation,
)
from src.preprocessing.normalizer_wrapper import standardize_and_root_tree
from src.schemas.audit_schema import AuditReport


def test_diagnostic_agent_blocks_silent_offline_mock_without_flag(golden_toy_env):
    """Ensure offline template fallback is blocked unless allow_offline_mock=True is explicit."""
    with pytest.raises(RuntimeError, match="allow_offline_mock"):
        run_diagnostic_audit(
            dataset_id="toy_guard_check",
            target_model="Fusang",
            fasta_path=golden_toy_env["fasta_path"],
            pred_newick_path=golden_toy_env["pred_newick_path"],
            pred_meta_path=golden_toy_env["pred_meta_path"],
            true_newick_path=golden_toy_env["true_newick_path"],
            asr_state_path=golden_toy_env["asr_state_path"],
            stressor_gt_path=golden_toy_env["stressor_gt_path"],
            lm_studio_base_url=None,
            allow_offline_mock=False,
        )


def test_diagnostic_agent_concordant_tree(golden_toy_env, tmp_path):
    """Verify full 4-state LangGraph loop on isolated Golden Toy fixture."""
    out_json = tmp_path / "toy_fusang_audit_report.json"

    report = run_diagnostic_audit(
        dataset_id="toy_00_fusang",
        target_model="Fusang",
        fasta_path=golden_toy_env["fasta_path"],
        pred_newick_path=golden_toy_env["pred_newick_path"],
        pred_meta_path=golden_toy_env["pred_meta_path"],
        true_newick_path=golden_toy_env["true_newick_path"],
        asr_state_path=golden_toy_env["asr_state_path"],
        stressor_gt_path=golden_toy_env["stressor_gt_path"],
        output_json_path=str(out_json),
        allow_offline_mock=True,
    )

    validated = AuditReport(**report)
    assert validated.audit_metadata.orchestrator_llm == "OFFLINE_DETERMINISTIC_MOCK"
    assert validated.global_metrics.normalized_rf_distance == 0.0
    assert validated.global_metrics.false_confidence_index_fci == 0.0
    assert validated.global_metrics.total_internal_nodes_audited == 13
    assert validated.global_metrics.claim_to_coordinate_verification_rate == 1.0
    assert validated.global_metrics.logical_consistency_rate_lcr == 1.0
    assert validated.global_metrics.overall_tree_goodness_score >= 0.90
    assert out_json.exists()


def test_diagnostic_agent_discordant_lba_detection_and_edr(golden_toy_env, tmp_path):
    """Verify discordant LBA swap triggers State 3 LBA diagnosis and achieves EDR = 1.0."""
    nwk_str = Path(golden_toy_env["true_newick_path"]).read_text().strip()
    discordant_nwk = (
        nwk_str.replace("Taxon_14", "__TMP__")
        .replace("Taxon_08", "Taxon_14")
        .replace("__TMP__", "Taxon_08")
    )
    disc_prefix = str(tmp_path / "disc_fusang")
    standardize_and_root_tree(
        discordant_nwk, [discordant_nwk] * 10, "Taxon_01", "Fusang", disc_prefix
    )

    report = run_diagnostic_audit(
        dataset_id="toy_discordant_lba",
        target_model="Fusang",
        fasta_path=golden_toy_env["fasta_path"],
        pred_newick_path=f"{disc_prefix}_normalized.nwk",
        pred_meta_path=f"{disc_prefix}_metadata.json",
        true_newick_path=golden_toy_env["true_newick_path"],
        asr_state_path=golden_toy_env["asr_state_path"],
        stressor_gt_path=golden_toy_env["stressor_gt_path"],
        allow_offline_mock=True,
    )

    validated = AuditReport(**report)
    assert validated.global_metrics.normalized_rf_distance > 0.0
    assert validated.global_metrics.false_confidence_index_fci > 0.0
    assert validated.global_metrics.error_detection_rate_edr == 1.0
    assert validated.global_metrics.claim_to_coordinate_verification_rate == 1.0
    assert any(
        n.diagnosed_failure_mode.primary_stressor == "LBA" for n in validated.node_audits
    )


def test_state_4_gate_self_corrects_and_persistent_failure_never_infinite_loops(golden_toy_env):
    """
    1. Attempt-0 hallucination is blocked and self-corrected to CCVR = 1.00 on retry 1.
    2. Persistent hallucination across all max_retries terminates cleanly without infinite recursion.
    """
    attempt_counter = {"calls": 0}

    def single_hallucinating_synthesizer(state):
        attempt_counter["calls"] += 1
        return (
            [{"site_coordinate": 9999, "ancestral_state": "A", "derived_state": "G"}],
            "Hallucinated draft claiming A9999G with 100.0% support (definitive high confidence).",
        )

    rep_recovered = run_diagnostic_audit(
        dataset_id="toy_hallucination_recovery",
        target_model="Fusang",
        fasta_path=golden_toy_env["fasta_path"],
        pred_newick_path=golden_toy_env["pred_newick_path"],
        pred_meta_path=golden_toy_env["pred_meta_path"],
        true_newick_path=golden_toy_env["true_newick_path"],
        asr_state_path=golden_toy_env["asr_state_path"],
        stressor_gt_path=golden_toy_env["stressor_gt_path"],
        custom_synthesizer=single_hallucinating_synthesizer,
        max_retries=3,
    )
    assert attempt_counter["calls"] == 13
    assert rep_recovered["global_metrics"]["total_hallucination_retries_triggered"] == 13
    assert rep_recovered["global_metrics"]["claim_to_coordinate_verification_rate"] == 1.0

    # Persistent failure test (Regression test for Bug 1.4 off-by-one infinite loop)
    persist_counter = {"calls": 0}

    def persistent_bad_synthesizer(state):
        persist_counter["calls"] += 1
        return (
            [{"site_coordinate": 8888, "ancestral_state": "A", "derived_state": "G"}],
            "Persistent unverified site 8888 with 100.0% support (definitive high confidence).",
        )

    persistent_bad_synthesizer.persist_across_retries = True
    rep_persist = run_diagnostic_audit(
        dataset_id="toy_persistent_termination",
        target_model="Fusang",
        fasta_path=golden_toy_env["fasta_path"],
        pred_newick_path=golden_toy_env["pred_newick_path"],
        pred_meta_path=golden_toy_env["pred_meta_path"],
        true_newick_path=golden_toy_env["true_newick_path"],
        asr_state_path=golden_toy_env["asr_state_path"],
        stressor_gt_path=golden_toy_env["stressor_gt_path"],
        custom_synthesizer=persistent_bad_synthesizer,
        max_retries=2,
    )
    # Exactly (1 initial + 2 retries) * 13 nodes = 39 calls, and exactly 13 node records (no duplicates!)
    assert persist_counter["calls"] == 39
    assert len(rep_persist["node_audits"]) == 13
    assert rep_persist["global_metrics"]["claim_to_coordinate_verification_rate"] == 0.0


def test_metrics_anti_loophole_bsf_fgs_lcr_and_stats(golden_toy_env):
    """Verify negation-aware BSF, non-vacuous FGS, plural site parsing, and LCR prose checking."""
    prec, rec, f1 = compute_feature_precision_recall_f1({4, 128, 464}, {4, 128, 464, 505})
    assert prec == 1.0 and rec == 0.75 and abs(f1 - 0.8571) < 1e-3

    # Negation-aware BSF
    assert compute_bsf_local(65.0, "Branch does not have high support; shows moderate confidence.") == 1
    assert compute_bsf_local(95.0, "Split has 95.0% support (definitive high confidence).") == 1
    assert compute_bsf_local(95.0, "Split has low confidence.") == 0

    # FGS must reject vacuous ungrounded prose AND catch plural 'sites 505, 9999' hallucinations
    syns = [{"site": 505, "parent_state": "A", "child_state": "T", "is_indel": False}]
    stressor_none = {"primary_artifact": "NONE"}
    assert compute_fgs_local("Cosmic rays caused rapid evolution on Mars.", "N1", 100.0, syns, stressor_none) == 0.0
    assert compute_fgs_local("Node N1 is supported by sites 505 and 9999.", "N1", 100.0, syns, stressor_none) == 0.0
    assert compute_fgs_local("Node N1 has 100.0% support and verified mutation at site 505.", "N1", 100.0, syns, stressor_none) == 1.0

    # LCR must penalize prose calling a discordant node 'concordant'
    bad_audits = [
        {
            "node_id": "N1",
            "predicted_clade_taxa": ["Taxon_14", "Taxon_15"],
            "model_support_pct": 92.0,
            "topological_status": "FALSE_CONFIDENCE_ERROR",
            "diagnosed_failure_mode": {
                "agent_causal_explanation": "Clade N1 is concordant with the reference tree with low confidence."
            },
        }
    ]
    assert compute_lcr(bad_audits, golden_toy_env["pred_newick_path"]) < 1.0

    assert compute_node_goodness(0.0, 1, 1.0, 1.0, 1.0) == 0.0
    assert compute_edr({"N4:LBA"}, {"N4:LBA"}) == 1.0

    stats_res = run_statistical_validation(
        agent_continuous=[0.95, 0.92, 0.96, 0.94, 0.91, 0.93, 0.97, 0.95],
        baseline_continuous=[0.22, 0.18, 0.25, 0.20, 0.15, 0.21, 0.24, 0.19],
        agent_binary=[1, 1, 1, 1, 1, 1, 1, 1],
        baseline_binary=[0, 0, 1, 0, 0, 0, 0, 0],
    )
    assert stats_res["wilcoxon_significant"] is True
