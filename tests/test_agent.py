"""
End-to-End Tests for Pipeline Step 3 (LangGraph Diagnostic Agent Control Loop)
and Pipeline Step 4 (Multi-Dimensional Evaluation Metrics & Validated JSON Audit Report).
"""
import json
from pathlib import Path
from src.preprocessing.ground_truth_gen import run_ground_truth_pipeline
from src.preprocessing.normalizer_wrapper import standardize_and_root_tree
from src.agent.diagnostic_agent import run_diagnostic_audit
from src.schemas.audit_schema import AuditReport
from src.evaluation.metrics import (
    compute_feature_precision_recall_f1,
    compute_bsf_local,
    compute_node_goodness,
    compute_edr,
    run_statistical_validation,
)


def _ensure_toy_ground_truth_and_pred():
    if not Path("data/02_ground_truths/toy/M_true.tsv").exists():
        run_ground_truth_pipeline(
            "data/00_golden_toy/toy_alignment.fa",
            "data/00_golden_toy/T_true.nwk",
            "data/02_ground_truths/toy",
        )
    if not Path("data/03_normalized_preds/toy_fusang_normalized.nwk").exists():
        with open("data/02_ground_truths/toy/T_true_rooted.nwk") as f:
            nwk_str = f.read().strip()
        standardize_and_root_tree(
            nwk_str,
            [nwk_str] * 10,
            "Taxon_01",
            "Fusang",
            "data/03_normalized_preds/toy_fusang",
        )


def test_diagnostic_agent_concordant_tree():
    """Verify full 4-state LangGraph loop on Golden Toy dataset and output audit_report.json."""
    _ensure_toy_ground_truth_and_pred()
    out_json = "data/04_audit_reports/toy_fusang_audit_report.json"

    report = run_diagnostic_audit(
        dataset_id="toy_00_fusang",
        target_model="Fusang",
        fasta_path="data/00_golden_toy/toy_alignment.fa",
        pred_newick_path="data/03_normalized_preds/toy_fusang_normalized.nwk",
        pred_meta_path="data/03_normalized_preds/toy_fusang_metadata.json",
        true_newick_path="data/02_ground_truths/toy/T_true_rooted.nwk",
        asr_state_path="data/02_ground_truths/toy/M_true.tsv",
        stressor_gt_path="data/02_ground_truths/toy/A_true.json",
        output_json_path=out_json,
    )

    # Validate returned dict against Pydantic AuditReport schema
    validated = AuditReport(**report)
    assert validated.audit_metadata.dataset_id == "toy_00_fusang"
    assert validated.audit_metadata.taxon_count == 15
    assert validated.audit_metadata.sequence_length == 1000
    assert validated.global_metrics.normalized_rf_distance == 0.0
    assert validated.global_metrics.false_confidence_index_fci == 0.0
    assert validated.global_metrics.total_internal_nodes_audited == 13
    assert validated.global_metrics.claim_to_coordinate_verification_rate == 1.0
    assert validated.global_metrics.logical_consistency_rate_lcr == 1.0
    assert validated.global_metrics.overall_tree_goodness_score >= 0.85
    assert Path(out_json).exists()


def test_diagnostic_agent_discordant_stressor_routing(tmp_path):
    """Verify State 2 -> State 3 stressor interrogation when discordant high-support splits occur."""
    _ensure_toy_ground_truth_and_pred()
    with open("data/02_ground_truths/toy/T_true_rooted.nwk") as f:
        nwk_str = f.read().strip()

    # Swap Taxon_14 and Taxon_08 (long branch) to induce a false-confidence topological error
    discordant_nwk = (
        nwk_str.replace("Taxon_14", "__TMP__")
        .replace("Taxon_08", "Taxon_14")
        .replace("__TMP__", "Taxon_08")
    )
    disc_prefix = str(tmp_path / "disc_phyloformer")
    standardize_and_root_tree(
        discordant_nwk,
        [discordant_nwk] * 10,
        "Taxon_01",
        "Phyloformer",
        disc_prefix,
    )

    report = run_diagnostic_audit(
        dataset_id="toy_discordant_lba",
        target_model="Phyloformer",
        fasta_path="data/00_golden_toy/toy_alignment.fa",
        pred_newick_path=f"{disc_prefix}_normalized.nwk",
        pred_meta_path=f"{disc_prefix}_metadata.json",
        true_newick_path="data/02_ground_truths/toy/T_true_rooted.nwk",
        asr_state_path="data/02_ground_truths/toy/M_true.tsv",
        stressor_gt_path="data/02_ground_truths/toy/A_true.json",
    )

    validated = AuditReport(**report)
    assert validated.global_metrics.normalized_rf_distance > 0.0
    assert validated.global_metrics.false_confidence_index_fci > 0.0
    assert validated.global_metrics.false_confidence_nodes_count > 0
    assert validated.global_metrics.claim_to_coordinate_verification_rate == 1.0
    assert validated.global_metrics.logical_consistency_rate_lcr == 1.0


def test_state_4_verification_gate_blocks_hallucination_and_self_corrects():
    """
    Verify that if the LLM drafts a hallucinated coordinate on attempt 0,
    State 4 (verify_claim_coordinates) rejects it, loops back, and self-corrects to CCVR = 1.00.
    """
    _ensure_toy_ground_truth_and_pred()
    attempt_counter = {"calls": 0}

    def hallucinating_synthesizer(state):
        attempt_counter["calls"] += 1
        # Inject an out-of-bounds hallucinated coordinate (site 9999) on attempt 0
        return (
            [
                {
                    "site_coordinate": 9999,
                    "ancestral_state": "A",
                    "derived_state": "G",
                    "posterior_probability": 0.99,
                    "consistency_index_ci": 1.0,
                    "mutation_class": "SYNAPOMORPHY",
                }
            ],
            "Hallucinated draft claiming A9999G with 100.0% support (definitive high confidence).",
        )

    report = run_diagnostic_audit(
        dataset_id="toy_hallucination_recovery",
        target_model="Fusang",
        fasta_path="data/00_golden_toy/toy_alignment.fa",
        pred_newick_path="data/03_normalized_preds/toy_fusang_normalized.nwk",
        pred_meta_path="data/03_normalized_preds/toy_fusang_metadata.json",
        true_newick_path="data/02_ground_truths/toy/T_true_rooted.nwk",
        asr_state_path="data/02_ground_truths/toy/M_true.tsv",
        stressor_gt_path="data/02_ground_truths/toy/A_true.json",
        custom_synthesizer=hallucinating_synthesizer,
        max_retries=3,
    )

    assert attempt_counter["calls"] == 13  # Called once per node on attempt 0, then rejected & corrected on retry 1
    assert report["global_metrics"]["claim_to_coordinate_verification_rate"] == 1.0
    for node_rec in report["node_audits"]:
        assert node_rec["explanation_quality_vector"]["ccvr_local"] == 1.0
        assert all(ev["site_coordinate"] != 9999 for ev in node_rec["causal_sequence_evidence"])


def test_evaluation_metrics_and_statistical_validation():
    """Verify mathematical properties of F1, BSF, Multiplicative G_node Gate (Eq. 4.14), EDR, and Statistical Tests."""
    prec, rec, f1 = compute_feature_precision_recall_f1({4, 128, 464}, {4, 128, 464, 505})
    assert prec == 1.0
    assert rec == 0.75
    assert abs(f1 - 0.8571) < 1e-3

    # BSF confidence tier matching (Eq. 4.7)
    assert compute_bsf_local(95.0, "Split has 95.0% support (definitive high confidence).") == 1
    assert compute_bsf_local(95.0, "Split has low confidence.") == 0
    assert compute_bsf_local(65.0, "Branch exhibits moderate confidence.") == 1
    assert compute_bsf_local(30.0, "Weak split with low confidence.") == 1

    # Multiplicative G_node gate collapses to 0.0 when CCVR=0 or BSF=0 (Eq. 4.14)
    assert compute_node_goodness(0.0, 1, 1.0, 1.0, 1.0) == 0.0
    assert compute_node_goodness(1.0, 0, 1.0, 1.0, 1.0) == 0.0
    assert compute_node_goodness(1.0, 1, 0.9, 0.9, 0.9) == 0.9

    # Error Detection Rate (Eq. 4.4)
    assert compute_edr({"N4:LBA", "N7:INDEL_GAP"}, {"N4:LBA", "N7:INDEL_GAP"}) == 1.0

    # Wilcoxon signed-rank + McNemar + Benjamini-Hochberg FDR (Section 4.3.1)
    stats_res = run_statistical_validation(
        agent_continuous=[0.95, 0.92, 0.96, 0.94, 0.91, 0.93, 0.97, 0.95, 0.94, 0.96],
        baseline_continuous=[0.42, 0.38, 0.45, 0.40, 0.35, 0.41, 0.44, 0.39, 0.40, 0.43],
        agent_binary=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        baseline_binary=[0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    )
    assert "wilcoxon_p_fdr_bh" in stats_res
    assert "mcnemar_p_fdr_bh" in stats_res
    assert stats_res["wilcoxon_significant"] is True
