"""
Unit tests for Step C:
  - AI Baseline 1 (Zero-Shot LLM)
  - AI Baseline 2 (Generic Unconstrained Agent)
  - Separate directory/JSON report serialization and paired statistical comparison
"""
from pathlib import Path

from scripts.run_ai_baselines import run_three_way_comparison
from tests.test_agent import _ensure_toy_ground_truth_and_pred


def test_three_way_comparison_and_separate_reports(tmp_path: Path) -> None:
    _ensure_toy_ground_truth_and_pred()
    out_reports = tmp_path / "04_audit_reports"

    comp = run_three_way_comparison(
        dataset_id="TOY_DEV_15TAXA",
        target_model="Fusang",
        fasta_path="data/00_golden_toy/toy_alignment.fa",
        pred_newick_path="data/03_normalized_preds/toy_fusang_normalized.nwk",
        pred_meta_path="data/03_normalized_preds/toy_fusang_metadata.json",
        true_newick_path="data/02_ground_truths/toy/T_true_rooted.nwk",
        asr_state_path="data/02_ground_truths/toy/M_true.tsv",
        stressor_gt_path="data/02_ground_truths/toy/A_true.json",
        out_reports_root=str(out_reports),
        agent_model="qwen/qwen3-14b",
        zeroshot_model="custom/llama3-8b",
        generic_agent_model="qwen/qwen3-14b",
        lm_studio_base_url=None,  # offline deterministic test mode
        require_live_llm=False,
    )

    # Verify separate folders and JSON reports were created
    prop_files = list((out_reports / "proposed_agent").glob("*.json"))
    b1_files = list((out_reports / "baseline1_zeroshot_llm").glob("*.json"))
    b2_files = list((out_reports / "baseline2_generic_agent").glob("*.json"))
    stat_files = list((out_reports / "statistical_comparisons").glob("*.json"))
    tsv_file = out_reports / "statistical_comparisons" / "master_benchmark_metrics_table.tsv"

    assert len(prop_files) == 1
    assert len(b1_files) == 1
    assert len(b2_files) == 1
    assert len(stat_files) == 1
    assert tsv_file.exists()

    # Verify custom model IDs were recorded separately
    rows = {r["auditor_method"]: r for r in comp["summary_table"]}
    assert rows["Proposed_4State_Agent"]["llm_model_id"] == "qwen/qwen3-14b"
    assert rows["Baseline1_ZeroShot_LLM"]["llm_model_id"] == "custom/llama3-8b"
    assert rows["Baseline2_Generic_Agent"]["llm_model_id"] == "qwen/qwen3-14b"

    # Proposed 4-State Agent must enforce CCVR = 1.0 and outperform unverified baselines on G_tree
    assert rows["Proposed_4State_Agent"]["ccvr"] == 1.0
    assert rows["Proposed_4State_Agent"]["g_tree"] > rows["Baseline1_ZeroShot_LLM"]["g_tree"]
    assert rows["Proposed_4State_Agent"]["g_tree"] > rows["Baseline2_Generic_Agent"]["g_tree"]
