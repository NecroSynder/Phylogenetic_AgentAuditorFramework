"""
Unit tests for Step C:
  - Unbiased 3-way comparison (Proposed 4-State Agent vs. Zero-Shot Baseline 1 vs. Generic Agent Baseline 2)
  - Verifies non-self-referential EDR calculation against ground-truth A_true.json on discordant LBA splits
  - Verifies TSV upsert deduplication and replicate-level statistical suite across >= 2 datasets
"""
from pathlib import Path
import pandas as pd

from scripts.run_ai_baselines import run_three_way_comparison
from src.preprocessing.normalizer_wrapper import standardize_and_root_tree


def test_three_way_comparison_unbiased_edr_and_replicate_stats(
    golden_toy_env, tmp_path: Path
) -> None:
    out_reports = tmp_path / "04_audit_reports"

    # Build a discordant LBA prediction (swapping Taxon_14 with long-branch Taxon_08)
    nwk_str = Path(golden_toy_env["true_newick_path"]).read_text().strip()
    disc_nwk = (
        nwk_str.replace("Taxon_14", "__TMP__")
        .replace("Taxon_08", "Taxon_14")
        .replace("__TMP__", "Taxon_08")
    )
    disc_prefix = str(tmp_path / "disc_fusang")
    standardize_and_root_tree(
        disc_nwk, [disc_nwk] * 10, "Taxon_01", "Fusang", disc_prefix
    )

    comp1 = run_three_way_comparison(
        dataset_id="DS_REP_01",
        target_model="Fusang",
        fasta_path=golden_toy_env["fasta_path"],
        pred_newick_path=f"{disc_prefix}_normalized.nwk",
        pred_meta_path=f"{disc_prefix}_metadata.json",
        true_newick_path=golden_toy_env["true_newick_path"],
        asr_state_path=golden_toy_env["asr_state_path"],
        stressor_gt_path=golden_toy_env["stressor_gt_path"],
        out_reports_root=str(out_reports),
        lm_studio_base_url=None,
        allow_offline_mock=True,
    )

    rows = {r["auditor_method"]: r for r in comp1["summary_table"]}
    # Proposed Agent must achieve CCVR = 1.0, LCR = 1.0, EDR = 1.0, and outperform baselines without bias
    assert rows["Proposed_4State_Agent"]["ccvr"] == 1.0
    assert rows["Proposed_4State_Agent"]["lcr"] == 1.0
    assert rows["Proposed_4State_Agent"]["edr"] == 1.0
    assert rows["Proposed_4State_Agent"]["edr"] > rows["Baseline1_ZeroShot_LLM"]["edr"]
    assert rows["Proposed_4State_Agent"]["g_tree"] > rows["Baseline1_ZeroShot_LLM"]["g_tree"]
    assert rows["Proposed_4State_Agent"]["g_tree"] > rows["Baseline2_Generic_Agent"]["g_tree"]

    # Re-running the exact same dataset_id must deduplicate (upsert) rather than duplicate TSV rows
    run_three_way_comparison(
        dataset_id="DS_REP_01",
        target_model="Fusang",
        fasta_path=golden_toy_env["fasta_path"],
        pred_newick_path=f"{disc_prefix}_normalized.nwk",
        pred_meta_path=f"{disc_prefix}_metadata.json",
        true_newick_path=golden_toy_env["true_newick_path"],
        asr_state_path=golden_toy_env["asr_state_path"],
        stressor_gt_path=golden_toy_env["stressor_gt_path"],
        out_reports_root=str(out_reports),
        lm_studio_base_url=None,
        allow_offline_mock=True,
    )
    tsv_path = out_reports / "statistical_comparisons" / "master_benchmark_metrics_table.tsv"
    df_after_dup = pd.read_csv(tsv_path, sep="\t")
    assert len(df_after_dup) == 3

    # Running a second independent dataset triggers replicate-level Wilcoxon/McNemar + BH-FDR suite
    comp2 = run_three_way_comparison(
        dataset_id="DS_REP_02",
        target_model="Fusang",
        fasta_path=golden_toy_env["fasta_path"],
        pred_newick_path=golden_toy_env["pred_newick_path"],
        pred_meta_path=golden_toy_env["pred_meta_path"],
        true_newick_path=golden_toy_env["true_newick_path"],
        asr_state_path=golden_toy_env["asr_state_path"],
        stressor_gt_path=golden_toy_env["stressor_gt_path"],
        out_reports_root=str(out_reports),
        lm_studio_base_url=None,
        allow_offline_mock=True,
    )
    assert "vs_baseline1_zeroshot_across_replicates" in comp2["replicate_level_suite"]
    assert (out_reports / "statistical_comparisons" / "replicate_level_statistical_suite.json").exists()
