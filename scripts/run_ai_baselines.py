#!/usr/bin/env python3
"""
Master Benchmark & Statistical Comparison Runner for Step C (Manuscript Section 4.2.3 & Section 4.3.3).

Executes and compares:
  1. Proposed 4-State Neuro-Symbolic Diagnostic Agent -> data/04_audit_reports/proposed_agent/
  2. AI Baseline 1 (Zero-Shot LLM)                    -> data/04_audit_reports/baseline1_zeroshot_llm/
  3. AI Baseline 2 (Generic Unconstrained Agent)      -> data/04_audit_reports/baseline2_generic_agent/
  4. Paired Statistical Validation (Wilcoxon, McNemar, Benjamini-Hochberg FDR)
     -> data/04_audit_reports/statistical_comparisons/

Allows independent LLM model selection for each method (--agent-model, --zeroshot-model, --generic-agent-model).
"""
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.ai_baselines import (
    run_generic_coding_agent_baseline,
    run_zero_shot_llm_baseline,
)
from src.agent.diagnostic_agent import run_diagnostic_audit
from src.evaluation.metrics import compute_edr, run_statistical_validation


def _sanitize_model_tag(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")


def _extract_summary_row(
    method_label: str,
    llm_model_id: str,
    report_dict: dict,
    json_path: str,
) -> dict:
    gm = report_dict["global_metrics"]
    nodes = report_dict["node_audits"]
    mean_f1 = round(
        float(np.mean([n["explanation_quality_vector"]["feature_f1"] for n in nodes])),
        4,
    ) if nodes else 0.0
    mean_fgs = round(
        float(np.mean([n["explanation_quality_vector"]["faithfulness_fgs"] for n in nodes])),
        4,
    ) if nodes else 0.0

    # Compute Error Detection Recall (EDR) over discordant nodes (or 1.0 if tree is 100% concordant)
    detected_errors = {
        f"{n['node_id']}:{n['diagnosed_failure_mode']['primary_stressor']}"
        for n in nodes
        if n["topological_status"] != "CONCORDANT" and n["diagnosed_failure_mode"]["primary_stressor"] != "NONE"
    }
    true_errors = {
        f"{n['node_id']}:{n['diagnosed_failure_mode']['primary_stressor']}"
        for n in nodes
        if n["topological_status"] != "CONCORDANT"
    }
    edr_val = compute_edr(detected_errors, true_errors)

    return {
        "dataset_id": report_dict["audit_metadata"]["dataset_id"],
        "target_model": report_dict["audit_metadata"]["target_model"],
        "auditor_method": method_label,
        "llm_model_id": llm_model_id,
        "ccvr": float(gm["claim_to_coordinate_verification_rate"]),
        "lcr": float(gm["logical_consistency_rate_lcr"]),
        "g_tree": float(gm["overall_tree_goodness_score"]),
        "mean_feature_f1": mean_f1,
        "mean_fgs": mean_fgs,
        "edr": edr_val,
        "report_json_path": str(json_path),
    }


def run_three_way_comparison(
    dataset_id: str,
    target_model: str,
    fasta_path: str,
    pred_newick_path: str,
    pred_meta_path: str,
    true_newick_path: str,
    asr_state_path: str,
    stressor_gt_path: str,
    out_reports_root: str = "data/04_audit_reports",
    agent_model: str = "qwen/qwen3-14b",
    zeroshot_model: str = "qwen/qwen3-14b",
    generic_agent_model: str = "qwen/qwen3-14b",
    lm_studio_base_url: str | None = None,
    require_live_llm: bool = False,
) -> dict:
    """
    Runs Proposed 4-State Agent, AI Baseline 1 (Zero-Shot LLM), and AI Baseline 2 (Generic Agent)
    into separate directories under out_reports_root and computes paired statistical tests.
    """
    root = Path(out_reports_root)
    prop_dir = root / "proposed_agent"
    b1_dir = root / "baseline1_zeroshot_llm"
    b2_dir = root / "baseline2_generic_agent"
    stat_dir = root / "statistical_comparisons"

    for d in (prop_dir, b1_dir, b2_dir, stat_dir):
        d.mkdir(parents=True, exist_ok=True)

    agent_tag = _sanitize_model_tag(agent_model)
    zs_tag = _sanitize_model_tag(zeroshot_model)
    ga_tag = _sanitize_model_tag(generic_agent_model)

    prop_json = prop_dir / f"{dataset_id}_{target_model}_proposed_{agent_tag}_report.json"
    b1_json = b1_dir / f"{dataset_id}_{target_model}_zeroshot_{zs_tag}_report.json"
    b2_json = b2_dir / f"{dataset_id}_{target_model}_generic_agent_{ga_tag}_report.json"

    prop_rep = run_diagnostic_audit(
        dataset_id=dataset_id,
        target_model=target_model,
        fasta_path=fasta_path,
        pred_newick_path=pred_newick_path,
        pred_meta_path=pred_meta_path,
        true_newick_path=true_newick_path,
        asr_state_path=asr_state_path,
        stressor_gt_path=stressor_gt_path,
        output_json_path=str(prop_json),
        orchestrator_llm=agent_model,
        lm_studio_base_url=lm_studio_base_url,
        require_live_llm=require_live_llm,
    )

    b1_rep = run_zero_shot_llm_baseline(
        dataset_id=dataset_id,
        target_model=target_model,
        fasta_path=fasta_path,
        pred_newick_path=pred_newick_path,
        pred_meta_path=pred_meta_path,
        true_newick_path=true_newick_path,
        asr_state_path=asr_state_path,
        stressor_gt_path=stressor_gt_path,
        output_json_path=str(b1_json),
        zeroshot_llm=zeroshot_model,
        lm_studio_base_url=lm_studio_base_url,
        require_live_llm=require_live_llm,
    )

    b2_rep = run_generic_coding_agent_baseline(
        dataset_id=dataset_id,
        target_model=target_model,
        fasta_path=fasta_path,
        pred_newick_path=pred_newick_path,
        pred_meta_path=pred_meta_path,
        true_newick_path=true_newick_path,
        asr_state_path=asr_state_path,
        stressor_gt_path=stressor_gt_path,
        output_json_path=str(b2_json),
        generic_agent_llm=generic_agent_model,
        lm_studio_base_url=lm_studio_base_url,
        require_live_llm=require_live_llm,
    )

    # Paired node-level statistical validation (Proposed Agent vs Zero-Shot & Generic Agent)
    prop_g_nodes = [float(n["node_goodness_score"]) for n in prop_rep["node_audits"]]
    b1_g_nodes = [float(n["node_goodness_score"]) for n in b1_rep["node_audits"]]
    b2_g_nodes = [float(n["node_goodness_score"]) for n in b2_rep["node_audits"]]

    prop_verified_flags = [
        1 if n["explanation_quality_vector"]["ccvr_local"] >= 0.999 else 0
        for n in prop_rep["node_audits"]
    ]
    b1_verified_flags = [
        1 if n["explanation_quality_vector"]["ccvr_local"] >= 0.999 else 0
        for n in b1_rep["node_audits"]
    ]
    b2_verified_flags = [
        1 if n["explanation_quality_vector"]["ccvr_local"] >= 0.999 else 0
        for n in b2_rep["node_audits"]
    ]

    stats_vs_b1 = run_statistical_validation(
        agent_continuous=prop_g_nodes,
        baseline_continuous=b1_g_nodes,
        agent_binary=prop_verified_flags,
        baseline_binary=b1_verified_flags,
    )
    stats_vs_b2 = run_statistical_validation(
        agent_continuous=prop_g_nodes,
        baseline_continuous=b2_g_nodes,
        agent_binary=prop_verified_flags,
        baseline_binary=b2_verified_flags,
    )

    rows = [
        _extract_summary_row("Proposed_4State_Agent", agent_model, prop_rep, str(prop_json)),
        _extract_summary_row("Baseline1_ZeroShot_LLM", zeroshot_model, b1_rep, str(b1_json)),
        _extract_summary_row("Baseline2_Generic_Agent", generic_agent_model, b2_rep, str(b2_json)),
    ]

    comparison_summary = {
        "dataset_id": dataset_id,
        "target_model": target_model,
        "summary_table": rows,
        "statistical_tests_vs_baseline1_zeroshot": stats_vs_b1,
        "statistical_tests_vs_baseline2_generic_agent": stats_vs_b2,
    }
    comp_json_path = stat_dir / f"{dataset_id}_{target_model}_comparison_summary.json"
    with open(comp_json_path, "w") as f:
        json.dump(comparison_summary, f, indent=2)

    tsv_path = stat_dir / "master_benchmark_metrics_table.tsv"
    write_header = not tsv_path.exists()
    with open(tsv_path, "a") as f:
        if write_header:
            f.write(
                "dataset_id\ttarget_model\tauditor_method\tllm_model_id\t"
                "CCVR\tLCR\tG_tree\tMean_F1_site\tMean_FGS\tEDR\treport_json_path\n"
            )
        for r in rows:
            f.write(
                f"{r['dataset_id']}\t{r['target_model']}\t{r['auditor_method']}\t{r['llm_model_id']}\t"
                f"{r['ccvr']:.4f}\t{r['lcr']:.4f}\t{r['g_tree']:.4f}\t"
                f"{r['mean_feature_f1']:.4f}\t{r['mean_fgs']:.4f}\t{r['edr']:.4f}\t{r['report_json_path']}\n"
            )

    return comparison_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Proposed 4-State Agent vs. AI Baseline 1 (Zero-Shot LLM) & AI Baseline 2 (Generic Agent)."
    )
    parser.add_argument("--dataset-id", default="TOY_DEV_15TAXA", help="Dataset identifier.")
    parser.add_argument(
        "--target-model",
        default="Fusang",
        choices=["Fusang", "Phyloformer"],
        help="Target deep learning phylogenetic model under audit.",
    )
    parser.add_argument("--fasta", default="data/00_golden_toy/toy_alignment.fa")
    parser.add_argument("--pred-newick", default="data/03_normalized_preds/toy_fusang_normalized.nwk")
    parser.add_argument("--pred-meta", default="data/03_normalized_preds/toy_fusang_metadata.json")
    parser.add_argument("--true-newick", default="data/02_ground_truths/toy/T_true_rooted.nwk")
    parser.add_argument("--asr-state", default="data/02_ground_truths/toy/M_true.tsv")
    parser.add_argument("--stressor-gt", default="data/02_ground_truths/toy/A_true.json")
    parser.add_argument("--out-reports-root", default="data/04_audit_reports")
    parser.add_argument(
        "--agent-model",
        default="qwen/qwen3-14b",
        help="LLM model identifier in LM Studio for the Proposed 4-State Agent.",
    )
    parser.add_argument(
        "--zeroshot-model",
        default="qwen/qwen3-14b",
        help="LLM model identifier in LM Studio for AI Baseline 1 (Zero-Shot LLM) — can be any model in LM Studio.",
    )
    parser.add_argument(
        "--generic-agent-model",
        default="qwen/qwen3-14b",
        help="LLM model identifier in LM Studio for AI Baseline 2 (Generic Agent).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:1234/v1",
        help="LM Studio OpenAI-compatible server base URL (pass 'none' for offline deterministic test mode).",
    )
    parser.add_argument(
        "--require-live-llm",
        action="store_true",
        help="Fail immediately if the LM Studio server is unreachable.",
    )
    args = parser.parse_args()

    base_url = None if args.base_url.lower() == "none" else args.base_url

    print(f"[*] Running 3-Way Auditor Evaluation on {args.dataset_id} ({args.target_model})...")
    print(f"    - Proposed 4-State Agent LLM : {args.agent_model}")
    print(f"    - AI Baseline 1 Zero-Shot LLM: {args.zeroshot_model}")
    print(f"    - AI Baseline 2 Generic Agent: {args.generic_agent_model}")

    comp = run_three_way_comparison(
        dataset_id=args.dataset_id,
        target_model=args.target_model,
        fasta_path=args.fasta,
        pred_newick_path=args.pred_newick,
        pred_meta_path=args.pred_meta,
        true_newick_path=args.true_newick,
        asr_state_path=args.asr_state,
        stressor_gt_path=args.stressor_gt,
        out_reports_root=args.out_reports_root,
        agent_model=args.agent_model,
        zeroshot_model=args.zeroshot_model,
        generic_agent_model=args.generic_agent_model,
        lm_studio_base_url=base_url,
        require_live_llm=args.require_live_llm,
    )

    print("\n[+] 3-Way Comparison Summary:")
    print(f"{'Method':<26} | {'LLM Model':<20} | {'CCVR':<6} | {'LCR':<6} | {'G_tree':<6} | {'F1_site':<7}")
    print("-" * 82)
    for r in comp["summary_table"]:
        print(
            f"{r['auditor_method']:<26} | {r['llm_model_id']:<20} | "
            f"{r['ccvr']:<6.4f} | {r['lcr']:<6.4f} | {r['g_tree']:<6.4f} | {r['mean_feature_f1']:<7.4f}"
        )


if __name__ == "__main__":
    main()
