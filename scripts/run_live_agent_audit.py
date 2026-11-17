#!/usr/bin/env python3
"""
Live CLI Runner for Pipeline Step 3 (LangGraph + Local Qwen 2.5 / Qwen 3 via LM Studio).
Usage:
    python scripts/run_live_agent_audit.py --base-url http://localhost:1234/v1 --model qwen/qwen3-14b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.diagnostic_agent import run_diagnostic_audit
from src.preprocessing.ground_truth_gen import run_ground_truth_pipeline
from src.preprocessing.normalizer_wrapper import standardize_and_root_tree
from src.utils.io_cache import resolve_lm_studio_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Live LangGraph + LM Studio Phylogenetic Diagnostic Audit")
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="LM Studio OpenAI-compatible server URL")
    parser.add_argument("--model", default=None, help="Model identifier in LM Studio (auto-detected if omitted)")
    parser.add_argument(
        "--output",
        default="data/04_audit_reports/live_lmstudio_toy_audit_report.json",
        help="Path to write validated JSON Audit Report",
    )
    args = parser.parse_args()

    print(f"[1/4] Checking LM Studio server at {args.base_url} ...")
    try:
        resolved_base_url, loaded_models = resolve_lm_studio_url(args.base_url)
    except Exception as exc:
        print(
            f"❌ Could not reach LM Studio server ({exc}).\n"
            "   Please open LM Studio -> Developer / Local Server tab ->\n"
            "   Enable 'Serve on Local Network' (for WSL2 access) -> Load Qwen -> Click 'Start Server' (port 1234)."
        )
        sys.exit(1)

    if not loaded_models:
        print("❌ LM Studio server is running, but no model is currently loaded in memory.")
        sys.exit(1)

    model_id = args.model or loaded_models[0]
    print(f"✅ Connected to LM Studio at {resolved_base_url}! Active model: {model_id}")

    gt_dir = Path("data/02_ground_truths/TOY_DEV_15TAXA")
    std_prefix = Path("data/03_standardized_trees/TOY_DEV_15TAXA/Fusang")
    pred_nwk = Path(f"{std_prefix}_normalized.nwk")
    pred_meta = Path(f"{std_prefix}_metadata.json")

    print("[2/4] Ensuring Golden Toy Ground Truths (Step 1) and Standardized Tree (Step 2) exist...")
    if not (gt_dir / "M_true.tsv").exists():
        run_ground_truth_pipeline(
            "data/00_golden_toy/toy_alignment.fa",
            "data/00_golden_toy/T_true.nwk",
            str(gt_dir),
        )
    if not pred_nwk.exists():
        nwk_str = (gt_dir / "T_true_rooted.nwk").read_text().strip()
        standardize_and_root_tree(
            nwk_str,
            [nwk_str] * 10,
            "Taxon_01",
            "Fusang",
            str(std_prefix),
        )

    print(f"[3/4] Running LangGraph 4-State Diagnostic Agent with live LLM ({model_id})...")
    report = run_diagnostic_audit(
        dataset_id="TOY_DEV_15TAXA",
        target_model="Fusang",
        fasta_path="data/00_golden_toy/toy_alignment.fa",
        pred_newick_path=str(pred_nwk),
        pred_meta_path=str(pred_meta),
        true_newick_path=str(gt_dir / "T_true_rooted.nwk"),
        asr_state_path=str(gt_dir / "M_true.tsv"),
        stressor_gt_path=str(gt_dir / "A_true.json"),
        output_json_path=args.output,
        orchestrator_llm=model_id,
        lm_studio_base_url=resolved_base_url,
        require_live_llm=True,
        allow_offline_mock=False,
    )

    gm = report["global_metrics"]
    print("\n[4/4] ✅ Live Audit Complete! Validated JSON Report Summary:")
    print(f"  • Output File:          {args.output}")
    print(f"  • Nodes Audited:        {gm['total_internal_nodes_audited']}")
    print(f"  • Normalized RF (Eq.1): {gm['normalized_rf_distance']}")
    print(f"  • Global FCI (Eq.2):    {gm['false_confidence_index_fci']}")
    print(f"  • Global EDR (Eq.4):    {gm['error_detection_rate_edr']}")
    print(f"  • Global CCVR (Eq.8):   {gm['claim_to_coordinate_verification_rate']}")
    print(f"  • Global LCR (Eq.10):   {gm['logical_consistency_rate_lcr']}")
    print(f"  • Tree Goodness G_tree: {gm['overall_tree_goodness_score']}")
    print("\nSample Node N1 Causal Explanation from Live LLM:")
    print(f"  \"{report['node_audits'][0]['diagnosed_failure_mode']['agent_causal_explanation']}\"")


if __name__ == "__main__":
    main()
