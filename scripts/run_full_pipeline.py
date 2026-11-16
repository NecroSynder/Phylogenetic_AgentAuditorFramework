#!/usr/bin/env python3
"""
================================================================================
MASTER END-TO-END PIPELINE RUNNER (scripts/run_full_pipeline.py)
================================================================================
Executes the entire 4-Step Neuro-Symbolic Auditing Framework from start to finish
with clear, step-by-step console tracking:

  [Step 1/4] Ground-Truth Extraction (T_true_rooted.nwk, M_true.tsv, A_true.json)
             - Uses known T_true if provided, or runs Tier 3 3-Engine Consensus
               (IQ-TREE 2 + RAxML-NG + FastTree 2) if no T_true is provided.
  [Step 2/4] Target Model Inference & Rules 1–3 Standardization
             - Runs Fusang (N <= 40) and/or Phyloformer (+ BIONJ) with B = 100
               non-parametric bootstrap replicates.
  [Step 3/4] 4-State LangGraph Diagnostic Agent Interrogation
             - Audits every internal split N1..Nk and enforces State 4 coordinate
               verification (CCVR == 1.0).
  [Step 4/4] 3-Way Baseline Benchmarking & Statistical Validation
             - Compares Proposed 4-State Agent vs. Baseline 1 (Zero-Shot LLM) and
               Baseline 2 (Generic Agent) via Wilcoxon, McNemar, and BH-FDR tests.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ai_baselines import run_three_way_comparison
from src.preprocessing.empirical_consensus import build_empirical_tier3_ground_truth
from src.preprocessing.ground_truth_gen import run_ground_truth_pipeline
from src.preprocessing.target_model_runner import run_target_model_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified End-to-End 4-Step Pipeline Runner (Ground Truth -> Standardization -> 4-State Agent -> Statistical Evaluation)."
    )
    parser.add_argument(
        "--dataset-id",
        default="TOY_DEV_15TAXA",
        help="Unique dataset identifier.",
    )
    parser.add_argument(
        "--fasta",
        default="data/00_golden_toy/toy_alignment.fa",
        help="Path to input FASTA alignment.",
    )
    parser.add_argument(
        "--true-tree",
        default="data/00_golden_toy/T_true.nwk",
        help="Path to generative T_true.nwk (pass 'empirical' to build Tier 3 3-Engine Consensus automatically).",
    )
    parser.add_argument(
        "--outgroup",
        default="Taxon_01",
        help="Designated outgroup taxon name for universal rooting.",
    )
    parser.add_argument(
        "--target-model",
        default="Fusang",
        choices=["Fusang", "Phyloformer"],
        help="Target deep-learning model to audit.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=10,
        help="Number of non-parametric bootstrap replicates B (use 100 for production runs, 10 for quick runs).",
    )
    parser.add_argument(
        "--agent-model",
        default="qwen/qwen3-14b",
        help="LM Studio model ID for the Proposed 4-State Agent.",
    )
    parser.add_argument(
        "--zeroshot-model",
        default="qwen/qwen3-14b",
        help="LM Studio model ID for AI Baseline 1 (Zero-Shot LLM).",
    )
    parser.add_argument(
        "--generic-agent-model",
        default="qwen/qwen3-14b",
        help="LM Studio model ID for AI Baseline 2 (Generic Agent).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:1234/v1",
        help="LM Studio server URL (pass 'none' for offline deterministic mode).",
    )
    parser.add_argument(
        "--require-live-llm",
        action="store_true",
        help="Require active LM Studio connection.",
    )
    args = parser.parse_args()

    gt_dir = Path("data/02_ground_truths") / args.dataset_id
    std_prefix = str(Path("data/03_standardized_trees") / args.dataset_id / args.target_model)
    reports_root = "data/04_audit_reports"
    base_url = None if args.base_url.lower() == "none" else args.base_url

    print("=" * 80)
    print(f" END-TO-END PHYLOGENETIC AGENTIC AUDIT PIPELINE: {args.dataset_id} ({args.target_model})")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # STEP 1: Ground-Truth & Evolutionary Feature Extraction
    # --------------------------------------------------------------------------
    print(f"\n[Step 1/4] Extracting Ground Truths (T_true_rooted.nwk, M_true.tsv, A_true.json) -> {gt_dir}")
    if args.true_tree.lower() == "empirical":
        build_empirical_tier3_ground_truth(
            fasta_path=args.fasta,
            out_dir=str(gt_dir),
            outgroup=args.outgroup,
            ufboot_replicates=0,
        )
    else:
        run_ground_truth_pipeline(
            fasta_path=args.fasta,
            true_tree_path=args.true_tree,
            out_dir=str(gt_dir),
            outgroup=args.outgroup,
        )
    print("           ✓ Ground truths ready.")

    # --------------------------------------------------------------------------
    # STEP 2: Target Model Inference, B Bootstraps & Rules 1-3 Standardization
    # --------------------------------------------------------------------------
    print(f"\n[Step 2/4] Running {args.target_model} (B={args.bootstraps} bootstraps) & Rules 1-3 Standardization...")
    std_meta = run_target_model_pipeline(
        fasta_path=args.fasta,
        target_model=args.target_model,
        output_prefix=std_prefix,
        outgroup_taxon=args.outgroup,
        b_replicates=args.bootstraps,
    )
    if std_meta is None:
        print(f"[!] {args.target_model} skipped (N > 40 scaling boundary).")
        return
    pred_nwk = f"{std_prefix}_normalized.nwk"
    pred_meta = f"{std_prefix}_metadata.json"
    print(f"           ✓ Standardized tree saved -> {pred_nwk} ({std_meta['total_internal_nodes']} internal nodes)")

    # --------------------------------------------------------------------------
    # STEP 3 & STEP 4: 4-State Diagnostic Agent + 3-Way Baseline Evaluation
    # --------------------------------------------------------------------------
    print("\n[Step 3/4 & Step 4/4] Running 4-State Diagnostic Agent, Baselines 1 & 2, and Statistical Validation...")
    comp = run_three_way_comparison(
        dataset_id=args.dataset_id,
        target_model=args.target_model,
        fasta_path=args.fasta,
        pred_newick_path=pred_nwk,
        pred_meta_path=pred_meta,
        true_newick_path=str(gt_dir / "T_true_rooted.nwk"),
        asr_state_path=str(gt_dir / "M_true.tsv"),
        stressor_gt_path=str(gt_dir / "A_true.json"),
        out_reports_root=reports_root,
        agent_model=args.agent_model,
        zeroshot_model=args.zeroshot_model,
        generic_agent_model=args.generic_agent_model,
        lm_studio_base_url=base_url,
        require_live_llm=args.require_live_llm,
    )

    print("\n" + "=" * 82)
    print(f"{'Method':<26} | {'LLM Model':<20} | {'CCVR':<6} | {'LCR':<6} | {'G_tree':<6} | {'F1_site':<7}")
    print("-" * 82)
    for r in comp["summary_table"]:
        print(
            f"{r['auditor_method']:<26} | {r['llm_model_id']:<20} | "
            f"{r['ccvr']:<6.4f} | {r['lcr']:<6.4f} | {r['g_tree']:<6.4f} | {r['mean_feature_f1']:<7.4f}"
        )
    print("=" * 82)
    print(f"[+] All JSON Audit Reports & Statistical Tables saved under: {reports_root}/")


if __name__ == "__main__":
    main()
