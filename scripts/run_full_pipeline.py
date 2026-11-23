#!/usr/bin/env python3
"""
================================================================================
MASTER END-TO-END PIPELINE RUNNER (scripts/run_full_pipeline.py)
================================================================================
Executes the entire 4-Step Neuro-Symbolic Auditing Framework from start to finish:

  [Step 1/4] Ground-Truth Extraction (T_true_rooted.nwk, M_true.tsv, A_true.json)
             - Uses known T_true if provided, or runs Tier 3 3-Engine Consensus
               (IQ-TREE 2 ModelFinder + UFBoot 1000 + RAxML-NG + FastTree 2) if 'empirical'.
  [Step 2/4] Target Model Inference & Rules 1–3 Standardization
             - Runs Fusang (N <= 40), Phyloformer (+ BIONJ), or classical baselines with
               B = 100 non-parametric bootstrap replicates.
  [Step 3/4] 4-State LangGraph Diagnostic Agent Interrogation
  [Step 4/4] 3-Way Baseline Benchmarking & Statistical Validation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ai_baselines import run_three_way_comparison
from src.preprocessing.empirical_consensus import build_empirical_tier3_ground_truth
from src.preprocessing.ground_truth_gen import run_ground_truth_pipeline
from src.preprocessing.target_model_runner import run_target_model_pipeline
from src.utils.io_cache import (
    DatasetPaths,
    infer_tier_group_from_path,
    load_alignment_data,
    load_dataset_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified End-to-End 4-Step Pipeline Runner (Ground Truth -> Standardization -> 4-State Agent -> Statistical Evaluation)."
    )
    parser.add_argument("--dataset-id", default="TOY_DEV_15TAXA", help="Unique dataset identifier.")
    parser.add_argument(
        "--fasta",
        default="data/00_golden_toy/toy_alignment.fa",
        help="Path to input FASTA/PHYLIP alignment.",
    )
    parser.add_argument(
        "--true-tree",
        default="data/00_golden_toy/T_true.nwk",
        help="Path to generative T_true.nwk (pass 'empirical' to build Tier 3 3-Engine Consensus automatically).",
    )
    parser.add_argument("--outgroup", default="Taxon_01", help="Designated outgroup taxon name.")
    parser.add_argument(
        "--target-model",
        default="Fusang",
        choices=["Fusang", "Phyloformer", "IQ-TREE2", "RAxML-NG", "FastTree2"],
        help="Target model to audit.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=100,
        help="Number of non-parametric bootstrap replicates B (default: 100 per manuscript; pass --quick for B=10).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick development mode (sets B=10 bootstraps and fast empirical search if not overridden).",
    )
    parser.add_argument("--fusang-cmd", default=None, help="External single-MSA CLI template for Fusang.")
    parser.add_argument("--fusang-batch-cmd", default=None, help="External batch-dir CLI template for Fusang.")
    parser.add_argument("--phyloformer-cmd", default=None, help="External single-MSA CLI template for Phyloformer.")
    parser.add_argument("--phyloformer-batch-cmd", default=None, help="External batch-dir CLI template for Phyloformer.")
    parser.add_argument(
        "--use-surrogate",
        action="store_true",
        help="Explicitly allow built-in surrogate inference when real Fusang/Phyloformer CLI weights are not configured.",
    )
    parser.add_argument("--workers", type=int, default=None, help="CPU worker count for bootstrap replicates.")
    parser.add_argument("--agent-model", default="qwen/qwen3-14b", help="LM Studio model ID for Proposed Agent.")
    parser.add_argument("--zeroshot-model", default="qwen/qwen3-14b", help="LM Studio model ID for Baseline 1.")
    parser.add_argument("--generic-agent-model", default="qwen/qwen3-14b", help="LM Studio model ID for Baseline 2.")
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="LM Studio server URL.")
    parser.add_argument("--require-live-llm", action="store_true", help="Require active LM Studio connection.")
    parser.add_argument(
        "--tier-group",
        default="auto",
        help="Optional tier subdirectory ('tier1', 'tier2', 'tier3', '' or 'auto').",
    )
    parser.add_argument(
        "--offline-mock",
        action="store_true",
        help="Explicitly run deterministic offline mock synthesis without LM Studio.",
    )
    args = parser.parse_args()

    b_reps = 10 if (args.quick and args.bootstraps == 100) else args.bootstraps
    tier_group = (
        infer_tier_group_from_path(args.fasta)
        if args.tier_group == "auto"
        else args.tier_group
    )
    ds_paths = DatasetPaths(
        dataset_id=args.dataset_id,
        target_model=args.target_model,
        tier_group=tier_group,
    )
    gt_dir = ds_paths.ground_truth_dir
    std_prefix = str(ds_paths.standardized_prefix)
    reports_root = "data/04_audit_reports"
    base_url = None if (args.base_url.lower() == "none" or args.offline_mock) else args.base_url

    print("=" * 80)
    print(f" END-TO-END PHYLOGENETIC AGENTIC AUDIT PIPELINE: {args.dataset_id} ({args.target_model})")
    print("=" * 80)

    ds_meta = load_dataset_metadata(args.fasta)
    sim_recomb_bp = ds_meta.get("recombination_breakpoints") or None
    sim_expected_stressors = ds_meta.get("expected_stressors") or None
    doc_lba_taxa = ds_meta.get("documented_lba_taxa") or ds_meta.get("lba_long_taxa") or None
    doc_clade_anomalies = ds_meta.get("documented_clade_anomalies") or None
    sub_model = ds_meta.get("substitution_model") or None
    exp_anc_fa = ds_meta.get("experimental_ancestors_fasta") or None
    if exp_anc_fa and not Path(exp_anc_fa).is_absolute():
        cand_anc = Path(args.fasta).resolve().parent / exp_anc_fa
        if cand_anc.exists():
            exp_anc_fa = str(cand_anc)
    cat_muts = ds_meta.get("cataloged_mutations") or None
    kept_cols = ds_meta.get("kept_original_columns_1based") or None

    resolved_outgroup = args.outgroup
    if resolved_outgroup == "Taxon_01" and ds_meta.get("outgroup_taxon"):
        resolved_outgroup = ds_meta["outgroup_taxon"]

    msa_taxa_tup, _, _ = load_alignment_data(args.fasta)
    msa_taxa_sorted = sorted(msa_taxa_tup)
    if resolved_outgroup not in msa_taxa_sorted:
        fallback_og = msa_taxa_sorted[0]
        print(
            f"[!] Warning: Requested outgroup '{resolved_outgroup}' not found in {args.fasta}; "
            f"falling back to '{fallback_og}'. Pass --outgroup <taxon_name> for exact biological rooting."
        )
        resolved_outgroup = fallback_og

    print(f"\n[Step 1/4] Extracting Ground Truths (T_true_rooted.nwk, M_true.tsv, A_true.json) -> {gt_dir}")
    if args.true_tree.lower() == "empirical":
        pub_tree = ds_meta.get("published_tree_path")
        if pub_tree and not Path(pub_tree).is_absolute():
            cand_pt = Path(args.fasta).resolve().parent / pub_tree
            if cand_pt.exists():
                pub_tree = str(cand_pt)
        build_empirical_tier3_ground_truth(
            fasta_path=args.fasta,
            out_dir=str(gt_dir),
            outgroup=resolved_outgroup,
            ufboot_replicates=0 if args.quick else 1000,
            recombination_breakpoints=sim_recomb_bp,
            published_tree_path=pub_tree,
            expected_stressors=sim_expected_stressors,
            documented_lba_taxa=doc_lba_taxa,
            documented_clade_anomalies=doc_clade_anomalies,
        )
    else:
        run_ground_truth_pipeline(
            fasta_path=args.fasta,
            true_tree_path=args.true_tree,
            out_dir=str(gt_dir),
            outgroup=resolved_outgroup,
            recombination_breakpoints=sim_recomb_bp,
            expected_stressors=sim_expected_stressors,
            substitution_model=sub_model,
            experimental_ancestors_fasta=exp_anc_fa,
            cataloged_mutations=cat_muts,
            kept_original_columns_1based=kept_cols,
            documented_lba_taxa=doc_lba_taxa,
            documented_clade_anomalies=doc_clade_anomalies,
        )
    print("           ✓ Ground truths ready.")

    a_true_file = gt_dir / "A_true.json"
    if a_true_file.exists() and not sub_model:
        sub_model = json.loads(a_true_file.read_text()).get("substitution_model")

    print(f"\n[Step 2/4] Running {args.target_model} (B={b_reps} bootstraps) & Rules 1-3 Standardization...")
    std_meta = run_target_model_pipeline(
        fasta_path=args.fasta,
        target_model=args.target_model,
        output_prefix=std_prefix,
        outgroup_taxon=resolved_outgroup,
        b_replicates=b_reps,
        fusang_cmd_template=args.fusang_cmd,
        fusang_batch_cmd_template=args.fusang_batch_cmd,
        phyloformer_cmd_template=args.phyloformer_cmd,
        phyloformer_batch_cmd_template=args.phyloformer_batch_cmd,
        allow_surrogate=args.use_surrogate,
        max_workers=args.workers,
        substitution_model=sub_model,
    )
    if std_meta is None:
        print(f"[!] {args.target_model} skipped (N > 40 scaling boundary).")
        return
    pred_nwk = f"{std_prefix}_normalized.nwk"
    pred_meta = f"{std_prefix}_metadata.json"
    print(f"           ✓ Standardized tree saved -> {pred_nwk} ({std_meta['total_internal_nodes']} internal nodes)")

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
        allow_offline_mock=args.offline_mock,
    )

    print("\n" + "=" * 102)
    print(
        f"{'Method':<26} | {'LLM Model':<28} | {'CCVR':<6} | {'LCR':<6} | {'EDR':<6} | {'G_tree':<6} | {'F1_site':<7}"
    )
    print("-" * 102)
    for r in comp["summary_table"]:
        print(
            f"{r['auditor_method']:<26} | {r['llm_model_id'][:28]:<28} | "
            f"{r['ccvr']:<6.4f} | {r['lcr']:<6.4f} | {r['edr']:<6.4f} | {r['g_tree']:<6.4f} | {r['mean_feature_f1']:<7.4f}"
        )
    print("=" * 102)
    print(f"[+] All JSON Audit Reports & Statistical Tables saved under: {reports_root}/")


if __name__ == "__main__":
    main()
