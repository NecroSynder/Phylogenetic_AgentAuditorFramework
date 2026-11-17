#!/usr/bin/env python3
"""
CLI Driver for Target Model & Classical Baseline Batch Inference + Standardization
(Manuscript Section 4.1.1 Item 4, Table 4.2 & Pipeline Step 2).

Runs Fusang (N <= 40), Phyloformer (with BIONJ wrapper), and/or classical baselines
(IQ-TREE2, RAxML-NG, FastTree2) with B = 100 non-parametric bootstrap replicates and
Rules 1-3 standardization into data/03_standardized_trees/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.target_model_runner import (
    detect_sequence_alphabet,
    run_target_model_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Target Models (Fusang, Phyloformer) & Classical Baselines with B=100 Bootstraps and Rules 1-3 Standardization."
    )
    parser.add_argument("--fasta", help="Optional single FASTA file to run inference on.")
    parser.add_argument("--dataset-id", help="Dataset identifier when --fasta is specified.")
    parser.add_argument("--outgroup", default=None, help="Outgroup taxon name.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Fusang", "Phyloformer"],
        choices=["Fusang", "Phyloformer", "IQ-TREE2", "RAxML-NG", "FastTree2"],
    )
    parser.add_argument("--bootstraps", type=int, default=100)
    parser.add_argument("--raw-root", default="data/01_raw_alignments")
    parser.add_argument("--out-std-root", default="data/03_standardized_trees")
    parser.add_argument("--fusang-cmd", default=None, help="Single-MSA CLI template for Fusang.")
    parser.add_argument(
        "--fusang-batch-cmd",
        default=None,
        help="Batch-directory CLI template for Fusang (see scripts/adapters/run_fusang_cli.py).",
    )
    parser.add_argument("--phyloformer-cmd", default=None, help="Single-MSA CLI template for Phyloformer.")
    parser.add_argument(
        "--phyloformer-batch-cmd",
        default=None,
        help="Batch-directory GPU CLI template for Phyloformer (see scripts/adapters/run_phyloformer_cli.py).",
    )
    parser.add_argument(
        "--use-surrogate",
        action="store_true",
        help="Explicitly allow built-in surrogate execution when real Fusang/Phyloformer CLI weights are not configured.",
    )
    parser.add_argument("--workers", type=int, default=None, help="CPU worker count for classical bootstraps.")
    args = parser.parse_args()

    tasks = []
    if args.fasta:
        fa_path = Path(args.fasta)
        ds_id = args.dataset_id or fa_path.parent.name
        tasks.append(
            {"dataset_id": ds_id, "tier_dir": "custom", "fasta_path": fa_path, "outgroup": args.outgroup}
        )
    else:
        raw_root = Path(args.raw_root)
        for fa_path in sorted(raw_root.rglob("alignment.fa")):
            ds_id = fa_path.parent.name
            tier_dir = fa_path.parent.parent.name
            sim_meta_path = fa_path.parent / "simulation_metadata.json"
            outgroup = args.outgroup
            if outgroup is None and sim_meta_path.exists():
                with open(sim_meta_path) as f:
                    outgroup = json.load(f).get("outgroup_taxon")
            tasks.append(
                {"dataset_id": ds_id, "tier_dir": tier_dir, "fasta_path": fa_path, "outgroup": outgroup}
            )

    if not tasks:
        print("[!] No alignments found to process.")
        return

    run_records = []
    for item in tasks:
        ds_id = item["dataset_id"]
        tier_dir = item["tier_dir"]
        fa_path = item["fasta_path"]
        outgroup = item["outgroup"]
        seq_alpha = detect_sequence_alphabet(str(fa_path))

        for model_name in args.models:
            if not args.use_surrogate:
                if model_name == "Fusang" and seq_alpha != "DNA":
                    print(f"    [Skipped] Fusang skipped for {ds_id} (DNA-only model on {seq_alpha} alignment).")
                    continue
                if model_name == "Phyloformer" and seq_alpha != "AA":
                    print(f"    [Skipped] Phyloformer skipped for {ds_id} (AA-only model on {seq_alpha} alignment).")
                    continue
            out_prefix = (
                str(Path(args.out_std_root) / ds_id / model_name)
                if tier_dir == "custom"
                else str(Path(args.out_std_root) / tier_dir / ds_id / model_name)
            )
            print(f"[*] Running {model_name} (B={args.bootstraps}) on {ds_id}...")
            meta = run_target_model_pipeline(
                fasta_path=str(fa_path),
                target_model=model_name,
                output_prefix=out_prefix,
                outgroup_taxon=outgroup,
                b_replicates=args.bootstraps,
                fusang_cmd_template=args.fusang_cmd,
                fusang_batch_cmd_template=args.fusang_batch_cmd,
                phyloformer_cmd_template=args.phyloformer_cmd,
                phyloformer_batch_cmd_template=args.phyloformer_batch_cmd,
                allow_surrogate=args.use_surrogate,
                max_workers=args.workers,
            )
            if meta is None:
                print(f"    [Skipped] {model_name} skipped for {ds_id} (N > 40 scaling constraint).")
                continue
            print(
                f"    [+] Standardized {model_name} -> {out_prefix}_normalized.nwk "
                f"({meta['total_internal_nodes']} internal nodes, backend={meta['inference_backend']})"
            )
            run_records.append(
                {
                    "dataset_id": ds_id,
                    "tier_dir": tier_dir,
                    "target_model": model_name,
                    "inference_backend": meta["inference_backend"],
                    "normalized_newick": f"{out_prefix}_normalized.nwk",
                    "metadata_json": f"{out_prefix}_metadata.json",
                    "total_internal_nodes": meta["total_internal_nodes"],
                    "branch_lengths_present": meta["branch_lengths_present"],
                }
            )

    summary_path = Path(args.out_std_root) / "standardized_runs_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"total_runs": len(run_records), "runs": run_records}, f, indent=2)
    print(f"[+] Saved standardized runs manifest ({len(run_records)} model runs) -> {summary_path}")


if __name__ == "__main__":
    main()
