#!/usr/bin/env python3
"""
CLI Driver for Tier 3 Empirical 3-Engine Consensus Silver Standard Preparation
(Manuscript Section 4.2.2 & Table 4.1).

Processes real biological FASTA files (TreeBASE, OrthoMaM v12, SARS-CoV-2 / HIV-1)
by running IQ-TREE 2, RAxML-NG, and FastTree 2 to establish the 3-engine consensus
reference tree and generating T_true_rooted.nwk, M_true.tsv, and A_true.json.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.empirical_consensus import build_empirical_tier3_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Tier 3 Empirical 3-Engine Consensus Silver Standard (IQ-TREE 2 + RAxML-NG + FastTree 2)."
    )
    parser.add_argument(
        "--fasta",
        help="Path to a single empirical biological FASTA file to process.",
    )
    parser.add_argument(
        "--dataset-id",
        help="Dataset identifier when --fasta is used (defaults to FASTA stem).",
    )
    parser.add_argument(
        "--outgroup",
        default=None,
        help="Outgroup taxon name (defaults to lexicographically first taxon).",
    )
    parser.add_argument(
        "--recomb-breakpoints",
        default=None,
        help="Optional JSON list of known recombinant intervals, e.g. '[[400, 650]]'.",
    )
    parser.add_argument(
        "--raw-tier3-dir",
        default="data/01_raw_alignments/tier3",
        help="Directory to scan for empirical FASTA files when --fasta is omitted.",
    )
    parser.add_argument(
        "--out-gt-root",
        default="data/02_ground_truths/tier3",
        help="Output root directory for Tier 3 ground truths.",
    )
    parser.add_argument(
        "--ufboot",
        type=int,
        default=1000,
        help="Number of UFBoot replicates for IQ-TREE 2 (set 0 for fast ML search without UFBoot).",
    )
    parser.add_argument(
        "--raxml-bs",
        type=int,
        default=0,
        help="Number of RAxML-NG bootstrap trees (0 = ML search + IQ-TREE/FastTree support check).",
    )
    args = parser.parse_args()

    recomb_bps = json.loads(args.recomb_breakpoints) if args.recomb_breakpoints else []

    if args.fasta:
        fa_path = Path(args.fasta)
        ds_id = args.dataset_id or fa_path.stem
        out_dir = Path(args.out_gt_root) / ds_id
        print(f"[*] Building Tier 3 3-Engine Consensus Silver Standard for {ds_id} ({fa_path})...")
        summary = build_empirical_tier3_ground_truth(
            fasta_path=str(fa_path),
            out_dir=str(out_dir),
            outgroup=args.outgroup,
            recombination_breakpoints=recomb_bps,
            ufboot_replicates=args.ufboot,
            raxml_bs_trees=args.raxml_bs,
        )
        print(
            f"[+] Completed {ds_id}: {summary['three_engine_concordant_splits']}/"
            f"{summary['total_internal_splits']} splits concordant across all 3 engines "
            f"({summary['three_engine_concordance_rate']*100:.1f}%)."
        )
        return

    raw_root = Path(args.raw_tier3_dir)
    if not raw_root.exists():
        print(f"[!] Directory {raw_root} does not exist yet. Place empirical FASTAs there or pass --fasta.")
        return

    fasta_files = sorted(
        list(raw_root.rglob("*.fa")) + list(raw_root.rglob("*.fasta")) + list(raw_root.rglob("*.fas"))
    )
    if not fasta_files:
        print(f"[!] No FASTA files found under {raw_root}.")
        return

    for fa_path in fasta_files:
        ds_id = fa_path.parent.name if fa_path.name == "alignment.fa" else fa_path.stem
        out_dir = Path(args.out_gt_root) / ds_id
        print(f"[*] Building Tier 3 3-Engine Consensus Silver Standard for {ds_id} ({fa_path})...")
        summary = build_empirical_tier3_ground_truth(
            fasta_path=str(fa_path),
            out_dir=str(out_dir),
            outgroup=args.outgroup,
            recombination_breakpoints=recomb_bps,
            ufboot_replicates=args.ufboot,
            raxml_bs_trees=args.raxml_bs,
        )
        print(
            f"[+] Completed {ds_id}: {summary['three_engine_concordant_splits']}/"
            f"{summary['total_internal_splits']} splits concordant across all 3 engines."
        )


if __name__ == "__main__":
    main()
