#!/usr/bin/env python3
"""
CLI Driver for Tier 3 Empirical 3-Engine Consensus Silver Standard Preparation
(Manuscript Section 4.1.1.1.2 & Table 4.5).

Processes real biological FASTA files (TreeBASE, OrthoMaM v12, SARS-CoV-2 / HIV-1)
by auto-detecting DNA vs. Protein alphabet, running IQ-TREE 2 (ModelFinder -m MFP),
RAxML-NG, and FastTree 2, and generating T_true_rooted.nwk, M_true.tsv, and A_true.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.empirical_consensus import build_empirical_tier3_ground_truth
from src.utils.io_cache import load_dataset_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Tier 3 Empirical 3-Engine Consensus Silver Standard (IQ-TREE 2 ModelFinder + RAxML-NG + FastTree 2)."
    )
    parser.add_argument("--fasta", help="Path to a single empirical biological FASTA file to process.")
    parser.add_argument("--dataset-id", help="Dataset identifier when --fasta is used.")
    parser.add_argument("--outgroup", default=None, help="Outgroup taxon name.")
    parser.add_argument(
        "--published-tree",
        default=None,
        help="Optional published reference Newick tree to cross-verify across all 3 ML engines.",
    )
    parser.add_argument(
        "--seq-type",
        choices=["AUTO", "DNA", "AA"],
        default="AUTO",
        help="Sequence alphabet (defaults to AUTO detection).",
    )
    parser.add_argument(
        "--recomb-breakpoints",
        default=None,
        help="Optional JSON list of known recombinant intervals, e.g. '[[400, 650]]'.",
    )
    parser.add_argument("--raw-tier3-dir", default="data/01_raw_alignments/tier3")
    parser.add_argument("--out-gt-root", default="data/02_ground_truths/tier3")
    parser.add_argument("--ufboot", type=int, default=1000)
    parser.add_argument("--raxml-bs", type=int, default=0)
    parser.add_argument("--no-modelfinder", action="store_true", help="Disable -m MFP and use fixed GTR+G4 / LG+G4.")
    args = parser.parse_args()

    cli_recomb_bps = json.loads(args.recomb_breakpoints) if args.recomb_breakpoints else None

    if args.fasta:
        fa_path = Path(args.fasta)
        ds_meta = load_dataset_metadata(fa_path)
        ds_id = args.dataset_id or ds_meta.get("dataset_id") or fa_path.stem
        out_dir = Path(args.out_gt_root) / ds_id
        pub_tree = args.published_tree or ds_meta.get("published_tree_path")
        if pub_tree and not Path(pub_tree).is_absolute() and (fa_path.parent / pub_tree).exists():
            pub_tree = str(fa_path.parent / pub_tree)
        anc_fa = ds_meta.get("experimental_ancestors_fasta")
        if anc_fa and not Path(anc_fa).is_absolute() and (fa_path.parent / anc_fa).exists():
            anc_fa = str(fa_path.parent / anc_fa)
        print(f"[*] Building Tier 3 3-Engine Consensus Silver Standard for {ds_id} ({fa_path})...")
        summary = build_empirical_tier3_ground_truth(
            fasta_path=str(fa_path),
            out_dir=str(out_dir),
            outgroup=args.outgroup or ds_meta.get("outgroup_taxon"),
            seq_type=args.seq_type,
            recombination_breakpoints=cli_recomb_bps if cli_recomb_bps is not None else ds_meta.get("recombination_breakpoints"),
            ufboot_replicates=args.ufboot,
            raxml_bs_trees=args.raxml_bs,
            use_modelfinder=(not args.no_modelfinder),
            published_tree_path=pub_tree,
            expected_stressors=ds_meta.get("expected_stressors"),
            documented_lba_taxa=ds_meta.get("documented_lba_taxa") or ds_meta.get("lba_long_taxa"),
            documented_clade_anomalies=ds_meta.get("documented_clade_anomalies"),
            experimental_ancestors_fasta=anc_fa,
            cataloged_mutations=ds_meta.get("cataloged_mutations"),
            kept_original_columns_1based=ds_meta.get("kept_original_columns_1based"),
        )
        print(
            f"[+] Completed {ds_id} ({summary['sequence_type']}, model={summary['best_fit_substitution_model']}): "
            f"{summary['three_engine_concordant_splits']}/{summary['total_internal_splits']} splits "
            f"concordant across all 3 engines ({summary['three_engine_concordance_rate']*100:.1f}%)."
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
        if "ancestors" in fa_path.stem.lower():
            continue
        ds_meta = load_dataset_metadata(fa_path)
        ds_id = ds_meta.get("dataset_id") or (fa_path.parent.name if fa_path.name == "alignment.fa" else fa_path.stem)
        out_dir = Path(args.out_gt_root) / ds_id
        pub_tree = args.published_tree or ds_meta.get("published_tree_path")
        if pub_tree and not Path(pub_tree).is_absolute() and (fa_path.parent / pub_tree).exists():
            pub_tree = str(fa_path.parent / pub_tree)
        anc_fa = ds_meta.get("experimental_ancestors_fasta")
        if anc_fa and not Path(anc_fa).is_absolute() and (fa_path.parent / anc_fa).exists():
            anc_fa = str(fa_path.parent / anc_fa)
        print(f"[*] Building Tier 3 3-Engine Consensus Silver Standard for {ds_id} ({fa_path})...")
        summary = build_empirical_tier3_ground_truth(
            fasta_path=str(fa_path),
            out_dir=str(out_dir),
            outgroup=args.outgroup or ds_meta.get("outgroup_taxon"),
            seq_type=args.seq_type,
            recombination_breakpoints=cli_recomb_bps if cli_recomb_bps is not None else ds_meta.get("recombination_breakpoints"),
            ufboot_replicates=args.ufboot,
            raxml_bs_trees=args.raxml_bs,
            use_modelfinder=(not args.no_modelfinder),
            published_tree_path=pub_tree,
            expected_stressors=ds_meta.get("expected_stressors"),
            documented_lba_taxa=ds_meta.get("documented_lba_taxa") or ds_meta.get("lba_long_taxa"),
            documented_clade_anomalies=ds_meta.get("documented_clade_anomalies"),
            experimental_ancestors_fasta=anc_fa,
            cataloged_mutations=ds_meta.get("cataloged_mutations"),
            kept_original_columns_1based=ds_meta.get("kept_original_columns_1based"),
        )
        print(
            f"[+] Completed {ds_id} ({summary['sequence_type']}, model={summary['best_fit_substitution_model']}): "
            f"{summary['three_engine_concordant_splits']}/{summary['total_internal_splits']} splits concordant."
        )


if __name__ == "__main__":
    main()
