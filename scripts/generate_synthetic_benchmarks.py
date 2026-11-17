#!/usr/bin/env python3
"""
CLI Driver for Synthetic Benchmark Generation (Manuscript Section 4.2.2, Table 4.1 & Table 4.5).

Generates:
  - Tier 1 (Medium-Scale Indel, LBA & Scaling Boundary):
      N_taxa in {15, 30, 40, 50}, L = 1000, DNA (GTR+G4) & Protein (LG+G4)
      Native AliSim CTMC Zipfian indels (POW{1.6/30}) at gap proportions G in {0.05, 0.15, 0.30}
      Terminal LBA multipliers k in {2, 10, 40}
  - Tier 2 (Large-Scale Short-Branch, LBA & Recombination):
      N_taxa in {50, 100, 200}, L in {1000, 2000}, DNA & Protein
      Internal branch lengths l_int = 0.002 (< 0.005 substitutions/site)
      Extreme gamma rate heterogeneity alpha = 0.2
      Localized recombinant block = 15% of alignment length L
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.synthetic_generator import generate_single_synthetic_dataset


def build_tier1_configs(
    replicates: int = 1,
    seq_type: str = "DNA",
    quick_smoke: bool = False,
) -> list[dict]:
    if quick_smoke:
        taxa_grid = [15, 40]
        gap_grid = [0.15]
        lba_grid = [10.0]
    else:
        taxa_grid = [15, 30, 40, 50]
        gap_grid = [0.05, 0.15, 0.30]
        lba_grid = [2.0, 10.0, 40.0]

    configs = []
    seed_counter = 1000 if seq_type.upper() == "DNA" else 3000
    for n_taxa in taxa_grid:
        for gap_prop in gap_grid:
            for k_mult in lba_grid:
                for rep in range(1, replicates + 1):
                    seed_counter += 1
                    gap_pct = int(round(gap_prop * 100))
                    k_int = int(round(k_mult))
                    ds_id = f"T1_{seq_type.upper()}_N{n_taxa}_L1000_G{gap_pct}_K{k_int}_R{rep:02d}"
                    configs.append(
                        {
                            "dataset_id": ds_id,
                            "tier": 1,
                            "n_taxa": n_taxa,
                            "seq_len": 1000,
                            "seq_type": seq_type.upper(),
                            "gap_proportion": gap_prop,
                            "lba_multiplier": k_mult,
                            "short_internal_len": 0.015,
                            "recomb_fraction": 0.0,
                            "seed": seed_counter,
                        }
                    )
    return configs


def build_tier2_configs(
    replicates: int = 1,
    seq_type: str = "DNA",
    quick_smoke: bool = False,
) -> list[dict]:
    if quick_smoke:
        taxa_grid = [50]
        len_grid = [1000]
        lba_grid = [10.0]
    else:
        taxa_grid = [50, 100, 200]
        len_grid = [1000, 2000]
        lba_grid = [10.0, 40.0]

    configs = []
    seed_counter = 5000 if seq_type.upper() == "DNA" else 7000
    for n_taxa in taxa_grid:
        for seq_len in len_grid:
            for k_mult in lba_grid:
                for rep in range(1, replicates + 1):
                    seed_counter += 1
                    k_int = int(round(k_mult))
                    ds_id = f"T2_{seq_type.upper()}_N{n_taxa}_L{seq_len}_K{k_int}_RECOMB15_R{rep:02d}"
                    configs.append(
                        {
                            "dataset_id": ds_id,
                            "tier": 2,
                            "n_taxa": n_taxa,
                            "seq_len": seq_len,
                            "seq_type": seq_type.upper(),
                            "gap_proportion": 0.0,
                            "lba_multiplier": k_mult,
                            "short_internal_len": 0.002,
                            "recomb_fraction": 0.15,
                            "seed": seed_counter,
                        }
                    )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Tier 1 & Tier 2 Synthetic Stressor Benchmarks via IQ-TREE 2 AliSim."
    )
    parser.add_argument("--tier", choices=["1", "2", "both"], default="both")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument(
        "--seq-type",
        choices=["DNA", "AA", "both"],
        default="DNA",
        help="Sequence alphabet ('DNA', 'AA', or 'both' per Table 4.5).",
    )
    parser.add_argument("--quick-smoke", action="store_true")
    parser.add_argument("--compute-ground-truths", action="store_true")
    parser.add_argument("--out-raw-root", default="data/01_raw_alignments")
    parser.add_argument("--out-gt-root", default="data/02_ground_truths")
    args = parser.parse_args()

    alphabets = ["DNA", "AA"] if args.seq_type == "both" else [args.seq_type]
    configs = []
    for st in alphabets:
        if args.tier in ("1", "both"):
            configs.extend(
                build_tier1_configs(
                    replicates=args.replicates,
                    seq_type=st,
                    quick_smoke=args.quick_smoke,
                )
            )
        if args.tier in ("2", "both"):
            configs.extend(
                build_tier2_configs(
                    replicates=args.replicates,
                    seq_type=st,
                    quick_smoke=args.quick_smoke,
                )
            )

    manifest = []
    print(f"[*] Generating {len(configs)} synthetic benchmark dataset(s)...")
    for idx, cfg in enumerate(configs, start=1):
        ds_id = cfg["dataset_id"]
        tier_num = cfg["tier"]
        raw_dir = Path(args.out_raw_root) / f"tier{tier_num}" / ds_id
        gt_dir = Path(args.out_gt_root) / f"tier{tier_num}" / ds_id
        print(f"  [{idx}/{len(configs)}] Simulating {ds_id} -> {raw_dir}")
        meta = generate_single_synthetic_dataset(
            dataset_id=ds_id,
            tier=tier_num,
            n_taxa=cfg["n_taxa"],
            seq_len=cfg["seq_len"],
            out_raw_dir=str(raw_dir),
            seq_type=cfg["seq_type"],
            gap_proportion=cfg["gap_proportion"],
            lba_multiplier=cfg["lba_multiplier"],
            short_internal_len=cfg["short_internal_len"],
            recomb_fraction=cfg["recomb_fraction"],
            seed=cfg["seed"],
            compute_ground_truth=args.compute_ground_truths,
            out_gt_dir=str(gt_dir),
        )
        manifest.append(meta)

    manifest_path = Path(args.out_raw_root) / "benchmark_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({"total_datasets": len(manifest), "datasets": manifest}, f, indent=2)
    print(f"[+] Saved master benchmark manifest ({len(manifest)} datasets) -> {manifest_path}")


if __name__ == "__main__":
    main()
