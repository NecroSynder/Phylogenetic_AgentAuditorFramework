#!/usr/bin/env python3
"""
================================================================================
RESUMABLE EMPIRICAL CORPUS RUNNER (STAGE A + STAGE B)
(Proposal Sections 6.2.1 & 6.3.1 — N = 45 Empirical Datasets)
================================================================================
Executes the complete empirical validation protocol across Tiers 1-3 with
file-level checkpointing so interrupted runs resume cleanly without losing progress:

  - Stage A:
      1. Extracts Ground Truths (T_true_rooted.nwk, M_true.tsv, A_true.json),
         using sequenced ancestors / cataloged mutations (Tiers 1-2) and/or 3-engine
         ModelFinder + UFBoot2 consensus cross-verification (Tiers 2-3).
      2. Runs Target Neural Models (Fusang for DNA N <= 40; Phyloformer for AA)
         alongside Classical ML References (IQ-TREE2, RAxML-NG, FastTree2) with
         B = 100 non-parametric bootstrap replicates.
      3. Computes Stage A topological & false-confidence tables and paired
         Wilcoxon + Benjamini-Hochberg FDR tests (stage_a_statistical_suite.json).

  - Stage B:
      1. Evaluates the Proposed 4-State Verification-Gated Agent alongside
         Baseline 1 (Zero-Shot LLM) and Baseline 2 (Unconstrained Window Agent)
         on local Qwen3-14B (temperature = 0.0).
      2. Syncs both new and resumed checkpoint reports into master_benchmark_metrics_table.tsv
         and computes paired Wilcoxon + McNemar + Benjamini-Hochberg FDR tests
         scoped strictly to the active corpus (replicate_level_statistical_suite.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ai_baselines import (
    _sanitize_model_tag,
    run_three_way_comparison,
    sync_checkpoint_reports_to_master_tables,
)
from src.evaluation.metrics import (
    evaluate_stage_a_model_row,
    run_stage_a_statistical_suite,
)
from src.preprocessing.empirical_consensus import build_empirical_tier3_ground_truth
from src.preprocessing.ground_truth_gen import (
    detect_sequence_alphabet,
    run_ground_truth_pipeline,
)
from src.preprocessing.target_model_runner import run_target_model_pipeline
from src.utils.io_cache import (
    DatasetPaths,
    infer_tier_group_from_path,
    load_alignment_data,
    load_dataset_metadata,
)


def discover_empirical_datasets(
    raw_root: Path,
    manifest_path: Path | None = None,
    include_synthetic: bool = False,
) -> list[dict]:
    """Discovers curated empirical datasets under raw_root/{tier1,tier2,tier3} or from manifest_path."""
    if manifest_path and manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        entries = data.get("datasets", data if isinstance(data, list) else [])
        discovered = []
        for item in entries:
            if (not include_synthetic) and item.get("is_synthetic", False):
                continue
            fa_rel = item.get("fasta_path")
            if not fa_rel:
                continue
            fa_p = Path(fa_rel) if Path(fa_rel).is_absolute() else (PROJECT_ROOT / fa_rel)
            if fa_p.exists():
                discovered.append(
                    {
                        "dataset_id": item["dataset_id"],
                        "tier_group": item.get("tier_group") or infer_tier_group_from_path(fa_p),
                        "fasta_path": fa_p,
                        "meta": {**load_dataset_metadata(fa_p), **item},
                    }
                )
        return discovered

    discovered = []
    for tier_name in ("tier1", "tier2", "tier3"):
        t_dir = raw_root / tier_name
        if not t_dir.exists():
            continue
        for fa_p in sorted(t_dir.rglob("alignment.fa")):
            # Exclude synthetic AliSim fixtures (which carry simulation_metadata.json) unless requested
            if (fa_p.parent / "simulation_metadata.json").exists() and not include_synthetic:
                continue
            meta = load_dataset_metadata(fa_p)
            ds_id = meta.get("dataset_id") or fa_p.parent.name
            discovered.append(
                {
                    "dataset_id": ds_id,
                    "tier_group": tier_name,
                    "fasta_path": fa_p,
                    "meta": meta,
                }
            )
    return discovered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable Stage A + Stage B Empirical Corpus Runner (N = 45 Datasets)."
    )
    parser.add_argument("--raw-root", default="data_empirical/01_raw_alignments")
    parser.add_argument("--manifest", default=None, help="Optional explicit empirical_manifest.json.")
    parser.add_argument(
        "--out-data-root",
        default="data_empirical",
        help="Root directory for 02_ground_truths, 03_standardized_trees, 04_audit_reports (isolated from synthetic data/).",
    )
    parser.add_argument("--stage", choices=["A", "B", "both"], default="both")
    parser.add_argument("--bootstraps", type=int, default=100)
    parser.add_argument("--include-classical-ml", action="store_true", default=True)
    parser.add_argument("--skip-classical-ml", action="store_true", help="Skip IQ-TREE2/RAxML-NG/FastTree2 in Stage A.")
    parser.add_argument("--include-synthetic", action="store_true", help="Allow synthetic AliSim fixtures (for pre-flight testing only).")
    parser.add_argument("--fusang-cmd", default=None)
    parser.add_argument("--fusang-batch-cmd", default=None)
    parser.add_argument("--phyloformer-cmd", default=None)
    parser.add_argument("--phyloformer-batch-cmd", default=None)
    parser.add_argument("--use-surrogate", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--agent-model", default="qwen/qwen3-14b")
    parser.add_argument("--zeroshot-model", default="qwen/qwen3-14b")
    parser.add_argument("--generic-agent-model", default="qwen/qwen3-14b")
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--require-live-llm", action="store_true")
    parser.add_argument("--offline-mock", action="store_true")
    parser.add_argument("--redo", action="store_true", help="Recompute outputs even if checkpoint files exist.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort immediately on first dataset error instead of recording and continuing.")
    parser.add_argument(
        "--allow-alphabetical-outgroup-fallback",
        action="store_true",
        help="Allow fallback to sorted(taxa)[0] if outgroup_taxon is omitted/missing (disabled by default to prevent silent ingroup rooting).",
    )
    parser.add_argument(
        "--strict-proposal-scope",
        action="store_true",
        help="Enforce strict Section 5.3 scope: fail on Tier 2/3 AA datasets with N <= 40 or Tier 2/3 datasets lacking published T_true.nwk.",
    )
    args = parser.parse_args()

    data_root = Path(args.out_data_root)
    reports_root = data_root / "04_audit_reports"
    stat_dir = reports_root / "statistical_comparisons"
    stat_dir.mkdir(parents=True, exist_ok=True)
    base_url = None if (args.base_url.lower() == "none" or args.offline_mock) else args.base_url

    datasets = discover_empirical_datasets(
        raw_root=Path(args.raw_root),
        manifest_path=Path(args.manifest) if args.manifest else None,
        include_synthetic=args.include_synthetic,
    )
    if not datasets:
        print(
            f"[!] No empirical datasets found under {args.raw_root} "
            "(synthetic AliSim fixtures are excluded by default; pass --include-synthetic for pre-flight tests)."
        )
        return

    corpus_ids = {str(d["dataset_id"]) for d in datasets}
    print(f"[*] Discovered {len(datasets)} dataset(s) for Stage='{args.stage}' execution.")
    stage_a_rows: list[dict] = []
    failed_datasets: list[dict] = []

    for idx, item in enumerate(datasets, start=1):
        ds_id = item["dataset_id"]
        tier_grp = item["tier_group"]
        fa_path = item["fasta_path"]
        meta = item["meta"]

        try:
            taxa_tup, _, seq_len = load_alignment_data(fa_path)
            n_taxa = len(taxa_tup)
            seq_alpha = detect_sequence_alphabet(str(fa_path))
            outgroup = meta.get("outgroup_taxon")
            if not outgroup or outgroup not in taxa_tup:
                if args.allow_alphabetical_outgroup_fallback:
                    outgroup = sorted(taxa_tup)[0]
                    print(
                        f"  [WARN] Dataset '{ds_id}' missing valid outgroup_taxon; falling back to '{outgroup}'."
                    )
                else:
                    raise ValueError(
                        f"Dataset '{ds_id}' is missing a valid 'outgroup_taxon' in dataset_metadata.json "
                        f"(got {outgroup!r}, not found in {n_taxa} FASTA taxa). Refusing silent alphabetical "
                        "fallback to prevent reversed ancestor->derived polarity."
                    )

            # Scope & model-compatibility guards BEFORE spending hours on ML/neural execution
            if seq_alpha == "DNA" and n_taxa > 40 and not args.use_surrogate:
                raise ValueError(
                    f"Dataset '{ds_id}' is DNA with N={n_taxa} > 40 (exceeds Fusang's 40-taxon ceiling and "
                    "incompatible with AA-only Phyloformer). Subsample to <=40 taxa or translate CDS to AA."
                )
            if seq_alpha == "DNA" and seq_len > 14000 and not args.use_surrogate:
                raise ValueError(
                    f"Dataset '{ds_id}' has DNA length L={seq_len} > 14,000 bp (exceeds Fusang's hardcoded "
                    "length limit in fusang.py). Trim/select locus <= 10,000 bp."
                )
            if seq_alpha == "DNA" and (seq_len < 240 or seq_len > 10000):
                print(
                    f"  [WARN] DNA dataset '{ds_id}' has L={seq_len} bp outside Fusang's recommended [240, 10000] bp range."
                )
            if tier_grp in ("tier2", "tier3") and seq_alpha == "AA" and n_taxa <= 40:
                msg_scope = (
                    f"Dataset '{ds_id}' ({tier_grp}) is AA with N={n_taxa} <= 40 "
                    "(Proposal Section 5.3 scopes Phyloformer outside Tier 1 to >40 protein taxa)."
                )
                if args.strict_proposal_scope:
                    raise ValueError(msg_scope)
                print(f"  [WARN] {msg_scope}")

            base_paths = DatasetPaths(dataset_id=ds_id, tier_group=tier_grp, root_dir=data_root)
            gt_dir = base_paths.ground_truth_dir

            print(f"\n[{idx}/{len(datasets)}] Dataset={ds_id} ({tier_grp}, {seq_alpha}, N={n_taxa}, L={seq_len})")

            # ------------------------------------------------------------------
            # STEP 1: GROUND-TRUTH INDEXING (Checkpointed)
            # ------------------------------------------------------------------
            gt_ready = (
                (gt_dir / "T_true_rooted.nwk").exists()
                and (gt_dir / "M_true.tsv").exists()
                and (gt_dir / "A_true.json").exists()
            )
            if not gt_ready or args.redo:
                tree_cand = meta.get("published_tree_path") or (fa_path.parent / "T_true.nwk")
                tree_path = Path(tree_cand) if not isinstance(tree_cand, Path) else tree_cand
                if not tree_path.is_absolute() and (fa_path.parent / tree_path).exists():
                    tree_path = fa_path.parent / tree_path

                if tier_grp in ("tier2", "tier3") and not tree_path.exists():
                    msg_pub = (
                        f"Dataset '{ds_id}' ({tier_grp}) has no published reference tree (T_true.nwk); "
                        "falling back to IQ-TREE2_ModelFinder risks Stage A circularity."
                    )
                    if args.strict_proposal_scope:
                        raise ValueError(msg_pub)
                    print(f"  [WARN] {msg_pub}")

                anc_fa = meta.get("experimental_ancestors_fasta")
                if anc_fa and not Path(anc_fa).is_absolute() and (fa_path.parent / anc_fa).exists():
                    anc_fa = str(fa_path.parent / anc_fa)

                use_consensus = bool(
                    meta.get("require_three_engine_consensus", tier_grp in ("tier2", "tier3"))
                )
                if use_consensus or not tree_path.exists():
                    build_empirical_tier3_ground_truth(
                        fasta_path=str(fa_path),
                        out_dir=str(gt_dir),
                        outgroup=outgroup,
                        ufboot_replicates=0 if args.bootstraps <= 10 else 1000,
                        recombination_breakpoints=meta.get("recombination_breakpoints"),
                        auto_scan_recombination=meta.get("auto_scan_recombination"),
                        published_tree_path=str(tree_path) if tree_path.exists() else None,
                        expected_stressors=meta.get("expected_stressors"),
                        documented_lba_taxa=meta.get("documented_lba_taxa") or meta.get("lba_long_taxa"),
                        documented_clade_anomalies=meta.get("documented_clade_anomalies"),
                        experimental_ancestors_fasta=anc_fa,
                        cataloged_mutations=meta.get("cataloged_mutations"),
                        kept_original_columns_1based=meta.get("kept_original_columns_1based"),
                    )
                else:
                    run_ground_truth_pipeline(
                        fasta_path=str(fa_path),
                        true_tree_path=str(tree_path),
                        out_dir=str(gt_dir),
                        outgroup=outgroup,
                        recombination_breakpoints=meta.get("recombination_breakpoints"),
                        auto_scan_recombination=(
                            True
                            if meta.get("auto_scan_recombination") is None
                            else bool(meta.get("auto_scan_recombination"))
                        ),
                        expected_stressors=meta.get("expected_stressors"),
                        substitution_model=meta.get("substitution_model") or ("MFP" if args.bootstraps > 10 else None),
                        experimental_ancestors_fasta=anc_fa,
                        cataloged_mutations=meta.get("cataloged_mutations"),
                        kept_original_columns_1based=meta.get("kept_original_columns_1based"),
                        documented_lba_taxa=meta.get("documented_lba_taxa") or meta.get("lba_long_taxa"),
                        documented_clade_anomalies=meta.get("documented_clade_anomalies"),
                    )

            a_true_obj = json.loads((gt_dir / "A_true.json").read_text())
            sub_model = a_true_obj.get("substitution_model")

            # Determine applicable neural target model(s)
            neural_targets: list[str] = []
            if seq_alpha == "DNA" and n_taxa <= 40:
                neural_targets.append("Fusang")
            if seq_alpha == "AA" or args.use_surrogate:
                if seq_alpha == "AA":
                    neural_targets.append("Phyloformer")
                elif args.use_surrogate and n_taxa > 40:
                    neural_targets.append("Phyloformer")

            models_to_run = list(neural_targets)
            if args.stage in ("A", "both") and not args.skip_classical_ml:
                models_to_run.extend(["IQ-TREE2", "RAxML-NG", "FastTree2"])

            # ------------------------------------------------------------------
            # STAGE A: TARGET & CLASSICAL ML EXECUTION (Checkpointed)
            # ------------------------------------------------------------------
            if args.stage in ("A", "both"):
                for m_name in models_to_run:
                    m_paths = DatasetPaths(
                        dataset_id=ds_id, target_model=m_name, tier_group=tier_grp, root_dir=data_root
                    )
                    if (not m_paths.normalized_newick.exists()) or args.redo:
                        print(f"  [Stage A] Inferring & standardizing {m_name} (B={args.bootstraps})...")
                        run_target_model_pipeline(
                            fasta_path=str(fa_path),
                            target_model=m_name,
                            output_prefix=str(m_paths.standardized_prefix),
                            outgroup_taxon=outgroup,
                            b_replicates=args.bootstraps,
                            fusang_cmd_template=args.fusang_cmd,
                            fusang_batch_cmd_template=args.fusang_batch_cmd,
                            phyloformer_cmd_template=args.phyloformer_cmd,
                            phyloformer_batch_cmd_template=args.phyloformer_batch_cmd,
                            allow_surrogate=args.use_surrogate,
                            max_workers=args.workers,
                            substitution_model=sub_model,
                        )
                    if m_paths.normalized_newick.exists():
                        stage_a_rows.append(
                            evaluate_stage_a_model_row(
                                dataset_id=ds_id,
                                target_model=m_name,
                                pred_newick_path=str(m_paths.normalized_newick),
                                true_newick_path=str(gt_dir / "T_true_rooted.nwk"),
                                pred_meta_path=str(m_paths.normalized_metadata),
                                stressor_gt_path=str(gt_dir / "A_true.json"),
                                tier_group=tier_grp,
                            )
                        )

            # ------------------------------------------------------------------
            # STAGE B: 3-WAY AGENTIC AUDITING (Checkpointed + Master TSV Synced)
            # ------------------------------------------------------------------
            if args.stage in ("B", "both"):
                is_mock_mode = base_url is None and args.offline_mock
                agent_tag = _sanitize_model_tag("offline_mock" if is_mock_mode else args.agent_model)
                zs_tag = _sanitize_model_tag("offline_mock" if is_mock_mode else args.zeroshot_model)
                ga_tag = _sanitize_model_tag("offline_mock" if is_mock_mode else args.generic_agent_model)

                for n_mod in neural_targets:
                    m_paths = DatasetPaths(
                        dataset_id=ds_id, target_model=n_mod, tier_group=tier_grp, root_dir=data_root
                    )
                    if not m_paths.normalized_newick.exists():
                        continue
                    prop_ckpt = (
                        reports_root / "proposed_agent" / f"{ds_id}_{n_mod}_proposed_{agent_tag}_report.json"
                    )
                    b1_ckpt = (
                        reports_root / "baseline1_zeroshot_llm" / f"{ds_id}_{n_mod}_zeroshot_{zs_tag}_report.json"
                    )
                    b2_ckpt = (
                        reports_root / "baseline2_generic_agent" / f"{ds_id}_{n_mod}_generic_agent_{ga_tag}_report.json"
                    )
                    if prop_ckpt.exists() and b1_ckpt.exists() and b2_ckpt.exists() and not args.redo:
                        print(f"  [Stage B] Resuming {ds_id} ({n_mod}) — syncing existing checkpoint reports to master tables.")
                        sync_checkpoint_reports_to_master_tables(
                            prop_json=prop_ckpt,
                            b1_json=b1_ckpt,
                            b2_json=b2_ckpt,
                            fasta_path=str(fa_path),
                            pred_newick_path=str(m_paths.normalized_newick),
                            stressor_gt_path=str(gt_dir / "A_true.json"),
                            out_reports_root=str(reports_root),
                            agent_model=args.agent_model,
                            zeroshot_model=args.zeroshot_model,
                            generic_agent_model=args.generic_agent_model,
                            allowed_dataset_ids=corpus_ids,
                        )
                        continue

                    print(f"  [Stage B] Running Proposed Agent + Baselines 1 & 2 on {ds_id} ({n_mod})...")
                    run_three_way_comparison(
                        dataset_id=ds_id,
                        target_model=n_mod,
                        fasta_path=str(fa_path),
                        pred_newick_path=str(m_paths.normalized_newick),
                        pred_meta_path=str(m_paths.normalized_metadata),
                        true_newick_path=str(gt_dir / "T_true_rooted.nwk"),
                        asr_state_path=str(gt_dir / "M_true.tsv"),
                        stressor_gt_path=str(gt_dir / "A_true.json"),
                        out_reports_root=str(reports_root),
                        agent_model=args.agent_model,
                        zeroshot_model=args.zeroshot_model,
                        generic_agent_model=args.generic_agent_model,
                        lm_studio_base_url=base_url,
                        require_live_llm=args.require_live_llm,
                        allow_offline_mock=args.offline_mock,
                        allowed_dataset_ids=corpus_ids,
                    )
        except Exception as exc:
            if args.fail_fast:
                raise
            print(f"  [!] ERROR on dataset {ds_id}: {exc}", file=sys.stderr)
            failed_datasets.append({"dataset_id": ds_id, "error": str(exc)})

    if stage_a_rows:
        stage_a_df = pd.DataFrame(stage_a_rows)
        stage_a_tsv = stat_dir / "stage_a_model_metrics.tsv"
        stage_a_df.to_csv(stage_a_tsv, sep="\t", index=False, float_format="%.4f")
        stage_a_stats = run_stage_a_statistical_suite(stage_a_rows)
        stage_a_json = stat_dir / "stage_a_statistical_suite.json"
        with open(stage_a_json, "w") as f:
            json.dump(stage_a_stats, f, indent=2)
        print(f"\n[+] Saved Stage A metrics ({len(stage_a_rows)} runs) -> {stage_a_tsv}")
        print(f"[+] Saved Stage A Wilcoxon + BH-FDR suite -> {stage_a_json}")

    if failed_datasets:
        fail_path = stat_dir / "failed_corpus_datasets.json"
        with open(fail_path, "w") as f:
            json.dump(failed_datasets, f, indent=2)
        print(f"[!] {len(failed_datasets)} dataset(s) encountered errors; details in {fail_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
