"""
Target Model & Classical Baseline Inference + Pipeline Step 2 Standardization Runner
(Manuscript Section 4.1.1 Item 4, Table 4.2 & Section 4.3.1).

Executes:
  1. Non-Parametric Bootstrap Resampler (B = 100 replicates via generate_bootstrap_replicates)
  2. Target Model Inference across Primary MSA + B Bootstrap Replicates:
       - Fusang (N <= 40 taxa discrete topology predictor; supports isolated Conda batch CLI
                 adapter, single-MSA CLI hook, or explicit --use-surrogate mode for testing)
       - Phyloformer (Pairwise NxN distance matrix predictor; supports isolated Conda batch GPU
                      adapter, single-MSA CLI hook, or explicit --use-surrogate mode with both
                      4-state DNA JC69 and 20-state Amino Acid Poisson distance transforms)
       - Classical ML Baselines: IQ-TREE2, RAxML-NG, FastTree2 (parallelized across CPU workers)
  3. Tree & Support Normalizer (Rules 1, 2 & 3 via standardize_and_root_tree) ->
     <output_prefix>_normalized.nwk and <output_prefix>_metadata.json in data/03_standardized_trees/.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ete3 import Tree

from src.preprocessing.ground_truth_gen import (
    detect_sequence_alphabet,
    parse_iqtree_best_model,
)
from src.preprocessing.normalizer_wrapper import (
    distance_matrix_to_bionj_newick,
    generate_bootstrap_replicates,
    standardize_and_root_tree,
    to_skbio_distance_matrix,
)
from src.utils.io_cache import (
    default_cpu_threads,
    load_alignment_data,
    resolve_conda_binary,
)


def _run_external_command(cmd_str: str) -> None:
    """Safely executes an external CLI command string without shell=True injection hazards."""
    args_list = shlex.split(cmd_str)
    subprocess.run(args_list, check=True)


def compute_fusang_5state_distance_matrix(
    fasta_path: str, seed: int = 42
) -> tuple[np.ndarray, list[str]]:
    """
    Vectorized surrogate modeling Fusang's gapped-MSA discrete topology sensitivity (only used
    when allow_surrogate=True is explicitly enabled for pipeline/test validation).
    Supports both 4-state DNA and 20-state Amino Acid alignments alongside gap token '-'.
    """
    rng = np.random.default_rng(seed)
    taxa_ids_tup, seq_mat, seq_len = load_alignment_data(fasta_path)
    taxa_ids = list(taxa_ids_tup)
    n_taxa = len(taxa_ids)
    seq_type = detect_sequence_alphabet(fasta_path)

    is_gap = seq_mat == "-"
    is_ungapped = (~is_gap) & (seq_mat != "N") & (seq_mat != "?") & (seq_mat != "X")

    # Vectorized pairwise ungapped mismatch & gap-state mismatch matrices
    shared_ungapped = is_ungapped[:, None, :] & is_ungapped[None, :, :]
    n_ungapped = np.sum(shared_ungapped, axis=2)
    diff_ungapped = (seq_mat[:, None, :] != seq_mat[None, :, :]) & shared_ungapped
    sub_mismatch = np.where(
        n_ungapped > 0,
        np.sum(diff_ungapped, axis=2) / np.maximum(n_ungapped, 1),
        0.5,
    )

    gap_mismatch = np.sum(is_gap[:, None, :] != is_gap[None, :, :], axis=2) / max(seq_len, 1)

    lba_thresh = 0.35 if seq_type == "AA" else 0.18
    lba_discount = np.where(sub_mismatch > lba_thresh, 0.65, 1.0)
    scaling_noise_sd = 0.002 if n_taxa < 30 else (0.015 if n_taxa >= 38 else 0.006)

    noise = rng.normal(0.0, scaling_noise_sd, size=(n_taxa, n_taxa))
    noise = np.triu(noise, 1)
    noise = noise + noise.T

    D = lba_discount * (0.75 * sub_mismatch + 0.55 * gap_mismatch) + noise
    D = np.maximum(D, 1e-5)
    np.fill_diagonal(D, 0.0)
    return D, taxa_ids


def infer_fusang_topology_single(
    fasta_path: str,
    fusang_cmd_template: str | None = None,
    allow_surrogate: bool = False,
    seed: int = 42,
) -> str:
    """
    Infers a single unrooted topology Newick string (without biological branch lengths) for Fusang.
    """
    cmd_tmpl = fusang_cmd_template or os.environ.get("FUSANG_CMD")
    if cmd_tmpl:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_nwk = Path(tmpdir) / "fusang_out.nwk"
            cmd_str = cmd_tmpl.format(input_fasta=fasta_path, output_newick=str(out_nwk))
            _run_external_command(cmd_str)
            return out_nwk.read_text().strip()

    if not allow_surrogate:
        raise RuntimeError(
            "Real Fusang execution requires --fusang-cmd, --fusang-batch-cmd, or FUSANG_CMD / FUSANG_BATCH_CMD "
            "(see envs/env_fusang.yml & scripts/adapters/run_fusang_cli.py). "
            "Pass --use-surrogate (or allow_surrogate=True in tests) only when running surrogate validation."
        )

    D, taxa_ids = compute_fusang_5state_distance_matrix(fasta_path, seed=seed)
    bionj_nwk = distance_matrix_to_bionj_newick(D, taxa_labels=taxa_ids)
    t = Tree(bionj_nwk, format=1)
    for node in t.traverse():
        node.dist = 1.0
    return t.write(format=9)


def predict_phyloformer_distance_matrix(
    fasta_path: str,
    phyloformer_cmd_template: str | None = None,
    allow_surrogate: bool = False,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """
    Predicts an NxN pairwise evolutionary distance matrix D for Phyloformer.
    Uses external CLI command if provided, otherwise requires allow_surrogate=True and applies
    the alphabet-appropriate uniform-rate Markov distance transform (4-state JC69 for DNA;
    20-state Poisson/JC for Amino Acids).
    """
    taxa_ids_tup, seq_mat, _ = load_alignment_data(fasta_path)
    taxa_ids = list(taxa_ids_tup)
    n_taxa = len(taxa_ids)

    cmd_tmpl = phyloformer_cmd_template or os.environ.get("PHYLOFORMER_CMD")
    if cmd_tmpl:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_mat = Path(tmpdir) / "pf_dist.npy"
            cmd_str = cmd_tmpl.format(input_fasta=fasta_path, output_matrix=str(out_mat))
            _run_external_command(cmd_str)
            D = np.load(str(out_mat))
            return D, taxa_ids

    if not allow_surrogate:
        raise RuntimeError(
            "Real Phyloformer execution requires --phyloformer-cmd, --phyloformer-batch-cmd, or "
            "PHYLOFORMER_CMD / PHYLOFORMER_BATCH_CMD (see envs/env_phyloformer.yml & "
            "scripts/adapters/run_phyloformer_cli.py). Pass --use-surrogate (or allow_surrogate=True) "
            "only when running surrogate validation."
        )

    rng = np.random.default_rng(seed)
    seq_type = detect_sequence_alphabet(fasta_path)
    is_valid = (seq_mat != "-") & (seq_mat != "N") & (seq_mat != "?") & (seq_mat != "X")

    shared = is_valid[:, None, :] & is_valid[None, :, :]
    n_shared = np.sum(shared, axis=2)
    diffs = (seq_mat[:, None, :] != seq_mat[None, :, :]) & shared
    p_diff = np.where(
        n_shared > 0,
        np.sum(diffs, axis=2) / np.maximum(n_shared, 1),
        0.5,
    )

    if seq_type == "AA":
        # 20-state Amino Acid uniform-rate Poisson/Jukes-Cantor formula: -19/20 * ln(1 - 20/19 * p)
        p_clamped = np.minimum(p_diff, 0.92)
        d_raw = -0.95 * np.log(np.maximum(1.0 - (20.0 / 19.0) * p_clamped, 1e-6))
    else:
        # 4-state Nucleotide JC69 formula: -3/4 * ln(1 - 4/3 * p)
        p_clamped = np.minimum(p_diff, 0.72)
        d_raw = -0.75 * np.log(np.maximum(1.0 - (4.0 / 3.0) * p_clamped, 1e-6))

    noise = rng.normal(0.0, 0.0015, size=(n_taxa, n_taxa))
    noise = np.triu(noise, 1)
    noise = noise + noise.T
    D = np.maximum(d_raw + noise, 1e-5)
    np.fill_diagonal(D, 0.0)
    return D, taxa_ids


def save_distance_matrix_tsv(
    D: np.ndarray, taxa_ids: list[str], out_tsv_path: str
) -> None:
    Path(out_tsv_path).parent.mkdir(parents=True, exist_ok=True)
    dm = to_skbio_distance_matrix(D, taxa_ids)
    with open(out_tsv_path, "w") as f:
        f.write("taxon\t" + "\t".join(dm.ids) + "\n")
        for idx, tid in enumerate(dm.ids):
            row_vals = "\t".join(f"{val:.6f}" for val in dm.data[idx])
            f.write(f"{tid}\t{row_vals}\n")


def _select_iqtree_best_model_once(fasta_path: str, is_dna: bool, seed: int) -> str:
    """Runs IQ-TREE 2 ModelFinder (-m MFP -T 1) on the primary alignment to select best-fit model."""
    iq_bin = resolve_conda_binary("iqtree2")
    fallback = "GTR+G4" if is_dna else "LG+G4"
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "mfp"
        cmd = [
            iq_bin,
            "-s",
            str(fasta_path),
            "-m",
            "MFP",
            "-n",
            "0",
            "-nt",
            "1",
            "--prefix",
            str(prefix),
            "-seed",
            str(seed),
            "-quiet",
            "-redo",
        ]
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            chosen = parse_iqtree_best_model(Path(tmpdir) / "mfp.iqtree")
            if chosen:
                return chosen
        except Exception:
            pass
    return fallback


def _infer_single_classical_worker(args: tuple[str, str, bool, int, str | None, bool]) -> str:
    """Top-level picklable worker executing a single classical ML tree inference."""
    fa_file, model_name, is_dna, run_seed, sub_model_override, use_fast_iq = args
    if model_name == "FastTree2":
        ft_bin = resolve_conda_binary("fasttree")
        cmd = [ft_bin, "-quiet"]
        if is_dna:
            cmd.extend(["-nt", "-gtr"])
        else:
            cmd.extend(["-lg"])
        cmd.append(str(fa_file))
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return res.stdout.strip()

    if model_name == "IQ-TREE2":
        iq_bin = resolve_conda_binary("iqtree2")
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "iq"
            m_str = sub_model_override or ("GTR+G4" if is_dna else "LG+G4")
            cmd = [
                iq_bin,
                "-s",
                str(fa_file),
                "-m",
                m_str,
                "-nt",
                "1",
                "--prefix",
                str(prefix),
                "-seed",
                str(run_seed),
                "-quiet",
                "-redo",
            ]
            if use_fast_iq:
                cmd.append("-fast")
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return (Path(tmpdir) / "iq.treefile").read_text().strip()

    if model_name == "RAxML-NG":
        rax_bin = resolve_conda_binary("raxml-ng")
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "rax"
            m_str = "GTR+G" if is_dna else "LG+G"
            cmd = [
                rax_bin,
                "--msa",
                str(fa_file),
                "--model",
                m_str,
                "--prefix",
                str(prefix),
                "--seed",
                str(run_seed),
                "--threads",
                "1",
                "--redo",
            ]
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return (Path(tmpdir) / "rax.raxml.bestTree").read_text().strip()

    raise ValueError(f"Unsupported classical model_name: {model_name}")


def run_classical_engine_with_bootstraps(
    fasta_path: str,
    model_name: str,
    rep_paths: list[str],
    seq_type: str = "AUTO",
    seed: int = 42,
    max_workers: int | None = None,
    substitution_model: str | None = None,
) -> tuple[str, list[str]]:
    """
    Runs IQ-TREE2, RAxML-NG, or FastTree2 across the primary MSA and B bootstrap replicate MSAs
    concurrently across CPU workers so all models undergo identical DendroPy split-frequency standardization.
    """
    resolved_type = (
        detect_sequence_alphabet(fasta_path)
        if seq_type.upper() == "AUTO"
        else seq_type.upper()
    )
    is_dna = resolved_type == "DNA"
    workers = max_workers or default_cpu_threads(8)

    chosen_sub_model = substitution_model
    use_fast_iq = len(rep_paths) <= 10
    if model_name == "IQ-TREE2" and not chosen_sub_model and len(rep_paths) > 10:
        chosen_sub_model = _select_iqtree_best_model_once(fasta_path, is_dna, seed)

    all_tasks = [(fasta_path, model_name, is_dna, seed, chosen_sub_model, use_fast_iq)] + [
        (rp, model_name, is_dna, seed + idx + 1, chosen_sub_model, use_fast_iq)
        for idx, rp in enumerate(rep_paths)
    ]
    if workers <= 1 or len(all_tasks) <= 2:
        results = [_infer_single_classical_worker(t) for t in all_tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(all_tasks))) as pool:
            results = list(pool.map(_infer_single_classical_worker, all_tasks))
    return results[0], results[1:]


def _stage_batch_directory(fasta_path: str, rep_paths: list[str], msa_dir: Path) -> None:
    msa_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fasta_path, str(msa_dir / "primary.fasta"))
    for idx, rp in enumerate(rep_paths):
        shutil.copyfile(rp, str(msa_dir / f"rep_{idx:03d}.fasta"))


def run_target_model_pipeline(
    fasta_path: str,
    target_model: str,
    output_prefix: str,
    outgroup_taxon: str | None = None,
    b_replicates: int = 100,
    seq_type: str = "AUTO",
    fusang_cmd_template: str | None = None,
    fusang_batch_cmd_template: str | None = None,
    phyloformer_cmd_template: str | None = None,
    phyloformer_batch_cmd_template: str | None = None,
    enforce_fusang_taxa_limit: bool = True,
    allow_surrogate: bool = False,
    max_workers: int | None = None,
    substitution_model: str | None = None,
    seed: int = 42,
) -> dict | None:
    """
    Executes full Pipeline Step 2 for a single (dataset, target_model) pair:
      1. Checks N <= 40 constraint for Fusang
      2. Generates B non-parametric bootstrap replicates (generate_bootstrap_replicates)
      3. Runs target_model inference on primary MSA + B replicates (supporting batch GPU adapters)
      4. Applies Rules 1, 2 & 3 standardization (standardize_and_root_tree) ->
         <output_prefix>_normalized.nwk and <output_prefix>_metadata.json.
    """
    taxa_ids_tup, _, _ = load_alignment_data(fasta_path)
    taxa_ids_ordered = list(taxa_ids_tup)
    n_taxa = len(taxa_ids_ordered)
    taxa_names_sorted = sorted(taxa_ids_ordered)
    chosen_outgroup = (
        outgroup_taxon
        if (outgroup_taxon and outgroup_taxon in taxa_names_sorted)
        else taxa_names_sorted[0]
    )

    if target_model == "Fusang" and enforce_fusang_taxa_limit and n_taxa > 40:
        return None

    inference_backend = "external_cli"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        rep_paths = generate_bootstrap_replicates(
            fasta_path=fasta_path,
            out_dir=str(tmp_root),
            b_replicates=b_replicates,
            seed=seed,
        )

        if target_model == "Fusang":
            batch_tmpl = fusang_batch_cmd_template or os.environ.get("FUSANG_BATCH_CMD")
            single_tmpl = fusang_cmd_template or os.environ.get("FUSANG_CMD")
            if batch_tmpl:
                msa_dir = tmp_root / "batch_msas"
                out_dir = tmp_root / "batch_out"
                _stage_batch_directory(fasta_path, rep_paths, msa_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                _run_external_command(
                    batch_tmpl.format(msa_dir=str(msa_dir), out_dir=str(out_dir))
                )
                primary_nwk = (out_dir / "primary.nwk").read_text().strip()
                boot_nwks = [
                    (out_dir / f"rep_{idx:03d}.nwk").read_text().strip()
                    for idx in range(len(rep_paths))
                ]
            else:
                if not single_tmpl:
                    inference_backend = "surrogate"
                primary_nwk = infer_fusang_topology_single(
                    fasta_path=fasta_path,
                    fusang_cmd_template=single_tmpl,
                    allow_surrogate=allow_surrogate,
                    seed=seed,
                )
                boot_nwks = [
                    infer_fusang_topology_single(
                        fasta_path=rp,
                        fusang_cmd_template=single_tmpl,
                        allow_surrogate=allow_surrogate,
                        seed=seed + idx + 1,
                    )
                    for idx, rp in enumerate(rep_paths)
                ]

        elif target_model == "Phyloformer":
            batch_tmpl = phyloformer_batch_cmd_template or os.environ.get("PHYLOFORMER_BATCH_CMD")
            single_tmpl = phyloformer_cmd_template or os.environ.get("PHYLOFORMER_CMD")
            if batch_tmpl:
                msa_dir = tmp_root / "batch_msas"
                out_dir = tmp_root / "batch_out"
                _stage_batch_directory(fasta_path, rep_paths, msa_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                _run_external_command(
                    batch_tmpl.format(msa_dir=str(msa_dir), out_dir=str(out_dir))
                )
                D_primary = np.load(str(out_dir / "primary.npy"))
                save_distance_matrix_tsv(
                    D_primary, taxa_ids_ordered, f"{output_prefix}_distance_matrix.tsv"
                )
                primary_nwk = distance_matrix_to_bionj_newick(
                    D_primary, taxa_labels=taxa_ids_ordered
                )
                boot_nwks = []
                for idx in range(len(rep_paths)):
                    D_rep = np.load(str(out_dir / f"rep_{idx:03d}.npy"))
                    boot_nwks.append(
                        distance_matrix_to_bionj_newick(D_rep, taxa_labels=taxa_ids_ordered)
                    )
            else:
                if not single_tmpl:
                    inference_backend = "surrogate"
                D_primary, taxa_ids = predict_phyloformer_distance_matrix(
                    fasta_path=fasta_path,
                    phyloformer_cmd_template=single_tmpl,
                    allow_surrogate=allow_surrogate,
                    seed=seed,
                )
                save_distance_matrix_tsv(
                    D_primary, taxa_ids, f"{output_prefix}_distance_matrix.tsv"
                )
                primary_nwk = distance_matrix_to_bionj_newick(
                    D_primary, taxa_labels=taxa_ids
                )
                boot_nwks = []
                for idx, rp in enumerate(rep_paths):
                    D_rep, rep_ids = predict_phyloformer_distance_matrix(
                        fasta_path=rp,
                        phyloformer_cmd_template=single_tmpl,
                        allow_surrogate=allow_surrogate,
                        seed=seed + idx + 1,
                    )
                    boot_nwks.append(
                        distance_matrix_to_bionj_newick(D_rep, taxa_labels=rep_ids)
                    )

        elif target_model in ("IQ-TREE2", "RAxML-NG", "FastTree2"):
            primary_nwk, boot_nwks = run_classical_engine_with_bootstraps(
                fasta_path=fasta_path,
                model_name=target_model,
                rep_paths=rep_paths,
                seq_type=seq_type,
                seed=seed,
                max_workers=max_workers,
                substitution_model=substitution_model,
            )
        else:
            raise ValueError(
                f"Unknown target_model '{target_model}'. Expected one of: "
                "Fusang, Phyloformer, IQ-TREE2, RAxML-NG, FastTree2."
            )

    metadata = standardize_and_root_tree(
        primary_newick_str=primary_nwk,
        bootstrap_newick_strs=boot_nwks,
        outgroup_taxon=chosen_outgroup,
        target_model=target_model,
        output_prefix=output_prefix,
    )
    metadata["n_taxa"] = n_taxa
    metadata["b_replicates"] = b_replicates
    metadata["inference_backend"] = inference_backend
    with open(f"{output_prefix}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata
