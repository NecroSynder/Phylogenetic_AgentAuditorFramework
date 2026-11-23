#!/usr/bin/env python3
"""
Batch & Single-MSA GPU CLI Adapter for Phyloformer (lucanest/Phyloformer).
Designed to run inside `env_phyloformer` so the Phyloformer checkpoint (`models/pf.ckpt`
or `models/pf_indel.ckpt`) is loaded ONCE into GPU VRAM and predicts NxN pairwise distance
matrices (.npy) across primary.fasta + all B=100 bootstrap replicates in a single pass.

Matches `external_models/Phyloformer/infer_alns.py` and `phyloformer/data.py` directly,
and verifies CUDA kernel compatibility (e.g., RTX 50-series Blackwell sm_120 via cu128).
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO


PF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX-"
PF_LOOKUP = {ch: idx for idx, ch in enumerate(PF_ALPHABET)}


def _select_execution_device(preference: str = "auto") -> str:
    """Selects 'cuda' if a live kernel executes cleanly on the GPU, else falls back safely to 'cpu'."""
    if preference == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        try:
            probe = torch.ones((2, 2), device="cuda", dtype=torch.float32)
            _ = (probe @ probe).cpu()
            return "cuda"
        except Exception as exc:
            if preference == "cuda":
                raise
            print(
                f"[Phyloformer Adapter] CUDA kernel probe failed ({exc}); falling back to CPU. "
                "For RTX 50-series (sm_120) GPU acceleration, install cu128 PyTorch in env_phyloformer.",
                file=sys.stderr,
            )
    elif preference == "cuda":
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False.")
    return "cpu"


def _encode_protein_msa_for_phyloformer(fasta_path: Path) -> tuple[torch.Tensor, list[str]]:
    """
    Encodes a Protein FASTA into Phyloformer's exact (22, seq_len, n_seqs) one-hot tensor
    (phyloformer/data.py: ALPHABET = b'ARNDCQEGHILKMFPSTWYVX-'), safely mapping lowercase
    residues to uppercase and any rare ambiguity symbols (B, Z, J, U, O, *, ?) to 'X' / '-'.
    """
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        raise ValueError(f"Empty alignment: {fasta_path}")

    ids: list[str] = []
    encoded_rows: list[list[int]] = []
    x_idx = PF_LOOKUP["X"]
    non_gap_count = 0
    dna_like_count = 0
    dna_core = {"A", "C", "G", "T", "U"}

    for rec in records:
        ids.append(rec.id)
        seq_str = (
            str(rec.seq)
            .upper()
            .replace(".", "-")
            .replace("?", "-")
            .replace("*", "-")
        )
        for ch in seq_str:
            if ch not in {"-", "N", "X"}:
                non_gap_count += 1
                if ch in dna_core:
                    dna_like_count += 1
        encoded_rows.append([PF_LOOKUP.get(ch, x_idx) for ch in seq_str])

    if non_gap_count >= 20 and (dna_like_count / non_gap_count) >= 0.95:
        raise ValueError(
            f"Upstream Phyloformer checkpoints only support Protein (AA) alignments, "
            f"but {fasta_path.name} appears to be Nucleotide (DNA: {dna_like_count}/{non_gap_count} A/C/G/T/U). "
            "Run Phyloformer on AA datasets and Fusang on DNA datasets."
        )

    seqs_tensor = torch.tensor(encoded_rows, dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(
        seqs_tensor, num_classes=len(PF_ALPHABET)
    ).permute(2, 1, 0)
    return one_hot, ids


def _load_phyloformer_model(repo_dir: Path, checkpoint_path: Path, device: str):
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    import phyloformer.model as pf_model_mod
    from phyloformer.model import Phyloformer

    load_kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False

    ckpt = torch.load(str(checkpoint_path), **load_kwargs)
    params = dict(ckpt["hyper_parameters"])
    params["device"] = device
    model = Phyloformer(**params)
    state_dict = {
        k.replace("model.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k != "model.seq2pair"
    }
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    # Upstream Phyloformer stores `self.seq2pair` as a plain attribute (not register_buffer),
    # so `model.to(device)` leaves `self.seq2pair` on CPU unless moved explicitly.
    model.n_seqs = int(model.seq2pair.shape[1])
    model.seq2pair = model.seq2pair.to(device)
    model.eval()
    return model, pf_model_mod


def _predict_phyloformer_pairs(
    model, pf_model_mod, aln_tensor: torch.Tensor, device: str
) -> tuple[torch.Tensor, str]:
    n_taxa = int(aln_tensor.shape[-1])
    if n_taxa > int(pf_model_mod.SEQ2PAIR.shape[1]):
        pf_model_mod.SEQ2PAIR = pf_model_mod.seq2pair(n_taxa)
        model.n_seqs = -1  # force _set_seq2pair rebuild against expanded SEQ2PAIR

    if device == "cuda":
        try:
            if str(model.seq2pair.device) != str(torch.device("cuda")) and model.seq2pair.device.type != "cuda":
                model.seq2pair = model.seq2pair.to("cuda")
            out = model(aln_tensor[None, :].to("cuda").float())
            if n_taxa >= 100:
                torch.cuda.empty_cache()
            return out, "cuda"
        except torch.cuda.OutOfMemoryError:
            # Reset n_seqs=-1 in case OOM occurred inside _set_seq2pair after self.n_seqs was mutated
            model.n_seqs = -1
            torch.cuda.empty_cache()
            try:
                if str(model.seq2pair.device) != str(torch.device("cuda")) and model.seq2pair.device.type != "cuda":
                    model.seq2pair = model.seq2pair.to("cuda")
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out_fp16 = model(aln_tensor[None, :].to("cuda").float())
                torch.cuda.empty_cache()
                return out_fp16.float(), "cuda"
            except torch.cuda.OutOfMemoryError:
                print(
                    f"[Phyloformer Adapter] CUDA OOM at N={n_taxa} taxa; switching batch execution to CPU.",
                    file=sys.stderr,
                )
                model.n_seqs = -1
                torch.cuda.empty_cache()
                model.to("cpu")
                model.seq2pair = model.seq2pair.to("cpu")
                out_cpu = model(aln_tensor[None, :].to("cpu").float())
                return out_cpu, "cpu"

    if str(model.seq2pair.device) != str(torch.device(device)) and model.seq2pair.device.type != device:
        model.seq2pair = model.seq2pair.to(device)
    return model(aln_tensor[None, :].to(device).float()), device


def _preds_to_symmetric_numpy(preds: torch.Tensor, n_taxa: int) -> np.ndarray:
    dm = torch.zeros((n_taxa, n_taxa), dtype=torch.float64, device=preds.device)
    tri_idx = torch.triu_indices(row=n_taxa, col=n_taxa, offset=1, device=preds.device)
    dm[tri_idx[0], tri_idx[1]] = preds.reshape(-1).to(torch.float64)
    sym = (dm + dm.T).detach().cpu().numpy()
    np.fill_diagonal(sym, 0.0)
    return sym


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch GPU CLI Adapter for lucanest/Phyloformer.")
    parser.add_argument(
        "--phyloformer-repo",
        default=os.environ.get("PHYLOFORMER_REPO", "external_models/Phyloformer"),
        help="Path to cloned lucanest/Phyloformer repository.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("PHYLOFORMER_CKPT", None),
        help="Path to Phyloformer checkpoint (.ckpt). Defaults to <phyloformer-repo>/models/pf.ckpt.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device ('auto' probes GPU kernel execution and falls back to CPU if needed).",
    )
    parser.add_argument("--input-fasta", help="Single input FASTA path.")
    parser.add_argument("--output-matrix", help="Single output .npy matrix path.")
    parser.add_argument("--msa-dir", help="Batch directory containing primary.fasta and rep_*.fasta.")
    parser.add_argument("--out-dir", help="Batch output directory for <stem>.npy distance matrices.")
    args = parser.parse_args()

    repo_dir = Path(args.phyloformer_repo).resolve()
    if not (repo_dir / "phyloformer" / "model.py").exists():
        raise FileNotFoundError(
            f"Could not find phyloformer/model.py under {repo_dir}. Clone https://github.com/lucanest/Phyloformer.git first."
        )

    ckpt_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else (repo_dir / "models" / "pf.ckpt")
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Phyloformer checkpoint not found at {ckpt_path}")

    device = _select_execution_device(args.device)
    model, pf_model_mod = _load_phyloformer_model(repo_dir, ckpt_path, device)

    with torch.no_grad():
        if args.msa_dir and args.out_dir:
            in_dir = Path(args.msa_dir).resolve()
            out_dir = Path(args.out_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            fasta_files = sorted(list(in_dir.glob("*.fasta")) + list(in_dir.glob("*.fa")))
            total = len(fasta_files)
            t0 = time.perf_counter()
            print(
                f"           [Phyloformer Adapter] Loaded {ckpt_path.name} on {device} | Inferring {total} alignments...",
                flush=True,
            )
            for idx, fa in enumerate(fasta_files, start=1):
                aln_tensor, ids = _encode_protein_msa_for_phyloformer(fa)
                preds, device = _predict_phyloformer_pairs(model, pf_model_mod, aln_tensor, device)
                D = _preds_to_symmetric_numpy(preds, len(ids))
                np.save(str(out_dir / f"{fa.stem}.npy"), D)
                if idx == 1 or idx == total or idx % 10 == 0:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"           [Phyloformer Adapter] Progress: {idx}/{total} alignments ({elapsed:.2f}s elapsed)",
                        flush=True,
                    )
        elif args.input_fasta and args.output_matrix:
            src_fa = Path(args.input_fasta).resolve()
            out_mat = Path(args.output_matrix).resolve()
            out_mat.parent.mkdir(parents=True, exist_ok=True)
            aln_tensor, ids = _encode_protein_msa_for_phyloformer(src_fa)
            preds, device = _predict_phyloformer_pairs(model, pf_model_mod, aln_tensor, device)
            D = _preds_to_symmetric_numpy(preds, len(ids))
            np.save(str(out_mat), D)
        else:
            raise ValueError(
                "Provide either (--msa-dir and --out-dir) or (--input-fasta and --output-matrix)."
            )


if __name__ == "__main__":
    main()
