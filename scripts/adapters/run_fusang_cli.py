#!/usr/bin/env python3
"""
In-Process Batch & Single-MSA CLI Adapter for Fusang (Jerry-0591/Fusang).
Designed to execute inside the isolated `env_fusang` Conda environment.

Key optimizations & safeguards over spawning `fusang.py` per replicate:
  1. Loads TensorFlow + Fusang's Conv2D-BiLSTM Keras weights ONCE across all B+1
     alignments (primary.fasta + rep_000..rep_099.fasta) instead of 101 separate processes.
  2. Suppresses benign TensorFlow 2.13 C++ startup warnings on WSL2 / Blackwell GPUs
     (`TF-TRT`, `NUMA node`, `Skipping registering GPU devices`) and runs fork-safe CPU
     quartet inference (`fusang.py` uses `multiprocessing.Pool` via `fork()`).
  3. Vectorizes quartet tensor slicing while preserving upstream Fusang's exact encoding,
     sliding-window averaging, and quartet beam-search tree reconstruction (`fusang.py`).
  4. Validates nucleotide (DNA) alphabet upfront so AA alignments are never mis-encoded.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path

# Must be set BEFORE importing tensorflow / fusang to silence C++ stream_executor logs
# and guarantee fork-safety when fusang.py spawns multiprocessing.Pool workers.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
from Bio import AlignIO, SeqIO


DNA_ALLOWED = set("ATCGUNRYKMSWBDHVX-?.")

WEIGHT_SUBDIR_240 = {
    ("standard", "gamma"): "./dl_model/len_240/S1G/best_weights_clas",
    ("standard", "uniform"): "./dl_model/len_240/S1U/best_weights_clas",
    ("coding", "gamma"): "./dl_model/len_240/C1G/best_weights_clas",
    ("coding", "uniform"): "./dl_model/len_240/C1U/best_weights_clas",
    ("noncoding", "gamma"): "./dl_model/len_240/N1G/best_weights_clas",
    ("noncoding", "uniform"): "./dl_model/len_240/N1U/best_weights_clas",
}

WEIGHT_SUBDIR_1200 = {
    ("standard", "gamma"): "./dl_model/len_1200/S2G/best_weights_clas",
    ("standard", "uniform"): "./dl_model/len_1200/S2U/best_weights_clas",
    ("coding", "gamma"): "./dl_model/len_1200/C2G/best_weights_clas",
    ("coding", "uniform"): "./dl_model/len_1200/C2U/best_weights_clas",
    ("noncoding", "gamma"): "./dl_model/len_1200/N2G/best_weights_clas",
    ("noncoding", "uniform"): "./dl_model/len_1200/N2U/best_weights_clas",
}


def _prepare_fusang_compatible_fasta(src_fasta: Path, dst_fasta: Path) -> None:
    """Writes a clean uppercase `.fasta` file and validates that sequences are nucleotide (DNA)."""
    records = list(SeqIO.parse(str(src_fasta), "fasta"))
    if not records:
        raise ValueError(f"Empty FASTA alignment: {src_fasta}")

    sample_chars = set()
    for rec in records[: min(10, len(records))]:
        sample_chars.update(str(rec.seq).upper())
    non_dna = sample_chars - DNA_ALLOWED
    if non_dna:
        raise ValueError(
            f"Upstream Fusang only supports Nucleotide (DNA) alignments, but found non-DNA symbols "
            f"{sorted(non_dna)} in {src_fasta.name}. Run Fusang on DNA datasets and Phyloformer on AA datasets."
        )

    with open(dst_fasta, "w") as out_f:
        for rec in records:
            seq_clean = (
                str(rec.seq)
                .upper()
                .replace(".", "-")
                .replace("?", "-")
            )
            out_f.write(f">{rec.id}\n{seq_clean}\n")


def _configure_fusang_taxa_globals(fusang_mod, taxa_num: int) -> np.ndarray:
    """Populates upstream `fusang` module globals for `taxa_num` and returns `(Q, 4)` quartet indices."""
    start_end_list = [None, None, None]
    end = -1
    for i in range(3, 100):
        start = end + 1
        end = start + int(fusang_mod.comb_math(i, 3)) - 1
        start_end_list.append((start, end))

    id_for_taxa = list(range(taxa_num))
    comb_of_id = list(combinations(id_for_taxa, 4))
    comb_of_id.sort(key=lambda ele: ele[-1])

    leave_node_comb_name = []
    dic_for_leave_node_comb_name = {}
    for ele in comb_of_id:
        term = "".join(chr(ord("\u4e00") + idx) for idx in ele)
        dic_for_leave_node_comb_name[term] = len(dic_for_leave_node_comb_name)
        leave_node_comb_name.append(term)

    fusang_mod.taxa_num = taxa_num
    fusang_mod.start_end_list = start_end_list
    fusang_mod.comb_of_id = comb_of_id
    fusang_mod.leave_node_id = id_for_taxa
    fusang_mod.leave_node_name = [chr(ord("\u4e00") + i) for i in range(taxa_num)]
    fusang_mod.leave_node_comb_id = comb_of_id
    fusang_mod.leave_node_comb_name = leave_node_comb_name
    fusang_mod.dic_for_leave_node_comb_name = dic_for_leave_node_comb_name
    fusang_mod.internal_node_name_pool = [f"internal_node_{i}" for i in range(3, 3000)]

    return np.asarray(comb_of_id, dtype=np.int64)


class _FusangInProcessEngine:
    """Caches Fusang Keras weights and quartet tables across primary + bootstrap alignments."""

    def __init__(self, repo_dir: Path, seq_type: str, branch_model: str, beam_size: int) -> None:
        self.repo_dir = repo_dir.resolve()
        self.seq_type = seq_type
        self.branch_model = branch_model
        self.beam_size = beam_size
        self._models: dict[int, object] = {}
        self._cached_taxa_num: int | None = None
        self._comb_arr: np.ndarray | None = None

        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        import fusang as fusang_mod

        self.tf = tf
        self.fusang = fusang_mod

    def _get_model(self, bucket: int):
        if bucket not in self._models:
            prev_cwd = os.getcwd()
            os.chdir(str(self.repo_dir))
            try:
                if bucket == 240:
                    mdl = self.fusang.get_dl_model_240()
                    w_path = WEIGHT_SUBDIR_240[(self.seq_type, self.branch_model)]
                else:
                    mdl = self.fusang.get_dl_model_1200()
                    w_path = WEIGHT_SUBDIR_1200[(self.seq_type, self.branch_model)]
                mdl.load_weights(filepath=w_path).expect_partial()
                self._models[bucket] = mdl
            finally:
                os.chdir(prev_cwd)
        return self._models[bucket]

    def infer_one(self, fasta_file: Path, out_newick: Path) -> None:
        abs_fasta = fasta_file.resolve()
        abs_out = out_newick.resolve()
        abs_out.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            clean_fa = tmp_path / "clean.fasta"
            indexed_fa = tmp_path / "indexed_fusang.fas"
            _prepare_fusang_compatible_fasta(abs_fasta, clean_fa)

            alignment = AlignIO.read(str(clean_fa), "fasta")
            len_of_msa = len(alignment[0].seq)
            taxa_num = len(alignment)

            taxa_name: dict[int, str] = {}
            with open(indexed_fa, "w") as f:
                for record in alignment:
                    idx = len(taxa_name)
                    taxa_name[idx] = record.id
                    f.write(f">{idx}\n{record.seq}\n")

            org_seq = self.fusang.get_numpy(str(indexed_fa))

        if self._cached_taxa_num != taxa_num or self._comb_arr is None:
            self._comb_arr = _configure_fusang_taxa_globals(self.fusang, taxa_num)
            self._cached_taxa_num = taxa_num
        else:
            # Reset internal node name pool for deterministic tree search per alignment
            self.fusang.internal_node_name_pool = [f"internal_node_{i}" for i in range(3, 3000)]

        self.fusang.len_of_msa = len_of_msa
        comb_arr = self._comb_arr
        n_quartets = comb_arr.shape[0]

        if len_of_msa <= 1210:
            win_len = 240
            dl_model = self._get_model(240)
            window_number = int(len_of_msa // 240 + 1)
            step = ((len_of_msa - 240) // window_number) if len_of_msa > 240 else ((250 - 240) // window_number)
        else:
            win_len = 1200
            dl_model = self._get_model(1200)
            window_number = int(len_of_msa // 1200 + 1)
            step = (len_of_msa - 1200) // window_number

        dl_predict = np.zeros((n_quartets, 3), dtype=np.float64)
        start_idx = 0
        chunk_size = 50000

        for _ in range(window_number):
            end_idx = start_idx + win_len
            win_slice = org_seq[0, :, start_idx:end_idx]
            for c_start in range(0, n_quartets, chunk_size):
                c_end = min(c_start + chunk_size, n_quartets)
                batch_seq = win_slice[comb_arr[c_start:c_end]].astype(np.float32)
                preds = dl_model.predict(batch_seq[..., np.newaxis], batch_size=256, verbose=0)
                dl_predict[c_start:c_end] += preds
            start_idx += step

        dl_predict /= float(window_number)

        prev_cwd = os.getcwd()
        os.chdir(str(self.repo_dir))
        try:
            if taxa_num > 10:
                raw_tree = self.fusang.gen_phylogenetic_tree_2(dl_predict, int(self.beam_size))
            else:
                raw_tree = self.fusang.gen_phylogenetic_tree(dl_predict, int(self.beam_size))
            searched_tree = self.fusang.transform_str(raw_tree, taxa_name).strip()
        finally:
            os.chdir(prev_cwd)

        if not (searched_tree.startswith("(") and ";" in searched_tree):
            raise RuntimeError(f"Fusang emitted invalid Newick string for {abs_fasta}: {searched_tree!r}")
        abs_out.write_text(searched_tree + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="In-Process Batch CLI Adapter for Jerry-0591/Fusang.")
    parser.add_argument(
        "--fusang-repo",
        default=os.environ.get("FUSANG_REPO", "external_models/Fusang"),
        help="Path to cloned Jerry-0591/Fusang repository containing fusang.py.",
    )
    parser.add_argument("--input-fasta", help="Single input FASTA path.")
    parser.add_argument("--output-newick", help="Single output Newick path.")
    parser.add_argument("--msa-dir", help="Batch directory containing primary.fasta and rep_*.fasta.")
    parser.add_argument("--out-dir", help="Batch output directory for <stem>.nwk topologies.")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument(
        "--sequence-type",
        default="standard",
        choices=["standard", "coding", "noncoding"],
    )
    parser.add_argument(
        "--branch-model",
        default="gamma",
        choices=["gamma", "uniform"],
    )
    args = parser.parse_args()

    repo_dir = Path(args.fusang_repo).resolve()
    fusang_script = repo_dir / "fusang.py"
    if not fusang_script.exists():
        raise FileNotFoundError(
            f"Could not find fusang.py at {fusang_script}. Clone https://github.com/Jerry-0591/Fusang.git first."
        )

    engine = _FusangInProcessEngine(
        repo_dir=repo_dir,
        seq_type=args.sequence_type,
        branch_model=args.branch_model,
        beam_size=args.beam_size,
    )

    if args.msa_dir and args.out_dir:
        in_dir = Path(args.msa_dir).resolve()
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        fasta_files = sorted(list(in_dir.glob("*.fasta")) + list(in_dir.glob("*.fa")))
        total = len(fasta_files)
        t0 = time.perf_counter()
        print(
            f"           [Fusang Adapter] Loaded Keras model once (CPU mode) | Inferring {total} alignments...",
            flush=True,
        )
        for idx, fa in enumerate(fasta_files, start=1):
            engine.infer_one(fa, out_dir / f"{fa.stem}.nwk")
            if idx == 1 or idx == total or idx % 5 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"           [Fusang Adapter] Progress: {idx}/{total} alignments ({elapsed:.1f}s elapsed)",
                    flush=True,
                )
    elif args.input_fasta and args.output_newick:
        engine.infer_one(Path(args.input_fasta), Path(args.output_newick))
    else:
        raise ValueError(
            "Provide either (--msa-dir and --out-dir) or (--input-fasta and --output-newick)."
        )


if __name__ == "__main__":
    main()
