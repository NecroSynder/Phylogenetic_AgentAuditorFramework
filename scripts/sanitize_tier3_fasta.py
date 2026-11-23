#!/usr/bin/env python3
"""
TIER 3 EMPIRICAL FASTA COMPATIBILITY SANITIZER (scripts/sanitize_tier3_fasta.py)
================================================================================
Sanitizes external real-world biological FASTA alignments (TreeBASE, OrthoMaM v12,
NCBI Virus, Los Alamos HIV DB) for both Nucleotide ('DNA') and Amino Acid ('AA'):
  1. Strips Newick-breaking punctuation from FASTA headers and ensures unique taxon IDs.
  2. Converts '.' / '?' / '*' (stop codons in AA) -> '-', RNA 'U' -> 'T' (in DNA),
     and sanitizes non-IUPAC symbols cleanly ('N' for DNA, 'X' for Protein).
  3. Verifies equal aligned column length L and strips 100%-gap columns.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from ete3 import Tree


def clean_taxon_header(raw_header: str, max_len: int = 48) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_header.strip())
    cleaned = re.sub(r"\.+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "Taxon"
    return cleaned[:max_len].rstrip("_")


def remap_site_coordinate(
    orig_site_1based: int, kept_original_columns_1based: list[int]
) -> int | None:
    """Maps a 1-based site index from the pre-sanitized MSA to its 1-based cleaned column index."""
    try:
        return kept_original_columns_1based.index(int(orig_site_1based)) + 1
    except ValueError:
        return None


def _parse_newick_flexible(raw_newick_text: str) -> tuple[Tree, bool]:
    """Parses published Newick strings (stripping NEXUS [&R]/NHX comments and supporting quoted ': / ()' labels)."""
    cleaned = re.sub(r"\[[^\]]*\]", "", raw_newick_text).strip()
    has_lengths = ":" in cleaned
    for quoted in (True, False):
        for fmt in (1, 0, 2, 5):
            try:
                t = Tree(cleaned, format=fmt, quoted_node_names=quoted)
                for lf in t.iter_leaves():
                    lf.name = lf.name.strip("'\"")
                return t, has_lengths
            except Exception:
                continue
    raise ValueError("Unable to parse input reference Newick string with ETE 3.")


def _infer_alphabet_from_records(records: list) -> str:
    chars = set()
    for rec in records[: min(10, len(records))]:
        chars.update(str(rec.seq).upper())
    chars -= {"-", ".", "?", "*", "N", "X"}
    dna_chars = {"A", "C", "G", "T", "U", "R", "Y", "S", "W", "K", "M", "B", "D", "H", "V"}
    return "DNA" if (chars and chars.issubset(dna_chars)) else "AA"


def sanitize_tier3_empirical_fasta(
    input_fasta_path: str,
    output_fasta_path: str,
    outgroup: str | None = None,
    seq_type: str = "AUTO",
    strip_all_gap_cols: bool = True,
    input_tree_path: str | None = None,
    output_tree_path: str | None = None,
    drop_empty_sequences: bool = True,
    deduplicate_identical: bool = False,
) -> dict:
    raw_records = list(SeqIO.parse(input_fasta_path, "fasta"))
    if len(raw_records) < 4:
        raise ValueError(
            f"Tier 3 compatibility check failed: '{input_fasta_path}' has {len(raw_records)} sequences "
            "(minimum 4 taxa required for phylogenetic tree rooting and quartet resolution)."
        )

    lengths = {len(rec.seq) for rec in raw_records}
    if len(lengths) != 1:
        raise ValueError(
            f"Tier 3 compatibility check failed: sequences in '{input_fasta_path}' have unequal lengths "
            f"{sorted(lengths)}. Please align the FASTA (e.g. with MAFFT) before running Tier 3."
        )

    orig_len = lengths.pop()
    resolved_type = (
        _infer_alphabet_from_records(raw_records)
        if seq_type.upper() == "AUTO"
        else seq_type.upper()
    )
    is_dna = resolved_type == "DNA"

    used_names: set[str] = set()
    header_map: dict[str, str] = {}
    raw_id_to_clean: dict[str, str] = {}
    cleaned_ids: list[str] = []
    seq_rows: list[list[str]] = []

    for rec in raw_records:
        full_title = rec.description if rec.description else rec.id
        base_clean = clean_taxon_header(rec.id)
        candidate = base_clean
        dup_counter = 2
        while candidate in used_names:
            candidate = f"{base_clean}_{dup_counter}"
            dup_counter += 1
        used_names.add(candidate)
        header_map[full_title] = candidate
        raw_id_to_clean[rec.id] = candidate
        raw_id_to_clean[rec.id.strip("'\"")] = candidate
        cleaned_ids.append(candidate)

        s = str(rec.seq).upper()
        s = s.replace(".", "-").replace("?", "-").replace("*", "-")
        if is_dna:
            s = s.replace("U", "T")
            s = re.sub(r"[^ACGTN-]", "N", s)
        else:
            # Standard 20 IUPAC amino acids + X ambiguity + '-' gap
            s = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX-]", "X", s)
        seq_rows.append(list(s))

    char_mat = np.array(seq_rows)

    # Resolve requested outgroup early so we never silently drop it during row screening
    requested_clean_og: str | None = None
    if outgroup:
        if outgroup in cleaned_ids:
            requested_clean_og = outgroup
        elif outgroup in raw_id_to_clean:
            requested_clean_og = raw_id_to_clean[outgroup]
        else:
            cleaned_og_cand = clean_taxon_header(outgroup)
            if cleaned_og_cand in cleaned_ids:
                requested_clean_og = cleaned_og_cand

    # Screen for 100%-gap / all-ambiguous ('-', 'N', 'X') sequences
    ambig_tokens = {"-", "N", "X"}
    removed_empty_taxa: list[str] = []
    if drop_empty_sequences:
        keep_row_indices: list[int] = []
        for r_idx, cid in enumerate(cleaned_ids):
            non_ambig = set(char_mat[r_idx].tolist()) - ambig_tokens
            if not non_ambig:
                if cid == requested_clean_og:
                    raise ValueError(
                        f"Designated outgroup '{cid}' contains 0 informative residues (100% gap/ambiguous)."
                    )
                removed_empty_taxa.append(cid)
            else:
                keep_row_indices.append(r_idx)
        if removed_empty_taxa:
            char_mat = char_mat[keep_row_indices]
            cleaned_ids = [cleaned_ids[i] for i in keep_row_indices]

    # Screen for 100%-identical sequences across taxa
    seq_to_taxa: dict[str, list[str]] = {}
    for r_idx, cid in enumerate(cleaned_ids):
        s_joined = "".join(char_mat[r_idx].tolist())
        seq_to_taxa.setdefault(s_joined, []).append(cid)
    duplicate_groups = [grp for grp in seq_to_taxa.values() if len(grp) > 1]
    removed_duplicate_taxa: list[str] = []
    if deduplicate_identical and duplicate_groups:
        drop_dups: set[str] = set()
        for grp in duplicate_groups:
            keeper = requested_clean_og if requested_clean_og in grp else grp[0]
            for member in grp:
                if member != keeper:
                    drop_dups.add(member)
        keep_dedup_idx = [i for i, cid in enumerate(cleaned_ids) if cid not in drop_dups]
        removed_duplicate_taxa = [cid for cid in cleaned_ids if cid in drop_dups]
        char_mat = char_mat[keep_dedup_idx]
        cleaned_ids = [cleaned_ids[i] for i in keep_dedup_idx]

    if len(cleaned_ids) < 4:
        raise ValueError(
            f"Tier 3 compatibility check failed: only {len(cleaned_ids)} valid non-empty sequences remain "
            "(minimum 4 taxa required)."
        )

    removed_all_gap_cols = 0
    if strip_all_gap_cols:
        non_empty_mask = ~np.all(char_mat == "-", axis=0)
        removed_all_gap_cols = int(np.sum(~non_empty_mask))
        kept_original_cols = [int(idx + 1) for idx in np.flatnonzero(non_empty_mask)]
        char_mat = char_mat[:, non_empty_mask]
    else:
        kept_original_cols = list(range(1, orig_len + 1))

    final_len = int(char_mat.shape[1])
    if final_len == 0:
        raise ValueError("Tier 3 compatibility check failed: alignment contains only gap characters.")

    out_path = Path(output_fasta_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sanitized_records = [
        SeqRecord(Seq("".join(char_mat[i])), id=cleaned_ids[i], description="")
        for i in range(len(cleaned_ids))
    ]
    with open(out_path, "w") as f:
        SeqIO.write(MultipleSeqAlignment(sanitized_records), f, "fasta")

    resolved_outgroup = cleaned_ids[0]
    outgroup_matched = True
    if outgroup:
        if requested_clean_og and requested_clean_og in cleaned_ids:
            resolved_outgroup = requested_clean_og
        else:
            outgroup_matched = False

    sanitized_tree_out: str | None = None
    if input_tree_path:
        raw_tree_str = Path(input_tree_path).read_text()
        t_ref, has_lengths = _parse_newick_flexible(raw_tree_str)
        clean_set = set(cleaned_ids)
        for lf in t_ref.iter_leaves():
            orig_lf = lf.name
            if orig_lf in clean_set:
                continue
            elif orig_lf in raw_id_to_clean:
                lf.name = raw_id_to_clean[orig_lf]
            elif orig_lf in header_map:
                lf.name = header_map[orig_lf]
            else:
                cand_lf = clean_taxon_header(orig_lf)
                if cand_lf in clean_set:
                    lf.name = cand_lf

        tree_leaves_before = {lf.name for lf in t_ref.iter_leaves()}
        missing_in_tree = sorted(clean_set - tree_leaves_before)
        if missing_in_tree:
            raise ValueError(
                f"Reference tree '{input_tree_path}' is missing {len(missing_in_tree)} sanitized FASTA taxa: "
                f"{missing_in_tree[:10]}"
            )
        if tree_leaves_before != clean_set:
            t_ref.prune(list(cleaned_ids), preserve_branch_length=has_lengths)

        dst_tree = Path(output_tree_path) if output_tree_path else (out_path.parent / "T_true.nwk")
        dst_tree.parent.mkdir(parents=True, exist_ok=True)
        # format=8 preserves both leaf and internal ancestor names on branchless pedigrees
        t_ref.write(format=1 if has_lengths else 8, outfile=str(dst_tree))
        sanitized_tree_out = str(dst_tree)

    summary = {
        "purpose": "Tier 3 Empirical Biological FASTA Compatibility Pre-Processing Only",
        "sequence_type": resolved_type,
        "input_fasta": str(input_fasta_path),
        "output_fasta": str(output_fasta_path),
        "input_tree": str(input_tree_path) if input_tree_path else None,
        "output_tree": sanitized_tree_out,
        "n_taxa": len(cleaned_ids),
        "original_alignment_length": orig_len,
        "removed_all_gap_columns": removed_all_gap_cols,
        "final_alignment_length": final_len,
        "kept_original_columns_1based": kept_original_cols,
        "removed_empty_taxa": removed_empty_taxa,
        "removed_duplicate_taxa": removed_duplicate_taxa,
        "has_identical_sequences": bool(duplicate_groups),
        "duplicate_sequence_groups": duplicate_groups,
        "fusang_compatible_n_le_40": len(cleaned_ids) <= 40,
        "phyloformer_compatible": True,
        "recommended_outgroup": resolved_outgroup,
        "requested_outgroup_found": outgroup_matched,
        "cleaned_taxa_list": cleaned_ids,
        "header_mapping": header_map,
    }
    meta_out = out_path.parent / f"{out_path.stem}_tier3_sanitized_report.json"
    with open(meta_out, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="[TIER 3 COMPATIBILITY ONLY] Sanitize downloaded real-world biological FASTA files."
    )
    parser.add_argument("--input-fasta", required=True)
    parser.add_argument("--output-fasta", required=True)
    parser.add_argument("--input-tree", default=None, help="Optional published reference Newick tree to remap leaf IDs.")
    parser.add_argument("--output-tree", default=None, help="Output path for remapped reference Newick tree.")
    parser.add_argument("--outgroup", default=None)
    parser.add_argument("--seq-type", choices=["AUTO", "DNA", "AA"], default="AUTO")
    parser.add_argument("--keep-empty-sequences", action="store_true")
    parser.add_argument("--deduplicate-identical", action="store_true")
    args = parser.parse_args()

    summary = sanitize_tier3_empirical_fasta(
        input_fasta_path=args.input_fasta,
        output_fasta_path=args.output_fasta,
        outgroup=args.outgroup,
        seq_type=args.seq_type,
        input_tree_path=args.input_tree,
        output_tree_path=args.output_tree,
        drop_empty_sequences=(not args.keep_empty_sequences),
        deduplicate_identical=args.deduplicate_identical,
    )

    print(f"[+] Cleaned Tier 3 FASTA ({summary['sequence_type']}): {summary['output_fasta']}")
    print(f"[+] Taxon Count (N): {summary['n_taxa']} | Length (L): {summary['final_alignment_length']}")
    print(f"[+] Designated Outgroup: {summary['recommended_outgroup']} (Verified: {summary['requested_outgroup_found']})")
    if summary["output_tree"]:
        print(f"[+] Remapped Reference Tree: {summary['output_tree']}")


if __name__ == "__main__":
    main()
