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


def clean_taxon_header(raw_header: str, max_len: int = 48) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_header.strip())
    cleaned = re.sub(r"\.+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "Taxon"
    return cleaned[:max_len].rstrip("_")


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

    removed_all_gap_cols = 0
    if strip_all_gap_cols:
        non_empty_mask = ~np.all(char_mat == "-", axis=0)
        removed_all_gap_cols = int(np.sum(~non_empty_mask))
        char_mat = char_mat[:, non_empty_mask]

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
        if outgroup in cleaned_ids:
            resolved_outgroup = outgroup
        else:
            cleaned_og = clean_taxon_header(outgroup)
            if cleaned_og in cleaned_ids:
                resolved_outgroup = cleaned_og
            else:
                outgroup_matched = False

    summary = {
        "purpose": "Tier 3 Empirical Biological FASTA Compatibility Pre-Processing Only",
        "sequence_type": resolved_type,
        "input_fasta": str(input_fasta_path),
        "output_fasta": str(output_fasta_path),
        "n_taxa": len(cleaned_ids),
        "original_alignment_length": orig_len,
        "removed_all_gap_columns": removed_all_gap_cols,
        "final_alignment_length": final_len,
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
    parser.add_argument("--outgroup", default=None)
    parser.add_argument("--seq-type", choices=["AUTO", "DNA", "AA"], default="AUTO")
    args = parser.parse_args()

    summary = sanitize_tier3_empirical_fasta(
        input_fasta_path=args.input_fasta,
        output_fasta_path=args.output_fasta,
        outgroup=args.outgroup,
        seq_type=args.seq_type,
    )

    print(f"[+] Cleaned Tier 3 FASTA ({summary['sequence_type']}): {summary['output_fasta']}")
    print(f"[+] Taxon Count (N): {summary['n_taxa']} | Length (L): {summary['final_alignment_length']}")
    print(f"[+] Designated Outgroup: {summary['recommended_outgroup']} (Verified: {summary['requested_outgroup_found']})")


if __name__ == "__main__":
    main()
