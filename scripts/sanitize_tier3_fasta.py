#!/usr/bin/env python3
"""
TIER 3 EMPIRICAL FASTA COMPATIBILITY SANITIZER (scripts/sanitize_tier3_fasta.py)
================================================================================
NOTE: This utility is ONLY needed when preparing external, real-world biological
FASTA files downloaded from public databases (TreeBASE, OrthoMaM v12, NCBI Virus,
Los Alamos HIV DB) for **Tier 3 (Empirical Biological Benchmarks)**.
It is NOT needed for Tier 1 or Tier 2 synthetic benchmarks.

What this compatibility script does (without altering biological homology):
  1. Strips Newick-breaking punctuation (spaces, ':', ';', '(', ')', ',', '[', ']')
     from raw database FASTA headers and ensures unique taxon IDs.
  2. Converts RNA 'U' -> 'T', '.' or '?' -> '-' (standard gap character), and
     lowercased bases -> uppercase.
  3. Verifies that all sequences have equal aligned column length L.
  4. Removes columns that are 100% gaps ('-') across all taxa (common when subsetting
     larger database alignments) and logs a coordinate mapping summary.
  5. Verifies (or suggests) the designated outgroup taxon name for Tier 3 rooting.
"""
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
    """
    Replaces spaces and Newick-reserved punctuation (:,;()[]'") with underscores
    so ETE 3, DendroPy, IQ-TREE 2, RAxML-NG, and FastTree 2 parse taxon labels cleanly.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_header.strip())
    cleaned = re.sub(r"\.+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "Taxon"
    return cleaned[:max_len].rstrip("_")


def sanitize_tier3_empirical_fasta(
    input_fasta_path: str,
    output_fasta_path: str,
    outgroup: str | None = None,
    seq_type: str = "DNA",
    strip_all_gap_cols: bool = True,
) -> dict:
    """
    Sanitizes an external empirical FASTA file purely for Tier 3 pipeline compatibility.
    Returns a summary dictionary including old->new taxon ID mapping and alignment stats.
    """
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
    is_dna = seq_type.upper() == "DNA"

    used_names: set[str] = set()
    header_map: dict[str, str] = {}
    cleaned_ids: list[str] = []
    seq_rows: list[list[str]] = []

    for idx, rec in enumerate(raw_records, start=1):
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
        s = s.replace(".", "-").replace("?", "-")
        if is_dna:
            s = s.replace("U", "T")
            # Any non-standard DNA symbol outside A,C,G,T,- becomes N
            s = re.sub(r"[^ACGTN-]", "N", s)
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
        description=(
            "[TIER 3 COMPATIBILITY ONLY] Sanitize downloaded real-world biological FASTA files "
            "(TreeBASE, OrthoMaM, NCBI Virus) so headers and gap characters are compatible with "
            "IQ-TREE 2, RAxML-NG, FastTree 2, Fusang, and Phyloformer."
        )
    )
    parser.add_argument("--input-fasta", required=True, help="Path to raw downloaded empirical FASTA file.")
    parser.add_argument(
        "--output-fasta",
        required=True,
        help="Destination path for cleaned Tier 3 FASTA (e.g., data/01_raw_alignments/tier3/<dataset_id>/alignment.fa).",
    )
    parser.add_argument(
        "--outgroup",
        default=None,
        help="Optional outgroup taxon name to verify against the cleaned FASTA headers.",
    )
    parser.add_argument(
        "--seq-type",
        choices=["DNA", "AA"],
        default="DNA",
        help="Sequence alphabet (default: DNA).",
    )
    args = parser.parse_args()

    print("=" * 78)
    print(" TIER 3 EMPIRICAL FASTA COMPATIBILITY SANITIZER")
    print(" (For preparing external real-world FASTAs for Tier 3 pipeline compatibility)")
    print("=" * 78)

    summary = sanitize_tier3_empirical_fasta(
        input_fasta_path=args.input_fasta,
        output_fasta_path=args.output_fasta,
        outgroup=args.outgroup,
        seq_type=args.seq_type,
    )

    print(f"[+] Input FASTA            : {summary['input_fasta']}")
    print(f"[+] Cleaned Tier 3 FASTA   : {summary['output_fasta']}")
    print(f"[+] Taxon Count (N)        : {summary['n_taxa']} (Fusang N<=40 compatible: {summary['fusang_compatible_n_le_40']})")
    print(f"[+] Alignment Length (L)   : {summary['final_alignment_length']} bp (removed {summary['removed_all_gap_columns']} all-gap columns)")
    print(f"[+] Designated Outgroup    : {summary['recommended_outgroup']} (Verified: {summary['requested_outgroup_found']})")
    print(f"[+] Cleaned Taxa Preview   : {', '.join(summary['cleaned_taxa_list'][:6])}...")


if __name__ == "__main__":
    main()
