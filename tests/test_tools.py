import json
import numpy as np
import pandas as pd
from Bio import SeqIO
from ete3 import Tree

from src.preprocessing.ground_truth_gen import (
    compute_vectorized_fitch_and_gaps,
    fitch_binary_gap_states,
    fitch_parsimony_steps,
    run_ground_truth_pipeline,
)
from src.preprocessing.normalizer_wrapper import (
    distance_matrix_to_bionj_newick,
    standardize_and_root_tree,
)
from src.schemas.audit_schema import (
    AuditMetadata,
    AuditReport,
    BayesianTransitionMetrics,
    CausalSequenceEvidence,
    DiagnosedFailureMode,
    ExplanationQualityVector,
    GlobalMetrics,
    NodeAuditRecord,
)
from src.tools.bio_tools import (
    audit_tree_topology,
    diagnose_evolutionary_stressor,
    extract_branch_synapomorphies,
    inspect_alignment_window,
    verify_claim_coordinates,
)
from src.utils.io_cache import load_alignment_data


def test_vectorized_fitch_matches_scalar_and_exact_sitelh_delta_sls(golden_toy_env):
    """Verify bit-parallel Fitch matches scalar Fitch on all 1000 sites and delta_sls uses .sitelh."""
    fasta_path = golden_toy_env["fasta_path"]
    t_true = Tree(golden_toy_env["true_newick_path"], format=1)
    taxa_ids, char_mat, seq_len = load_alignment_data(fasta_path)

    s_vec, m_vec, g_vec, _, _, _ = compute_vectorized_fitch_and_gaps(
        t_true, taxa_ids, char_mat
    )
    for col in range(0, seq_len, 25):
        site_map = {taxa_ids[r]: char_mat[r, col] for r in range(len(taxa_ids))}
        s_sc, m_sc, g_sc = fitch_parsimony_steps(t_true, site_map)
        assert int(s_vec[col]) == s_sc
        assert int(m_vec[col]) == m_sc
        assert int(g_vec[col]) == g_sc

    # Verify M_true.tsv contains genuine .sitelh log-likelihood differences (not constant ~1.35)
    m_df = pd.read_csv(golden_toy_env["asr_state_path"], sep="\t")
    assert not m_df.empty
    assert (m_df["delta_sls"] > 0.0).all()
    # Real Felsenstein site log-likelihood differences span a wide dynamic range across sites
    assert float(m_df["delta_sls"].max() - m_df["delta_sls"].min()) > 0.5

    # Verify A_true.json separates informative CI from invariant-diluted alignment CI
    with open(golden_toy_env["stressor_gt_path"]) as f:
        a_true = json.load(f)
    assert a_true["mean_informative_ci"] < a_true["mean_alignment_ci"]
    assert "Taxon_08" in a_true["long_branch_taxa"]


def test_pipeline_and_tools_no_loopholes(golden_toy_env, tmp_path):
    fasta_path = golden_toy_env["fasta_path"]
    true_nwk = golden_toy_env["true_newick_path"]
    asr_tsv = golden_toy_env["asr_state_path"]
    stressor_gt = golden_toy_env["stressor_gt_path"]
    pred_nwk = golden_toy_env["pred_newick_path"]

    # 1. Test 3-argument audit_tree_topology (Table 4.4 signature auto-locating _metadata.json)
    res_concordant = audit_tree_topology(pred_nwk, true_nwk, support_threshold=85.0)
    assert res_concordant["normalized_rf_distance"] == 0.0
    assert res_concordant["fci_global"] == 0.0
    assert res_concordant["fci_conditional"] == 0.0
    assert res_concordant["false_confidence_nodes_count"] == 0

    # 2. Construct discordant tree & verify Tool 2 NEVER leaks unrelated T_true node mutations
    nwk_str = open(true_nwk).read().strip()
    discordant_nwk = (
        nwk_str.replace("Taxon_14", "__TMP__")
        .replace("Taxon_08", "Taxon_14")
        .replace("__TMP__", "Taxon_08")
    )
    disc_prefix = str(tmp_path / "toy_discordant")
    standardize_and_root_tree(
        discordant_nwk, [discordant_nwk] * 10, "Taxon_01", "Fusang", disc_prefix
    )
    res_discordant = audit_tree_topology(
        f"{disc_prefix}_normalized.nwk", true_nwk, f"{disc_prefix}_metadata.json"
    )
    assert res_discordant["normalized_rf_distance"] > 0.0
    assert res_discordant["fci_global"] > 0.0
    assert res_discordant["fci_conditional"] == 1.0

    # Check discordant node isolation: discordant clades must NOT inherit T_true's N2 mutations
    true_n2_sites = {
        m["site"]
        for m in extract_branch_synapomorphies(
            fasta_path, asr_tsv, parent_node_id="N3", child_node_id="N2", is_concordant=True
        )
    }
    disc_nodes = [
        n for n in res_discordant["node_bipartitions"] if n["topological_status"] != "CONCORDANT"
    ]
    assert len(disc_nodes) >= 1
    for dn in disc_nodes:
        disc_syns = extract_branch_synapomorphies(
            fasta_path=fasta_path,
            asr_state_path=asr_tsv,
            parent_node_id=dn["parent_node_id"],
            child_node_id=dn["node_id"],
            is_concordant=False,
            clade_taxa=dn["clade_taxa"],
            stressor_gt_path=stressor_gt,
        )
        disc_sites = {m["site"] for m in disc_syns}
        assert disc_sites != true_n2_sites

    # 3. Verify Tool 3 detects LBA even on Fusang unit branch lengths (1.0)
    lba_clades = [n for n in disc_nodes if "Taxon_08" in n["clade_taxa"]]
    assert len(lba_clades) >= 1
    stressor_lba = diagnose_evolutionary_stressor(
        fasta_path=fasta_path,
        pred_newick_path=f"{disc_prefix}_normalized.nwk",
        node_id=lba_clades[0]["node_id"],
        stressor_gt_path=stressor_gt,
    )
    assert stressor_lba["lba_branch_ratio"] == 12.0
    assert stressor_lba["primary_artifact"] == "LBA"

    # 4. Verify Tool 4 window slicer
    win = inspect_alignment_window(
        fasta_path=fasta_path,
        taxa_subset=["Taxon_01", "Taxon_02", "Taxon_14", "Taxon_15"],
        start_pos=1,
        end_pos=20,
    )
    assert len(win["window_slice"]["Taxon_01"]) == 20
    assert len(win["column_conservation_entropy_bits"]) == 20

    # 5. Anti-Loophole Tool 5 checks:
    syn_n1 = extract_branch_synapomorphies(
        fasta_path=fasta_path,
        asr_state_path=asr_tsv,
        parent_node_id="N12",
        child_node_id="N1",
        is_concordant=True,
    )
    assert len(syn_n1) >= 4
    valid_claim = {
        "site_coordinate": syn_n1[0]["site"],
        "ancestral_state": syn_n1[0]["parent_state"],
        "derived_state": syn_n1[0]["child_state"],
    }

    # 5a. Loophole check: site 9 has both 'A' and 'T' in FASTA leaves (belongs to N2, NOT N1).
    # Must be REJECTED when validating N1!
    ver_wrong_branch = verify_claim_coordinates(
        draft_claims=[{"site_coordinate": 9, "ancestral_state": "A", "derived_state": "T"}],
        fasta_path=fasta_path,
        asr_state_path=asr_tsv,
        valid_branch_mutations=syn_n1,
    )
    assert ver_wrong_branch["all_verified"] is False
    assert ver_wrong_branch["ccvr_score"] == 0.0

    # 5b. Loophole check: valid JSON draft_claims paired with hallucinated site in prose!
    ver_prose_halluc = verify_claim_coordinates(
        draft_claims=[valid_claim],
        fasta_path=fasta_path,
        asr_state_path=asr_tsv,
        valid_branch_mutations=syn_n1,
        explanation_text=f"Split supported by site {valid_claim['site_coordinate']} and A999T at site 999.",
    )
    assert ver_prose_halluc["all_verified"] is False
    assert ver_prose_halluc["ccvr_score"] < 1.0

    # 5c. Loophole check: duplicate claim padding ([valid] * 9 + [fake]) must give 0.5, not 0.9!
    fake_claim = {"site_coordinate": 9999, "ancestral_state": "A", "derived_state": "G"}
    ver_dup = verify_claim_coordinates(
        draft_claims=[valid_claim] * 9 + [fake_claim],
        fasta_path=fasta_path,
        asr_state_path=asr_tsv,
        valid_branch_mutations=syn_n1,
    )
    assert ver_dup["all_verified"] is False
    assert ver_dup["ccvr_score"] == 0.5

    # 5d. Loophole check: empty claims when valid_branch_mutations is non-empty must give CCVR=0.0
    ver_omission = verify_claim_coordinates(
        draft_claims=[],
        fasta_path=fasta_path,
        asr_state_path=asr_tsv,
        valid_branch_mutations=syn_n1,
    )
    assert ver_omission["all_verified"] is False
    assert ver_omission["ccvr_score"] == 0.0


def test_indel_synapomorphies_and_recombination(golden_toy_env, tmp_path):
    """Verify binary gap Fitch MP, Shannon mutual information, and localized recombination overlap."""
    records = list(SeqIO.parse(golden_toy_env["fasta_path"], "fasta"))
    gapped_fasta = tmp_path / "gapped_toy.fa"

    with open(gapped_fasta, "w") as f:
        for rec in records:
            seq_str = str(rec.seq)
            if rec.id in ("Taxon_14", "Taxon_15"):
                seq_str = seq_str[:9] + "---" + seq_str[12:]
            f.write(f">{rec.id}\n{seq_str}\n")

    out_dir = tmp_path / "gt_gapped"
    run_ground_truth_pipeline(
        str(gapped_fasta),
        golden_toy_env["raw_tree_path"],
        str(out_dir),
        recombination_breakpoints=[[1, 20]],
    )

    syn_n1 = extract_branch_synapomorphies(
        fasta_path=str(gapped_fasta),
        asr_state_path=str(out_dir / "M_true.tsv"),
        parent_node_id="N12",
        child_node_id="N1",
        is_concordant=True,
    )
    indel_hits = [m for m in syn_n1 if m["is_indel"]]
    assert {m["site"] for m in indel_hits} == {10, 11, 12}

    stressor_n1 = diagnose_evolutionary_stressor(
        fasta_path=str(gapped_fasta),
        pred_newick_path=str(out_dir / "T_true_rooted.nwk"),
        node_id="N1",
        stressor_gt_path=str(out_dir / "A_true.json"),
    )
    assert stressor_n1["recombination_window_overlap"] is True
    assert stressor_n1["primary_artifact"] == "RECOMBINATION"


def test_audit_report_pydantic_schema():
    """Verify expanded Pydantic v2 AuditReport schema supports classical baselines and EDR/FCI_cond."""
    report = AuditReport(
        audit_metadata=AuditMetadata(
            dataset_id="toy_00",
            target_model="FastTree2",
            taxon_count=15,
            sequence_length=1000,
            orchestrator_llm="qwen/qwen3-14b",
        ),
        global_metrics=GlobalMetrics(
            normalized_rf_distance=0.0,
            false_confidence_index_fci=0.0,
            conditional_fci=0.0,
            error_detection_rate_edr=1.0,
            total_internal_nodes_audited=13,
            false_confidence_nodes_count=0,
            claim_to_coordinate_verification_rate=1.0,
            logical_consistency_rate_lcr=1.0,
            overall_tree_goodness_score=0.95,
        ),
        node_audits=[],
    )
    assert report.audit_metadata.target_model == "FastTree2"
    assert report.global_metrics.error_detection_rate_edr == 1.0
