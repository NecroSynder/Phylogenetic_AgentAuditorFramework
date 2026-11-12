import json
import numpy as np
from Bio import SeqIO
from src.preprocessing.ground_truth_gen import (
    run_ground_truth_pipeline,
    fitch_binary_gap_states,
    shannon_mutual_information_gap_bits,
)
from src.preprocessing.normalizer_wrapper import (
    standardize_and_root_tree,
    distance_matrix_to_bionj_newick,
)
from src.tools.bio_tools import (
    audit_tree_topology,
    extract_branch_synapomorphies,
    diagnose_evolutionary_stressor,
    inspect_alignment_window,
    verify_claim_coordinates,
)
from src.schemas.audit_schema import (
    AuditMetadata,
    GlobalMetrics,
    BayesianTransitionMetrics,
    ExplanationQualityVector,
    CausalSequenceEvidence,
    DiagnosedFailureMode,
    NodeAuditRecord,
    AuditReport,
)


def test_pipeline_and_tools(tmp_path):
    # 1. Generate Ground Truths (T_true, M_true, A_true)
    run_ground_truth_pipeline(
        "data/00_golden_toy/toy_alignment.fa",
        "data/00_golden_toy/T_true.nwk",
        "data/02_ground_truths/toy",
    )
    with open("data/02_ground_truths/toy/T_true_rooted.nwk") as f:
        nwk_str = f.read().strip()

    with open("data/02_ground_truths/toy/A_true.json") as f:
        a_true = json.load(f)
    assert "global_rho_lba" in a_true
    assert "gap_tracts" in a_true
    assert "max_gap_tract_length" in a_true
    assert "branch_indel_synapomorphies" in a_true
    assert "recombination_breakpoints" in a_true

    # 2. Test Fusang Normalizer Wrapper
    standardize_and_root_tree(
        nwk_str,
        [nwk_str] * 10,
        "Taxon_01",
        "Fusang",
        "data/03_normalized_preds/toy_fusang",
    )

    # 3. Test Phyloformer BIONJ Distance Matrix Normalizer
    dummy_dist = np.array(
        [
            [0.0, 0.1, 0.4, 0.5],
            [0.1, 0.0, 0.4, 0.5],
            [0.4, 0.4, 0.0, 0.1],
            [0.5, 0.5, 0.1, 0.0],
        ]
    )
    bionj_nwk = distance_matrix_to_bionj_newick(dummy_dist, ["T1", "T2", "T3", "T4"])
    assert "T1:" in bionj_nwk and ";" in bionj_nwk

    # 4. Test Tool 1: audit_tree_topology (Concordant & False Confidence Discordant)
    res_concordant = audit_tree_topology(
        "data/03_normalized_preds/toy_fusang_normalized.nwk",
        "data/02_ground_truths/toy/T_true_rooted.nwk",
        "data/03_normalized_preds/toy_fusang_metadata.json",
    )
    assert res_concordant["normalized_rf_distance"] == 0.0
    assert res_concordant["fci_global"] == 0.0
    assert res_concordant["false_confidence_nodes_count"] == 0

    # Construct a discordant tree with high support (swap Taxon_14 and Taxon_12)
    discordant_nwk = (
        nwk_str.replace("Taxon_14", "__TMP__")
        .replace("Taxon_12", "Taxon_14")
        .replace("__TMP__", "Taxon_12")
    )
    disc_prefix = str(tmp_path / "toy_discordant")
    standardize_and_root_tree(
        discordant_nwk,
        [discordant_nwk] * 10,
        "Taxon_01",
        "Fusang",
        disc_prefix,
    )
    res_discordant = audit_tree_topology(
        f"{disc_prefix}_normalized.nwk",
        "data/02_ground_truths/toy/T_true_rooted.nwk",
        f"{disc_prefix}_metadata.json",
    )
    assert res_discordant["normalized_rf_distance"] > 0.0
    assert res_discordant["fci_global"] > 0.0
    assert res_discordant["false_confidence_nodes_count"] > 0
    assert any(
        n["topological_status"] == "FALSE_CONFIDENCE_ERROR"
        for n in res_discordant["node_bipartitions"]
    )

    # 5. Test Tool 2: extract_branch_synapomorphies
    syn_n1 = extract_branch_synapomorphies(
        fasta_path="data/00_golden_toy/toy_alignment.fa",
        asr_state_path="data/02_ground_truths/toy/M_true.tsv",
        parent_node_id="N12",
        child_node_id="N1",
    )
    assert len(syn_n1) >= 5
    sites_n1 = {m["site"] for m in syn_n1}
    assert {4, 128, 464, 505, 629}.issubset(sites_n1)
    for m in syn_n1:
        assert m["posterior_prob"] >= 0.90
        assert isinstance(m["is_indel"], bool)
        assert "indel_boundary" in m

    # 6. Test Tool 3: diagnose_evolutionary_stressor
    stressor_n1 = diagnose_evolutionary_stressor(
        fasta_path="data/00_golden_toy/toy_alignment.fa",
        pred_newick_path="data/02_ground_truths/toy/T_true_rooted.nwk",
        node_id="N1",
        stressor_gt_path="data/02_ground_truths/toy/A_true.json",
    )
    assert 0.0 <= stressor_n1["mean_ci"] <= 1.0
    assert 0.0 <= stressor_n1["mean_ri"] <= 1.0
    assert stressor_n1["lba_branch_ratio"] == 12.0
    assert isinstance(stressor_n1["recombination_window_overlap"], bool)
    assert stressor_n1["primary_artifact"] in {
        "NONE",
        "LBA",
        "INDEL_GAP",
        "RECOMBINATION",
        "SHORT_BRANCH_HOMOPLASY",
        "SHORT_BRANCH_HOMOPLASY_AND_LBA",
    }

    # 7. Test Tool 4: inspect_alignment_window
    win = inspect_alignment_window(
        fasta_path="data/00_golden_toy/toy_alignment.fa",
        taxa_subset=["Taxon_01", "Taxon_02", "Taxon_14", "Taxon_15"],
        start_pos=1,
        end_pos=20,
    )
    assert win["start_pos"] == 1
    assert win["end_pos"] == 20
    assert len(win["window_slice"]["Taxon_01"]) == 20
    assert len(win["column_conservation_entropy_bits"]) == 20
    assert len(win["column_gap_percentages"]) == 20
    assert win["mean_gap_density_pct"] == 0.0
    assert win["mean_shannon_entropy_bits"] >= 0.0

    # 8. Test Tool 5: verify_claim_coordinates (Valid vs Hallucinated Claims)
    valid_claims = [
        {
            "site_coordinate": syn_n1[0]["site"],
            "ancestral_state": syn_n1[0]["parent_state"],
            "derived_state": syn_n1[0]["child_state"],
        },
        {
            "site_coordinate": syn_n1[1]["site"],
            "ancestral_state": syn_n1[1]["parent_state"],
            "derived_state": syn_n1[1]["child_state"],
        },
    ]
    ver_valid = verify_claim_coordinates(
        draft_claims=valid_claims,
        fasta_path="data/00_golden_toy/toy_alignment.fa",
        asr_state_path="data/02_ground_truths/toy/M_true.tsv",
    )
    assert ver_valid["all_verified"] is True
    assert ver_valid["ccvr_score"] == 1.0
    assert len(ver_valid["mismatched_claims"]) == 0

    hallucinated_claims = [
        valid_claims[0],
        {
            "site_coordinate": 9999,
            "ancestral_state": "A",
            "derived_state": "G",
        },
        {
            "site_coordinate": syn_n1[0]["site"],
            "ancestral_state": "Z",
            "derived_state": "Q",
        },
    ]
    ver_halluc = verify_claim_coordinates(
        draft_claims=hallucinated_claims,
        fasta_path="data/00_golden_toy/toy_alignment.fa",
        asr_state_path="data/02_ground_truths/toy/M_true.tsv",
    )
    assert ver_halluc["all_verified"] is False
    assert abs(ver_halluc["ccvr_score"] - (1.0 / 3.0)) < 1e-3
    assert len(ver_halluc["mismatched_claims"]) == 2


def test_indel_synapomorphies_and_recombination(tmp_path):
    """Verify binary gap Fitch MP, Shannon mutual information I(X_i; b_k), and contiguous indel synapomorphies."""
    records = list(SeqIO.parse("data/00_golden_toy/toy_alignment.fa", "fasta"))
    gapped_fasta = tmp_path / "gapped_toy.fa"

    # Inject a shared 3-bp deletion (sites 10..12, 1-indexed) in clade N1 (Taxon_14, Taxon_15)
    with open(gapped_fasta, "w") as f:
        for rec in records:
            seq_str = str(rec.seq)
            if rec.id in ("Taxon_14", "Taxon_15"):
                seq_str = seq_str[:9] + "---" + seq_str[12:]
            f.write(f">{rec.id}\n{seq_str}\n")

    out_dir = tmp_path / "gt_gapped"
    run_ground_truth_pipeline(
        str(gapped_fasta),
        "data/00_golden_toy/T_true.nwk",
        str(out_dir),
        recombination_breakpoints=[[1, 20]],
    )

    with open(out_dir / "A_true.json") as f:
        a_true = json.load(f)

    assert a_true["mean_gap_density_pct"] > 0.0
    assert a_true["max_gap_tract_length"] == 3
    assert any(
        t["i_start"] == 10 and t["i_end"] == 12 and t["l_gap"] == 3
        for t in a_true["gap_tracts"]
    )
    assert "N12->N1" in a_true["branch_indel_synapomorphies"]
    n1_tracts = a_true["branch_indel_synapomorphies"]["N12->N1"]
    assert any(
        t["i_start"] == 10 and t["i_end"] == 12 and t["event"] == "DELETION"
        for t in n1_tracts
    )

    # Check Tool 2 extracts the indel synapomorphies on N12 -> N1
    syn_n1 = extract_branch_synapomorphies(
        fasta_path=str(gapped_fasta),
        asr_state_path=str(out_dir / "M_true.tsv"),
        parent_node_id="N12",
        child_node_id="N1",
    )
    indel_hits = [m for m in syn_n1 if m["is_indel"]]
    assert len(indel_hits) == 3
    assert {m["site"] for m in indel_hits} == {10, 11, 12}
    for m in indel_hits:
        assert m["indel_boundary"] == "10-12"
        assert m["delta_sls"] > 0.0

    # Check Tool 3 detects RECOMBINATION overlap when breakpoints are present
    stressor_n1 = diagnose_evolutionary_stressor(
        fasta_path=str(gapped_fasta),
        pred_newick_path=str(out_dir / "T_true_rooted.nwk"),
        node_id="N1",
        stressor_gt_path=str(out_dir / "A_true.json"),
    )
    assert stressor_n1["recombination_window_overlap"] is True
    assert stressor_n1["primary_artifact"] == "RECOMBINATION"

    # Check Tool 5 verifies indel claims
    ver_indel = verify_claim_coordinates(
        draft_claims=[
            {
                "site_coordinate": 10,
                "ancestral_state": indel_hits[0]["parent_state"],
                "derived_state": "-",
                "is_indel": True,
            }
        ],
        fasta_path=str(gapped_fasta),
        asr_state_path=str(out_dir / "M_true.tsv"),
    )
    assert ver_indel["all_verified"] is True
    assert ver_indel["ccvr_score"] == 1.0


def test_audit_report_pydantic_schema():
    """Verify Pydantic v2 AuditReport schema enforces bounds and serializes cleanly."""
    report = AuditReport(
        audit_metadata=AuditMetadata(
            dataset_id="toy_00",
            target_model="Fusang",
            taxon_count=15,
            sequence_length=1000,
            orchestrator_llm="Claude-3.5-Sonnet",
        ),
        global_metrics=GlobalMetrics(
            normalized_rf_distance=0.0,
            false_confidence_index_fci=0.0,
            total_internal_nodes_audited=13,
            false_confidence_nodes_count=0,
            claim_to_coordinate_verification_rate=1.0,
            logical_consistency_rate_lcr=1.0,
            overall_tree_goodness_score=0.95,
        ),
        node_audits=[
            NodeAuditRecord(
                node_id="N1",
                predicted_clade_taxa=["Taxon_14", "Taxon_15"],
                model_support_pct=100.0,
                topological_status="CONCORDANT",
                bayesian_transition_metrics=BayesianTransitionMetrics(
                    ctmc_branch_transition_loglik=-12.4,
                    clade_shannon_entropy_reduction_bits=1.38,
                    transition_information_gain_ig=0.92,
                ),
                explanation_quality_vector=ExplanationQualityVector(
                    ccvr_local=1.0,
                    bsf_local=1,
                    feature_f1=1.0,
                    information_gain_ig=0.92,
                    faithfulness_fgs=0.98,
                ),
                node_goodness_score=0.96,
                causal_sequence_evidence=[
                    CausalSequenceEvidence(
                        site_coordinate=4,
                        ancestral_state="G",
                        derived_state="C",
                        posterior_probability=0.9731,
                        consistency_index_ci=1.0,
                        mutation_class="SYNAPOMORPHY",
                    )
                ],
                diagnosed_failure_mode=DiagnosedFailureMode(
                    primary_stressor="NONE",
                    lba_branch_ratio=12.0,
                    local_gap_density_pct=0.0,
                    agent_causal_explanation="Branch N12->N1 is strongly supported by synapomorphy at site 4 (G->C, CI=1.0).",
                ),
            )
        ],
    )
    dumped = json.loads(report.model_dump_json())
    assert dumped["audit_metadata"]["dataset_id"] == "toy_00"
    assert dumped["node_audits"][0]["node_id"] == "N1"
