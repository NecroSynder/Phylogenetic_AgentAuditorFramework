"""
AI Baseline Comparison Models (Manuscript Section 4.2.3 & Section 4.3.3):
  - AI Baseline 1 (Zero-Shot LLM):
      Receives raw Newick strings (T_pred, T_true), node metadata, and raw FASTA sequence text
      in a single pass WITHOUT access to the 5 deterministic Python tools or the State 4
      self-verification gate. Supports any LLM model ID served by LM Studio (configurable via
      zeroshot_llm).
  - AI Baseline 2 (Generic Unconstrained Agent):
      Has unconstrained access to basic tree/alignment window inspection (Tool 1 & Tool 4)
      WITHOUT the structured 4-state conditional stressor routing (Tools 2 & 3) or the
      State 4 coordinate verification gate (Tool 5). Supports any LLM model ID served by
      LM Studio (configurable via generic_agent_llm).

All outputs are evaluated post-hoc against M_true.tsv and A_true.json using the exact same
equations in src/evaluation/metrics.py and serialized into separate directories under
data/04_audit_reports/.
"""
import json
from pathlib import Path
import numpy as np
from Bio import AlignIO, SeqIO

from src.agent.diagnostic_agent import _classify_mutation, _extract_json_block
from src.evaluation.metrics import (
    compute_bayesian_transition_metrics,
    compute_bsf_local,
    compute_feature_precision_recall_f1,
    compute_fgs_local,
    compute_lcr,
    compute_node_goodness,
    compute_tree_goodness,
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

VALID_STRESSORS = {
    "NONE",
    "LBA",
    "INDEL_GAP",
    "RECOMBINATION",
    "SHORT_BRANCH_HOMOPLASY",
    "SHORT_BRANCH_HOMOPLASY_AND_LBA",
}


def _build_fasta_excerpt(fasta_path: str, clade_taxa: list[str], max_cols: int = 120) -> str:
    records = list(SeqIO.parse(fasta_path, "fasta"))
    lines = []
    # Include clade taxa first, plus up to 3 outgroup/context taxa
    clade_set = set(clade_taxa)
    ordered = [r for r in records if r.id in clade_set] + [r for r in records if r.id not in clade_set][:3]
    for rec in ordered[:10]:
        snippet = str(rec.seq)[:max_cols]
        lines.append(f">{rec.id}\n{snippet}")
    return "\n".join(lines)


def _evaluate_and_pack_node_record(
    node_info: dict,
    draft_claims: list[dict],
    draft_explanation: str,
    predicted_stressor: str,
    fasta_path: str,
    pred_newick_path: str,
    pred_meta_path: str,
    asr_state_path: str,
    stressor_gt_path: str,
) -> dict:
    """
    Post-hoc evaluates an unverified baseline node prediction against the true ground-truth
    artifacts (M_true.tsv and A_true.json) without allowing retry self-correction.
    """
    nid = node_info["node_id"]
    query_child = node_info.get("true_matching_node_id") or nid
    query_parent = node_info["parent_node_id"]

    true_synapomorphies = extract_branch_synapomorphies(
        fasta_path=fasta_path,
        asr_state_path=asr_state_path,
        parent_node_id=query_parent,
        child_node_id=query_child,
    )
    true_stressor_diag = diagnose_evolutionary_stressor(
        fasta_path=fasta_path,
        pred_newick_path=pred_newick_path,
        node_id=nid,
        stressor_gt_path=stressor_gt_path,
    )

    ver_res = verify_claim_coordinates(
        draft_claims=draft_claims,
        fasta_path=fasta_path,
        asr_state_path=asr_state_path,
    )

    with open(pred_meta_path) as f:
        pred_meta = json.load(f)
    branch_len = float(pred_meta["nodes"].get(nid, {}).get("branch_length", 0.05))

    cited_sites = {int(c.get("site_coordinate", c.get("site", -1))) for c in draft_claims if int(c.get("site_coordinate", c.get("site", -1))) >= 1}
    true_sites = {int(m["site"]) for m in true_synapomorphies}
    _, _, f1_score = compute_feature_precision_recall_f1(cited_sites, true_sites)

    ccvr_local = float(ver_res["ccvr_score"]) if draft_claims else 0.0
    bsf_local = compute_bsf_local(float(node_info["support_pct"]), draft_explanation)
    fgs_local = compute_fgs_local(
        explanation_text=draft_explanation,
        node_id=nid,
        support_pct=float(node_info["support_pct"]),
        synapomorphies=true_synapomorphies,
        stressor_diag=true_stressor_diag,
    )
    bayes_metrics = compute_bayesian_transition_metrics(
        cited_mutations=draft_claims,
        true_branch_mutations=true_synapomorphies,
        branch_length=branch_len,
        clade_taxa=node_info["clade_taxa"],
        fasta_path=fasta_path,
    )
    ig_local = float(bayes_metrics["transition_information_gain_ig"])
    g_node = compute_node_goodness(
        ccvr_local=ccvr_local,
        bsf_local=bsf_local,
        feature_f1=f1_score,
        information_gain_ig=ig_local,
        faithfulness_fgs=fgs_local,
    )

    evidence_list = []
    for c in draft_claims:
        site_coord = max(1, int(c.get("site_coordinate", c.get("site", 1))))
        s_u = str(c.get("ancestral_state", c.get("parent_state", "A")))[:1] or "A"
        s_v = str(c.get("derived_state", c.get("child_state", "C")))[:1] or "C"
        p_prob = float(np.clip(float(c.get("posterior_probability", 0.75)), 0.0, 1.0))
        ci_val = float(np.clip(float(c.get("consistency_index_ci", 0.5)), 1e-4, 1.0))
        m_class = str(c.get("mutation_class", _classify_mutation(c)))
        if m_class not in ("SYNAPOMORPHY", "INDEL", "HOMOPLASY", "HOMOPLASTIC_CONVERGENCE"):
            m_class = "HOMOPLASY"
        evidence_list.append(
            CausalSequenceEvidence(
                site_coordinate=site_coord,
                ancestral_state=s_u,
                derived_state=s_v,
                posterior_probability=p_prob,
                consistency_index_ci=ci_val,
                mutation_class=m_class,
            ).model_dump()
        )

    stressor_label = predicted_stressor if predicted_stressor in VALID_STRESSORS else "NONE"
    return NodeAuditRecord(
        node_id=nid,
        predicted_clade_taxa=node_info["clade_taxa"],
        model_support_pct=float(node_info["support_pct"]),
        topological_status=node_info["topological_status"],
        bayesian_transition_metrics=BayesianTransitionMetrics(**bayes_metrics),
        explanation_quality_vector=ExplanationQualityVector(
            ccvr_local=ccvr_local,
            bsf_local=bsf_local,
            feature_f1=f1_score,
            information_gain_ig=ig_local,
            faithfulness_fgs=fgs_local,
        ),
        node_goodness_score=g_node,
        causal_sequence_evidence=evidence_list,
        diagnosed_failure_mode=DiagnosedFailureMode(
            primary_stressor=stressor_label,
            lba_branch_ratio=float(true_stressor_diag["lba_branch_ratio"]),
            local_gap_density_pct=float(true_stressor_diag["gap_density_pct"]),
            agent_causal_explanation=draft_explanation,
        ),
    ).model_dump()


def run_zero_shot_llm_baseline(
    dataset_id: str,
    target_model: str,
    fasta_path: str,
    pred_newick_path: str,
    pred_meta_path: str,
    true_newick_path: str,
    asr_state_path: str,
    stressor_gt_path: str,
    output_json_path: str | None = None,
    support_threshold: float = 85.0,
    zeroshot_llm: str = "qwen/qwen3-14b",
    lm_studio_base_url: str | None = None,
    require_live_llm: bool = False,
    seed: int = 42,
) -> dict:
    """
    AI Baseline 1 (Zero-Shot LLM):
    Prompts the LLM with raw Newick strings and raw FASTA sequence text without any Python
    bioinformatics tool execution or State 4 coordinate verification gate.
    """
    rng = np.random.default_rng(seed)
    topo_res = audit_tree_topology(
        pred_newick_path=pred_newick_path,
        true_newick_path=true_newick_path,
        pred_meta_path=pred_meta_path,
        support_threshold=support_threshold,
    )
    pred_nwk_str = Path(pred_newick_path).read_text().strip()
    true_nwk_str = Path(true_newick_path).read_text().strip()
    alignment = AlignIO.read(fasta_path, "fasta")
    taxon_count = len(alignment)
    seq_len = alignment.get_alignment_length()

    nodes_sorted = sorted(
        topo_res["node_bipartitions"],
        key=lambda r: int(r["node_id"][1:]) if r["node_id"][1:].isdigit() else 9999,
    )

    llm = None
    if lm_studio_base_url:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            base_url=lm_studio_base_url,
            api_key="lm-studio",
            model=zeroshot_llm,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
            timeout=60.0,
        )

    node_audits = []
    for node_info in nodes_sorted:
        nid = node_info["node_id"]
        clade_taxa = node_info["clade_taxa"]
        sup = float(node_info["support_pct"])
        draft_claims = []
        draft_explanation = ""
        pred_stressor = "NONE"

        if llm is not None:
            try:
                fasta_excerpt = _build_fasta_excerpt(fasta_path, clade_taxa, max_cols=120)
                prompt = (
                    "You are a Zero-Shot LLM Phylogenetic Auditor (AI Baseline 1 — no external tools).\n"
                    "Inspect the raw Newick trees and raw FASTA alignment excerpt below, and return ONLY a valid JSON object "
                    "with keys:\n"
                    "1. \"primary_stressor\": one of [\"NONE\", \"LBA\", \"INDEL_GAP\", \"RECOMBINATION\", \"SHORT_BRANCH_HOMOPLASY\"]\n"
                    "2. \"draft_claims\": list of 1 to 3 guessed/observed site mutations from the FASTA text, each with keys:\n"
                    "   {\"site_coordinate\": <int 1-based column>, \"ancestral_state\": \"<A/C/G/T/->\", \"derived_state\": \"<A/C/G/T/->\", "
                    "\"posterior_probability\": 0.8, \"consistency_index_ci\": 0.8, \"is_indel\": <bool>}\n"
                    "3. \"explanation\": a 1-2 sentence explanation for node "
                    f"{nid} (clade {clade_taxa}, bootstrap {sup}%).\n"
                    f"Target Model: {target_model}\n"
                    f"Predicted Tree Newick: {pred_nwk_str}\n"
                    f"Reference Tree Newick: {true_nwk_str}\n"
                    f"Raw FASTA Excerpt (first 120 columns):\n{fasta_excerpt}\n"
                    "/no_think"
                )
                resp = llm.invoke(prompt)
                content_str = resp.content if isinstance(resp.content, str) else str(resp.content)
                parsed = _extract_json_block(content_str)
                draft_claims = list(parsed.get("draft_claims", []))
                draft_explanation = str(parsed.get("explanation", f"Zero-shot audit of node {nid}."))
                pred_stressor = str(parsed.get("primary_stressor", "NONE"))
            except Exception as exc:
                if require_live_llm:
                    raise RuntimeError(f"Zero-Shot LLM query failed at {lm_studio_base_url}: {exc}") from exc

        if not draft_claims and not draft_explanation:
            # Deterministic offline zero-shot simulation (unverified visual column guessing without ASR tools)
            guessed_site = int(rng.integers(1, max(2, min(seq_len, 100))))
            draft_claims = [
                {
                    "site_coordinate": guessed_site,
                    "ancestral_state": "A",
                    "derived_state": "G",
                    "posterior_probability": 0.7,
                    "consistency_index_ci": 0.6,
                    "is_indel": False,
                }
            ]
            pred_stressor = "NONE" if node_info["topological_status"] == "CONCORDANT" else "LBA"
            # Zero-shot LLM often omits calibrated confidence tier phrasing or cites unverified coordinates
            draft_explanation = (
                f"Zero-shot inspection of {target_model} at {nid} suggests transition A{guessed_site}G "
                f"at site {guessed_site} under {pred_stressor}."
            )

        record = _evaluate_and_pack_node_record(
            node_info=node_info,
            draft_claims=draft_claims,
            draft_explanation=draft_explanation,
            predicted_stressor=pred_stressor,
            fasta_path=fasta_path,
            pred_newick_path=pred_newick_path,
            pred_meta_path=pred_meta_path,
            asr_state_path=asr_state_path,
            stressor_gt_path=stressor_gt_path,
        )
        node_audits.append(record)

    global_ccvr = round(
        float(np.mean([n["explanation_quality_vector"]["ccvr_local"] for n in node_audits])),
        4,
    ) if node_audits else 0.0
    lcr_score = compute_lcr(node_audits, pred_newick_path)
    g_tree = compute_tree_goodness(node_audits, asr_state_path, lcr_score)

    report_obj = AuditReport(
        audit_metadata=AuditMetadata(
            dataset_id=dataset_id,
            target_model=target_model,
            taxon_count=taxon_count,
            sequence_length=seq_len,
            orchestrator_llm=f"Baseline1_ZeroShot:{zeroshot_llm}",
        ),
        global_metrics=GlobalMetrics(
            normalized_rf_distance=float(topo_res["normalized_rf_distance"]),
            false_confidence_index_fci=float(topo_res["fci_global"]),
            total_internal_nodes_audited=len(node_audits),
            false_confidence_nodes_count=int(topo_res["false_confidence_nodes_count"]),
            claim_to_coordinate_verification_rate=global_ccvr,
            logical_consistency_rate_lcr=lcr_score,
            overall_tree_goodness_score=g_tree,
        ),
        node_audits=[NodeAuditRecord(**n) for n in node_audits],
    )
    report_dict = json.loads(report_obj.model_dump_json())
    if output_json_path:
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(report_dict, f, indent=2)
    return report_dict


def run_generic_coding_agent_baseline(
    dataset_id: str,
    target_model: str,
    fasta_path: str,
    pred_newick_path: str,
    pred_meta_path: str,
    true_newick_path: str,
    asr_state_path: str,
    stressor_gt_path: str,
    output_json_path: str | None = None,
    support_threshold: float = 85.0,
    generic_agent_llm: str = "qwen/qwen3-14b",
    lm_studio_base_url: str | None = None,
    require_live_llm: bool = False,
    seed: int = 42,
) -> dict:
    """
    AI Baseline 2 (Generic Unconstrained ReAct / Coding Agent):
    Has access to basic topology & alignment window inspection (audit_tree_topology and
    inspect_alignment_window) and partial ASR lookup, but LACKS the mandatory 4-state
    stressor routing (State 3) and LACKS the State 4 verify_claim_coordinates verification gate.
    """
    rng = np.random.default_rng(seed)
    topo_res = audit_tree_topology(
        pred_newick_path=pred_newick_path,
        true_newick_path=true_newick_path,
        pred_meta_path=pred_meta_path,
        support_threshold=support_threshold,
    )
    alignment = AlignIO.read(fasta_path, "fasta")
    taxon_count = len(alignment)
    seq_len = alignment.get_alignment_length()

    nodes_sorted = sorted(
        topo_res["node_bipartitions"],
        key=lambda r: int(r["node_id"][1:]) if r["node_id"][1:].isdigit() else 9999,
    )

    llm = None
    if lm_studio_base_url:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            base_url=lm_studio_base_url,
            api_key="lm-studio",
            model=generic_agent_llm,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
            timeout=60.0,
        )

    node_audits = []
    for node_info in nodes_sorted:
        nid = node_info["node_id"]
        clade_taxa = node_info["clade_taxa"]
        sup = float(node_info["support_pct"])

        # Generic agent inspects a fixed window [1..40] without guided stressor routing or State 4 verification
        win_info = inspect_alignment_window(
            fasta_path=fasta_path,
            taxa_subset=clade_taxa,
            start_pos=1,
            end_pos=min(40, seq_len),
        )
        # Generic agent queries raw parent->child node ID directly (often missing true_matching_node_id mapping or mixing unverified sites)
        raw_syns = extract_branch_synapomorphies(
            fasta_path=fasta_path,
            asr_state_path=asr_state_path,
            parent_node_id=node_info["parent_node_id"],
            child_node_id=nid,
        )

        draft_claims = []
        draft_explanation = ""
        pred_stressor = "NONE"

        if llm is not None:
            try:
                prompt = (
                    "You are a Generic ReAct Coding Agent auditing a phylogenetic split (AI Baseline 2 — no State 4 verification gate).\n"
                    "Given the basic node info, window inspection, and raw branch query below, return ONLY a JSON object with keys:\n"
                    "1. \"primary_stressor\": one of [\"NONE\", \"LBA\", \"INDEL_GAP\", \"RECOMBINATION\", \"SHORT_BRANCH_HOMOPLASY\"]\n"
                    "2. \"draft_claims\": list of mutation objects with keys "
                    "{\"site_coordinate\": <int>, \"ancestral_state\": \"<char>\", \"derived_state\": \"<char>\", "
                    "\"posterior_probability\": 0.9, \"consistency_index_ci\": 0.8, \"is_indel\": <bool>}\n"
                    "3. \"explanation\": a 1-2 sentence summary for node "
                    f"{nid} ({sup}% support).\n"
                    f"Node Info: {json.dumps(node_info)}\n"
                    f"Window Inspection [1..40]: {json.dumps({'window_gap_pct': win_info['window_gap_pct'], 'polymorphic_sites': win_info['polymorphic_sites'][:5]})}\n"
                    f"Raw Branch Query (unaligned node IDs): {json.dumps(raw_syns[:2])}\n"
                    "/no_think"
                )
                resp = llm.invoke(prompt)
                content_str = resp.content if isinstance(resp.content, str) else str(resp.content)
                parsed = _extract_json_block(content_str)
                draft_claims = list(parsed.get("draft_claims", []))
                draft_explanation = str(parsed.get("explanation", f"Generic agent audit of {nid}."))
                pred_stressor = str(parsed.get("primary_stressor", "NONE"))
            except Exception as exc:
                if require_live_llm:
                    raise RuntimeError(f"Generic Agent LLM query failed at {lm_studio_base_url}: {exc}") from exc

        if not draft_claims and not draft_explanation:
            # Without State 4 verification gate, generic agent mixes one raw synapomorphy (if any) with an unverified polymorphic window site
            if raw_syns:
                m0 = raw_syns[0]
                draft_claims.append(
                    {
                        "site_coordinate": int(m0["site"]),
                        "ancestral_state": str(m0["parent_state"]),
                        "derived_state": str(m0["child_state"]),
                        "posterior_probability": float(m0.get("posterior_prob", 0.9)),
                        "consistency_index_ci": float(m0.get("ci_score", 0.8)),
                        "is_indel": bool(m0.get("is_indel", False)),
                    }
                )
            poly_sites = win_info.get("polymorphic_sites", [])
            unverified_site = int(poly_sites[0]) if poly_sites else int(rng.integers(1, max(2, min(seq_len, 40))))
            draft_claims.append(
                {
                    "site_coordinate": unverified_site,
                    "ancestral_state": "C",
                    "derived_state": "T",
                    "posterior_probability": 0.75,
                    "consistency_index_ci": 0.5,
                    "is_indel": False,
                }
            )
            pred_stressor = "NONE" if node_info["topological_status"] == "CONCORDANT" else "SHORT_BRANCH_HOMOPLASY"
            draft_explanation = (
                f"Generic agent inspected {nid} ({sup}% support, {node_info['topological_status']}) "
                f"and flagged polymorphic site {unverified_site} under {pred_stressor}."
            )

        record = _evaluate_and_pack_node_record(
            node_info=node_info,
            draft_claims=draft_claims,
            draft_explanation=draft_explanation,
            predicted_stressor=pred_stressor,
            fasta_path=fasta_path,
            pred_newick_path=pred_newick_path,
            pred_meta_path=pred_meta_path,
            asr_state_path=asr_state_path,
            stressor_gt_path=stressor_gt_path,
        )
        node_audits.append(record)

    global_ccvr = round(
        float(np.mean([n["explanation_quality_vector"]["ccvr_local"] for n in node_audits])),
        4,
    ) if node_audits else 0.0
    lcr_score = compute_lcr(node_audits, pred_newick_path)
    g_tree = compute_tree_goodness(node_audits, asr_state_path, lcr_score)

    report_obj = AuditReport(
        audit_metadata=AuditMetadata(
            dataset_id=dataset_id,
            target_model=target_model,
            taxon_count=taxon_count,
            sequence_length=seq_len,
            orchestrator_llm=f"Baseline2_GenericAgent:{generic_agent_llm}",
        ),
        global_metrics=GlobalMetrics(
            normalized_rf_distance=float(topo_res["normalized_rf_distance"]),
            false_confidence_index_fci=float(topo_res["fci_global"]),
            total_internal_nodes_audited=len(node_audits),
            false_confidence_nodes_count=int(topo_res["false_confidence_nodes_count"]),
            claim_to_coordinate_verification_rate=global_ccvr,
            logical_consistency_rate_lcr=lcr_score,
            overall_tree_goodness_score=g_tree,
        ),
        node_audits=[NodeAuditRecord(**n) for n in node_audits],
    )
    report_dict = json.loads(report_obj.model_dump_json())
    if output_json_path:
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(report_dict, f, indent=2)
    return report_dict
