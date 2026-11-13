"""
Pipeline Step 3: Neuro-Symbolic Diagnostic Agent Control Loop (LangGraph State Machine)
Implements Algorithm 1 and Section 4.1.2 of the manuscript:
  - State 1: Topological and Confidence Triage (audit_tree_topology)
  - State 2: Causal Mutation and Indel Attribution (extract_branch_synapomorphies)
  - State 3: Adversarial Error and Stressor Diagnosis (diagnose_evolutionary_stressor + inspect_alignment_window)
  - State 4: Deterministic Self-Verification Gate (verify_claim_coordinates + Pydantic serialization)
"""
import json
from pathlib import Path
from typing import Any, Callable, TypedDict
import numpy as np
import pandas as pd
from Bio import AlignIO
from langgraph.graph import StateGraph, END

from src.tools.bio_tools import (
    audit_tree_topology,
    extract_branch_synapomorphies,
    diagnose_evolutionary_stressor,
    inspect_alignment_window,
    verify_claim_coordinates,
)
from src.evaluation.metrics import (
    confidence_tier_label,
    compute_feature_precision_recall_f1,
    compute_bsf_local,
    compute_fgs_local,
    compute_bayesian_transition_metrics,
    compute_node_goodness,
    compute_lcr,
    compute_tree_goodness,
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


class DiagnosticAgentState(TypedDict, total=False):
    # Input configuration
    dataset_id: str
    target_model: str
    fasta_path: str
    pred_newick_path: str
    pred_meta_path: str
    true_newick_path: str
    asr_state_path: str
    stressor_gt_path: str
    support_threshold: float
    orchestrator_llm: str
    lm_studio_base_url: str | None
    require_live_llm: bool
    custom_synthesizer: Callable[[dict], tuple[list[dict], str]] | None
    max_retries: int
    output_json_path: str | None

    # State 1 outputs
    topology_summary: dict
    pending_nodes: list[dict]

    # Per-node working memory (States 2-4)
    current_node: dict | None
    current_synapomorphies: list[dict]
    current_mean_ci: float
    current_stressor_diag: dict | None
    current_window_inspection: dict | None
    draft_claims: list[dict]
    draft_explanation: str
    verification_result: dict | None
    retry_count: int

    # Accumulated audited node records & final Pydantic report
    node_audits: list[dict]
    audit_report: dict | None


def classify_diagnostic_regime(in_true: bool, support_pct: float, threshold: float = 85.0) -> str:
    """
    Classifies internal split b_k into one of the four diagnostic regimes (Section 4.1.2, State 1):
      - TRUE_POSITIVE_SPLIT: b_k in B_true and Support(b_k) >= 85%
      - UNDER_CONFIDENT_TRUE_SPLIT: b_k in B_true and Support(b_k) < 85%
      - RECOGNIZED_UNCERTAINTY: b_k not in B_true and Support(b_k) < 85%
      - FALSE_CONFIDENCE_HALLUCINATION: b_k not in B_true and Support(b_k) >= 85%
    """
    if in_true and support_pct >= threshold:
        return "TRUE_POSITIVE_SPLIT"
    elif in_true and support_pct < threshold:
        return "UNDER_CONFIDENT_TRUE_SPLIT"
    elif (not in_true) and support_pct < threshold:
        return "RECOGNIZED_UNCERTAINTY"
    return "FALSE_CONFIDENCE_HALLUCINATION"


def _classify_mutation(m: dict) -> str:
    """Maps a single mutation record from M_true to the Pydantic CausalSequenceEvidence mutation_class."""
    if bool(m.get("is_indel", False)):
        return "INDEL"
    ci_val = float(m.get("ci_score", 1.0))
    if ci_val >= 0.8:
        return "SYNAPOMORPHY"
    elif ci_val >= 0.5:
        return "HOMOPLASY"
    return "HOMOPLASTIC_CONVERGENCE"


def _deterministic_neuro_symbolic_synthesize(
    node_info: dict,
    synapomorphies: list[dict],
    stressor_diag: dict | None,
    target_model: str,
) -> tuple[list[dict], str]:
    """
    Deterministic tool-grounded synthesizer that constructs draft_claims and natural-language
    causal explanation strictly from the verified tool output dictionaries (Table 4.3 & 4.4).
    """
    nid = node_info["node_id"]
    sup = float(node_info["support_pct"])
    status = node_info["topological_status"]
    clade_taxa = node_info["clade_taxa"]
    conf_tier = confidence_tier_label(sup)
    conf_phrase = (
        "definitive high confidence"
        if conf_tier == "Definitive/High"
        else ("moderate confidence" if conf_tier == "Moderate" else "low confidence")
    )

    draft_claims = []
    mut_descriptions = []
    for m in synapomorphies:
        site_pos = int(m["site"])
        s_u = str(m["parent_state"])
        s_v = str(m["child_state"])
        ci_val = float(m.get("ci_score", 1.0))
        is_indel = bool(m.get("is_indel", False))
        draft_claims.append(
            {
                "site_coordinate": site_pos,
                "ancestral_state": s_u,
                "derived_state": s_v,
                "posterior_probability": float(m.get("posterior_prob", 1.0)),
                "consistency_index_ci": max(ci_val, 1e-4),
                "mutation_class": _classify_mutation(m),
                "is_indel": is_indel,
            }
        )
        if is_indel:
            boundary = m.get("indel_boundary") or str(site_pos)
            mut_descriptions.append(f"indel at site {site_pos} (tract {boundary}, CI = {ci_val})")
        else:
            mut_descriptions.append(f"{s_u}{site_pos}{s_v} at site {site_pos} (CI = {ci_val})")

    taxa_preview = ", ".join(clade_taxa[:3]) + ("..." if len(clade_taxa) > 3 else "")
    primary_stressor = stressor_diag["primary_artifact"] if stressor_diag else "NONE"

    if status == "CONCORDANT":
        if mut_descriptions:
            mut_str = ", ".join(mut_descriptions[:3])
            explanation = (
                f"{target_model} resolved concordant clade {nid} ({taxa_preview}) with {sup}% support "
                f"({conf_phrase}); split is grounded by verified synapomorphies: {mut_str}."
            )
        else:
            explanation = (
                f"{target_model} resolved concordant clade {nid} ({taxa_preview}) with {sup}% support "
                f"({conf_phrase}); branch is topologically concordant with the reference tree."
            )
    else:
        ratio = stressor_diag["lba_branch_ratio"] if stressor_diag else 1.0
        gap_pct = stressor_diag["gap_density_pct"] if stressor_diag else 0.0
        mut_str = (
            f" driven by {', '.join(mut_descriptions[:2])}"
            if mut_descriptions
            else ""
        )
        explanation = (
            f"{target_model} grouped {taxa_preview} at node {nid} with {sup}% support "
            f"({conf_phrase}), resulting in {status} under stressor {primary_stressor} "
            f"(LBA ratio = {ratio}, gap density = {gap_pct}%){mut_str}."
        )

    return draft_claims, explanation


def _extract_json_block(text: str) -> dict:
    """Extracts a JSON object from raw LLM text, stripping Qwen3 <think>...</think> blocks and markdown fences."""
    import re

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return json.loads(candidate)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    return json.loads(cleaned)


def _llm_or_fallback_synthesize(state: DiagnosticAgentState) -> tuple[list[dict], str]:
    """
    Executes Orchestrator synthesis (Table 4.3):
      1. If a custom_synthesizer is provided (e.g., for testing hallucination injection/retries), invokes it
         on the first attempt (retry_count == 0), and enforces strict tool-dictionary grounding on retry.
      2. If lm_studio_base_url is set, queries the local Qwen 2.5 LLM via OpenAI-compatible API
         (temperature=0.0, top_p=1.0, seed=42). If attempt 0 had mismatched claims, re-prompts the LLM
         with the exact mismatched_claims rejection feedback from State 4.
      3. Otherwise (or if retries are exhausted) executes deterministic neuro-symbolic synthesis
         strictly over the verified tool dictionary.
    """
    node_info = state["current_node"]
    synapomorphies = state.get("current_synapomorphies", [])
    stressor_diag = state.get("current_stressor_diag")
    target_model = state.get("target_model", "Fusang")
    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", 3))

    custom_fn = state.get("custom_synthesizer")
    if custom_fn is not None:
        if retry_count == 0:
            return custom_fn(state)
        return _deterministic_neuro_symbolic_synthesize(
            node_info, synapomorphies, stressor_diag, target_model
        )

    base_url = state.get("lm_studio_base_url")
    if base_url and retry_count < max_retries:
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                base_url=base_url,
                api_key="lm-studio",
                model=state.get("orchestrator_llm", "qwen/qwen3-14b"),
                temperature=0.0,
                top_p=1.0,
                seed=42,
                timeout=60.0,
            )
            conf_tier = confidence_tier_label(float(node_info["support_pct"]))
            conf_phrase = (
                "definitive high confidence"
                if conf_tier == "Definitive/High"
                else ("moderate confidence" if conf_tier == "Moderate" else "low confidence")
            )
            retry_feedback = ""
            ver_prev = state.get("verification_result")
            if retry_count > 0 and ver_prev and ver_prev.get("mismatched_claims"):
                retry_feedback = (
                    "\nWARNING — PREVIOUS DRAFT REJECTED BY STATE 4 VERIFICATION GATE:\n"
                    f"Mismatched/Hallucinated claims: {json.dumps(ver_prev['mismatched_claims'])}\n"
                    "You MUST cite ONLY exact (site, parent_state, child_state) tuples present in the "
                    "Synapomorphies list below.\n"
                )

            prompt = (
                "You are the Lead Phylogenetic Diagnostic Orchestrator (Phase 1 Neuro-Symbolic Auditor).\n"
                "Given the deterministic bioinformatics tool outputs below, return ONLY a valid JSON object "
                "with exactly two keys:\n"
                "1. \"draft_claims\": list of objects, one for each entry in Synapomorphies, each having keys:\n"
                "   {\"site_coordinate\": <int site>, \"ancestral_state\": \"<parent_state>\", "
                "\"derived_state\": \"<child_state>\", \"posterior_probability\": <float posterior_prob>, "
                "\"consistency_index_ci\": <float ci_score>, \"is_indel\": <bool is_indel>}\n"
                "2. \"explanation\": a 1-2 sentence biological causal explanation for the split at "
                f"node {node_info['node_id']} citing the exact support percentage ({node_info['support_pct']}% "
                f"support, {conf_phrase}), topological status ({node_info['topological_status']}), "
                "and verified site coordinates from Synapomorphies (do not invent any unverified site numbers).\n"
                f"{retry_feedback}"
                f"Target Model: {target_model}\n"
                f"Node Info: {json.dumps(node_info)}\n"
                f"Synapomorphies (Tool 2 Output): {json.dumps(synapomorphies)}\n"
                f"Stressor Diagnosis (Tool 3 Output): {json.dumps(stressor_diag)}\n"
                "/no_think"
            )
            resp = llm.invoke(prompt)
            content_str = resp.content if isinstance(resp.content, str) else str(resp.content)
            parsed = _extract_json_block(content_str)
            if "draft_claims" in parsed and "explanation" in parsed:
                return list(parsed["draft_claims"]), str(parsed["explanation"])
        except Exception as exc:
            if state.get("require_live_llm"):
                raise RuntimeError(
                    f"Failed to query local LM Studio server at {base_url}: {exc}"
                ) from exc

    return _deterministic_neuro_symbolic_synthesize(
        node_info, synapomorphies, stressor_diag, target_model
    )


# -----------------------------------------------------------------------------
# LangGraph State Nodes (Algorithm 1 & Flowchart States 1 - 4)
# -----------------------------------------------------------------------------

def state_1_topological_triage(state: DiagnosticAgentState) -> dict:
    """
    State 1 — Topological and Confidence Triage:
    Invokes Tool 1 (audit_tree_topology) to compute global RF_norm and FCI, and classifies
    all internal splits b_k in B_pred into their diagnostic regimes.
    """
    threshold = float(state.get("support_threshold", 85.0))
    topo_res = audit_tree_topology(
        pred_newick_path=state["pred_newick_path"],
        true_newick_path=state["true_newick_path"],
        pred_meta_path=state["pred_meta_path"],
        support_threshold=threshold,
    )

    pending = []
    for node_rec in topo_res["node_bipartitions"]:
        in_true = node_rec["topological_status"] == "CONCORDANT"
        regime = classify_diagnostic_regime(in_true, float(node_rec["support_pct"]), threshold)
        pending.append({**node_rec, "diagnostic_regime": regime})

    # Sort deterministically by post-order node index (N1, N2, ..., Nk)
    pending.sort(key=lambda r: int(r["node_id"][1:]) if r["node_id"][1:].isdigit() else 9999)

    return {
        "topology_summary": topo_res,
        "pending_nodes": pending,
        "node_audits": [],
    }


def state_2_ancestral_extraction(state: DiagnosticAgentState) -> dict:
    """
    State 2 — Causal Mutation and Indel Attribution:
    Pops the next internal node b_k = (u -> v) and invokes Tool 2 (extract_branch_synapomorphies)
    to retrieve verified ASR substitutions and Fitch binary gap indel synapomorphies.
    """
    pending = list(state.get("pending_nodes", []))
    current_node = pending.pop(0)

    # Use true_matching_node_id when concordant so node-ID shifts do not miss ground-truth branch mutations
    query_child = current_node.get("true_matching_node_id") or current_node["node_id"]
    query_parent = current_node["parent_node_id"]

    synapomorphies = extract_branch_synapomorphies(
        fasta_path=state["fasta_path"],
        asr_state_path=state["asr_state_path"],
        parent_node_id=query_parent,
        child_node_id=query_child,
    )

    if synapomorphies:
        mean_ci = float(np.mean([float(m.get("ci_score", 1.0)) for m in synapomorphies]))
    else:
        mean_ci = 1.0 if current_node["topological_status"] == "CONCORDANT" else 0.0

    return {
        "pending_nodes": pending,
        "current_node": current_node,
        "current_synapomorphies": synapomorphies,
        "current_mean_ci": round(mean_ci, 4),
        "current_stressor_diag": None,
        "current_window_inspection": None,
        "retry_count": 0,
    }


def route_after_state_2(state: DiagnosticAgentState) -> str:
    """
    Flowchart Decision Diamond 1 & Algorithm 1 Line 7:
      if b_k not in B_true or mean_CI(synapomorphies) < 0.5 then
          route to State 3 (diagnose_evolutionary_stressor)
      else
          route directly to State 4 (Synthesize & Verify)
    """
    current_node = state["current_node"]
    mean_ci = float(state.get("current_mean_ci", 1.0))
    is_discordant = current_node["topological_status"] != "CONCORDANT"

    if is_discordant or mean_ci < 0.5:
        return "state_3_stressor_interrogation"
    return "state_4_synthesize_and_verify"


def state_3_stressor_interrogation(state: DiagnosticAgentState) -> dict:
    """
    State 3 — Adversarial Error and Stressor Diagnosis:
    Invokes Tool 3 (diagnose_evolutionary_stressor) and Tool 4 (inspect_alignment_window)
    to isolate LBA, Indel/Gap-Masking, Recombination, or Short-Branch Homoplasy.
    """
    current_node = state["current_node"]
    synapomorphies = state.get("current_synapomorphies", [])

    stressor_diag = diagnose_evolutionary_stressor(
        fasta_path=state["fasta_path"],
        pred_newick_path=state["pred_newick_path"],
        node_id=current_node["node_id"],
        stressor_gt_path=state["stressor_gt_path"],
    )

    # Also inspect local alignment window around the top causal/homoplastic site (or sites 1..30)
    if synapomorphies:
        center_site = int(synapomorphies[0]["site"])
        start_pos = max(1, center_site - 10)
        end_pos = center_site + 10
    else:
        start_pos, end_pos = 1, 30

    alignment = AlignIO.read(state["fasta_path"], "fasta")
    seq_len = alignment.get_alignment_length()
    end_pos = min(end_pos, seq_len)

    window_info = inspect_alignment_window(
        fasta_path=state["fasta_path"],
        taxa_subset=current_node["clade_taxa"],
        start_pos=start_pos,
        end_pos=end_pos,
    )

    return {
        "current_stressor_diag": stressor_diag,
        "current_window_inspection": window_info,
    }


def state_4_synthesize_and_verify(state: DiagnosticAgentState) -> dict:
    """
    State 4 — Deterministic Self-Verification Gate:
    Synthesizes draft biological claims and causal explanation, then passes draft_claims
    through Tool 5 (verify_claim_coordinates). If all_verified == True (or max_retries reached),
    computes node-level evaluation metrics and appends the verified NodeAuditRecord.
    """
    draft_claims, draft_explanation = _llm_or_fallback_synthesize(state)

    ver_res = verify_claim_coordinates(
        draft_claims=draft_claims,
        fasta_path=state["fasta_path"],
        asr_state_path=state["asr_state_path"],
    )

    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", 3))

    if (not ver_res["all_verified"]) and retry_count < max_retries:
        # Reject hallucinated draft and trigger re-generation constrained to tool dictionary
        return {
            "draft_claims": draft_claims,
            "draft_explanation": draft_explanation,
            "verification_result": ver_res,
            "retry_count": retry_count + 1,
        }

    # Commit verified node audit record
    current_node = state["current_node"]
    synapomorphies = state.get("current_synapomorphies", [])
    stressor_diag = state.get("current_stressor_diag")

    # If State 3 was skipped (concordant high-CI split), retrieve baseline stressor metrics
    if stressor_diag is None:
        with open(state["stressor_gt_path"]) as f:
            a_true = json.load(f)
        stressor_diag = {
            "mean_ci": float(state.get("current_mean_ci", a_true["mean_alignment_ci"])),
            "mean_ri": float(a_true["mean_alignment_ri"]),
            "lba_branch_ratio": float(a_true["global_rho_lba"]),
            "gap_density_pct": float(a_true["mean_gap_density_pct"]),
            "recombination_window_overlap": False,
            "primary_artifact": "NONE",
        }

    with open(state["pred_meta_path"]) as f:
        pred_meta = json.load(f)
    nid = current_node["node_id"]
    branch_len = float(pred_meta["nodes"].get(nid, {}).get("branch_length", 0.05))

    # Compute node-level multi-dimensional evaluation metrics (Section 4.3.2.B & C)
    cited_sites = {int(c.get("site_coordinate", c.get("site", -1))) for c in draft_claims}
    true_sites = {int(m["site"]) for m in synapomorphies}
    _, _, f1_score = compute_feature_precision_recall_f1(cited_sites, true_sites)

    ccvr_local = float(ver_res["ccvr_score"])
    bsf_local = compute_bsf_local(float(current_node["support_pct"]), draft_explanation)
    fgs_local = compute_fgs_local(
        explanation_text=draft_explanation,
        node_id=nid,
        support_pct=float(current_node["support_pct"]),
        synapomorphies=synapomorphies,
        stressor_diag=stressor_diag,
    )
    bayes_metrics = compute_bayesian_transition_metrics(
        cited_mutations=draft_claims,
        true_branch_mutations=synapomorphies,
        branch_length=branch_len,
        clade_taxa=current_node["clade_taxa"],
        fasta_path=state["fasta_path"],
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
        site_coord = int(c.get("site_coordinate", c.get("site", 1)))
        s_u = str(c.get("ancestral_state", c.get("parent_state", "A")))
        s_v = str(c.get("derived_state", c.get("child_state", "C")))
        p_prob = float(c.get("posterior_probability", c.get("posterior_prob", 1.0)))
        ci_val = max(float(c.get("consistency_index_ci", c.get("ci_score", 1.0))), 1e-4)
        m_class = str(c.get("mutation_class", _classify_mutation(c)))
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

    node_record = NodeAuditRecord(
        node_id=nid,
        predicted_clade_taxa=current_node["clade_taxa"],
        model_support_pct=float(current_node["support_pct"]),
        topological_status=current_node["topological_status"],
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
            primary_stressor=stressor_diag["primary_artifact"],
            lba_branch_ratio=float(stressor_diag["lba_branch_ratio"]),
            local_gap_density_pct=float(stressor_diag["gap_density_pct"]),
            agent_causal_explanation=draft_explanation,
        ),
    ).model_dump()

    updated_audits = list(state.get("node_audits", [])) + [node_record]
    return {
        "draft_claims": draft_claims,
        "draft_explanation": draft_explanation,
        "verification_result": ver_res,
        "node_audits": updated_audits,
    }


def route_after_state_4(state: DiagnosticAgentState) -> str:
    """
    Flowchart Decision Diamonds 2 & 3 (Algorithm 1 Lines 13-16):
      1. If verification.all_verified == False and retry_count <= max_retries:
         loop back to state_4_synthesize_and_verify (Hallucination Blocked -> Re-Prompt).
      2. Else if pending_nodes is non-empty:
         loop back to state_2_ancestral_extraction (Next Internal Node N1..Nk).
      3. Else:
         proceed to finalize_audit_report (All N1..Nk Nodes Audited).
    """
    ver_res = state.get("verification_result") or {"all_verified": True}
    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", 3))

    if (not ver_res["all_verified"]) and retry_count <= max_retries:
        return "state_4_synthesize_and_verify"

    if state.get("pending_nodes"):
        return "state_2_ancestral_extraction"

    return "finalize_audit_report"


def finalize_audit_report(state: DiagnosticAgentState) -> dict:
    """
    Pipeline Step 4 Bridge:
    Reconciles all node-level explanation scores into GlobalMetrics (CCVR, LCR, G_tree)
    and serializes a strict Pydantic-validated AuditReport JSON.
    """
    alignment = AlignIO.read(state["fasta_path"], "fasta")
    taxon_count = len(alignment)
    seq_len = alignment.get_alignment_length()

    topo = state["topology_summary"]
    node_audits = state.get("node_audits", [])

    if node_audits:
        global_ccvr = round(
            float(
                np.mean(
                    [n["explanation_quality_vector"]["ccvr_local"] for n in node_audits]
                )
            ),
            4,
        )
    else:
        global_ccvr = 1.0

    lcr_score = compute_lcr(node_audits, state["pred_newick_path"])
    g_tree = compute_tree_goodness(node_audits, state["asr_state_path"], lcr_score)

    report_obj = AuditReport(
        audit_metadata=AuditMetadata(
            dataset_id=state["dataset_id"],
            target_model=state["target_model"],
            taxon_count=taxon_count,
            sequence_length=seq_len,
            orchestrator_llm=state.get("orchestrator_llm", "Qwen-2.5-32B-Instruct-Q4_K_M"),
        ),
        global_metrics=GlobalMetrics(
            normalized_rf_distance=float(topo["normalized_rf_distance"]),
            false_confidence_index_fci=float(topo["fci_global"]),
            total_internal_nodes_audited=len(node_audits),
            false_confidence_nodes_count=int(topo["false_confidence_nodes_count"]),
            claim_to_coordinate_verification_rate=global_ccvr,
            logical_consistency_rate_lcr=lcr_score,
            overall_tree_goodness_score=g_tree,
        ),
        node_audits=[NodeAuditRecord(**n) for n in node_audits],
    )

    report_dict = json.loads(report_obj.model_dump_json())
    out_path = state.get("output_json_path")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report_dict, f, indent=2)

    return {"audit_report": report_dict}


def build_diagnostic_agent_graph():
    """
    Constructs and compiles the 4-state cyclic LangGraph StateMachine (Algorithm 1).
    """
    workflow = StateGraph(DiagnosticAgentState)

    workflow.add_node("state_1_topological_triage", state_1_topological_triage)
    workflow.add_node("state_2_ancestral_extraction", state_2_ancestral_extraction)
    workflow.add_node("state_3_stressor_interrogation", state_3_stressor_interrogation)
    workflow.add_node("state_4_synthesize_and_verify", state_4_synthesize_and_verify)
    workflow.add_node("finalize_audit_report", finalize_audit_report)

    workflow.set_entry_point("state_1_topological_triage")
    workflow.add_edge("state_1_topological_triage", "state_2_ancestral_extraction")

    workflow.add_conditional_edges(
        "state_2_ancestral_extraction",
        route_after_state_2,
        {
            "state_3_stressor_interrogation": "state_3_stressor_interrogation",
            "state_4_synthesize_and_verify": "state_4_synthesize_and_verify",
        },
    )

    workflow.add_edge("state_3_stressor_interrogation", "state_4_synthesize_and_verify")

    workflow.add_conditional_edges(
        "state_4_synthesize_and_verify",
        route_after_state_4,
        {
            "state_4_synthesize_and_verify": "state_4_synthesize_and_verify",
            "state_2_ancestral_extraction": "state_2_ancestral_extraction",
            "finalize_audit_report": "finalize_audit_report",
        },
    )

    workflow.add_edge("finalize_audit_report", END)
    return workflow.compile()


def run_diagnostic_audit(
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
    orchestrator_llm: str = "Qwen-2.5-14B-Instruct-Q4_K_M",
    lm_studio_base_url: str | None = None,
    require_live_llm: bool = False,
    custom_synthesizer: Callable[[dict], tuple[list[dict], str]] | None = None,
    max_retries: int = 3,
) -> dict:
    """
    Executes the compiled LangGraph Diagnostic Agent across all internal branches of T_pred
    and returns the Pydantic-validated AuditReport dictionary.
    """
    app = build_diagnostic_agent_graph()
    initial_state: DiagnosticAgentState = {
        "dataset_id": dataset_id,
        "target_model": target_model,
        "fasta_path": fasta_path,
        "pred_newick_path": pred_newick_path,
        "pred_meta_path": pred_meta_path,
        "true_newick_path": true_newick_path,
        "asr_state_path": asr_state_path,
        "stressor_gt_path": stressor_gt_path,
        "support_threshold": support_threshold,
        "orchestrator_llm": orchestrator_llm,
        "lm_studio_base_url": lm_studio_base_url,
        "require_live_llm": require_live_llm,
        "custom_synthesizer": custom_synthesizer,
        "max_retries": max_retries,
        "output_json_path": output_json_path,
    }
    # Increase recursion_limit so trees with up to 200 taxa (~400+ state transitions) complete cleanly
    final_state = app.invoke(initial_state, config={"recursion_limit": 2000})
    return final_state["audit_report"]
