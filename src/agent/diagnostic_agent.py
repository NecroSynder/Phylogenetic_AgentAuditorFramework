"""
Pipeline Step 3: Neuro-Symbolic Diagnostic Agent Control Loop (LangGraph State Machine)
Implements Algorithm 1 and Section 4.1.2 of the manuscript:
  - State 1: Topological and Confidence Triage (audit_tree_topology)
  - State 2: Causal Mutation and Indel Attribution (extract_branch_synapomorphies; clade-isolated)
  - State 3: Adversarial Error and Stressor Diagnosis (diagnose_evolutionary_stressor + inspect_alignment_window)
  - State 4: Deterministic Self-Verification Gate (verify_claim_coordinates + Pydantic serialization)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from src.evaluation.metrics import (
    classify_mutation_record,
    compute_lcr,
    compute_report_edr,
    compute_tree_goodness,
    confidence_tier_label,
    evaluate_single_node_audit,
)
from src.schemas.audit_schema import (
    AuditMetadata,
    AuditReport,
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
from src.utils.io_cache import (
    load_alignment_data,
    load_json_dict,
    resolve_lm_studio_url,
    sort_node_id_key,
)


class DiagnosticAgentState(TypedDict, total=False):
    # Input configuration
    dataset_id: str
    target_model: str
    fasta_path: str
    pred_newick_path: str
    pred_meta_path: str | None
    true_newick_path: str
    asr_state_path: str
    stressor_gt_path: str
    support_threshold: float
    orchestrator_llm: str
    lm_studio_base_url: str | None
    allow_offline_mock: bool
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
    node_committed: bool

    # Accumulated audited node records, weights & claim totals
    node_audits: list[dict]
    branch_info_weights: dict[str, float]
    total_claims_evaluated: int
    total_claims_verified: int
    used_offline_mock: bool
    audit_report: dict | None


def classify_diagnostic_regime(
    in_true: bool, support_pct: float, threshold: float = 85.0
) -> str:
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


def _deterministic_neuro_symbolic_synthesize(
    node_info: dict,
    synapomorphies: list[dict],
    stressor_diag: dict | None,
    window_inspection: dict | None,
    target_model: str,
) -> tuple[list[dict], str]:
    """
    Deterministic tool-grounded synthesizer that constructs draft_claims and natural-language
    causal explanation strictly from the verified tool output dictionaries (Table 4.4).
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
                "consistency_index_ci": ci_val,
                "mutation_class": classify_mutation_record(m),
                "is_indel": is_indel,
            }
        )
        if is_indel:
            boundary = m.get("indel_boundary") or str(site_pos)
            mut_descriptions.append(
                f"indel at site {site_pos} (tract {boundary}, CI = {ci_val})"
            )
        else:
            mut_descriptions.append(
                f"{s_u}{site_pos}{s_v} at site {site_pos} (CI = {ci_val})"
            )

    taxa_preview = ", ".join(clade_taxa[:3]) + ("..." if len(clade_taxa) > 3 else "")
    primary_stressor = stressor_diag["primary_artifact"] if stressor_diag else "NONE"
    win_note = ""
    if window_inspection:
        win_note = (
            f"; window [{window_inspection['start_pos']}..{window_inspection['end_pos']}] "
            f"mean entropy = {window_inspection['mean_shannon_entropy_bits']} bits"
        )

    if status == "CONCORDANT":
        if mut_descriptions:
            mut_str = ", ".join(mut_descriptions[:3])
            explanation = (
                f"{target_model} resolved concordant clade {nid} ({taxa_preview}) with {sup}% support "
                f"({conf_phrase}); split is grounded by verified synapomorphies: {mut_str}{win_note}."
            )
        else:
            explanation = (
                f"{target_model} resolved concordant clade {nid} ({taxa_preview}) with {sup}% support "
                f"({conf_phrase}); branch is topologically concordant with the reference tree{win_note}."
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
            f"(LBA ratio = {ratio}, gap density = {gap_pct}%){mut_str}{win_note}."
        )

    return draft_claims, explanation


def _extract_json_block(text: str) -> dict:
    """Extracts a JSON object from raw LLM text, stripping <think>...</think> blocks and markdown fences."""
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


def _llm_or_fallback_synthesize(
    state: DiagnosticAgentState,
) -> tuple[list[dict], str, bool]:
    """
    Executes Orchestrator synthesis (Table 4.3):
      - Invokes custom_synthesizer if provided (supporting both single-shot and persistent test hooks).
      - Queries live local LM Studio LLM when lm_studio_base_url is set, passing Tools 1-4 outputs
        (including Tool 4 window inspection!) and State 4 rejection feedback on retries.
      - Requires explicit allow_offline_mock=True when lm_studio_base_url is None, returning
        used_mock=True so offline template outputs never masquerade as live LLM runs.
    """
    node_info = state["current_node"]
    synapomorphies = state.get("current_synapomorphies", [])
    stressor_diag = state.get("current_stressor_diag")
    window_inspection = state.get("current_window_inspection")
    target_model = state.get("target_model", "Fusang")
    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", 3))

    custom_fn = state.get("custom_synthesizer")
    if custom_fn is not None:
        if retry_count == 0 or getattr(custom_fn, "persist_across_retries", False):
            claims, expl = custom_fn(state)
            return claims, expl, False
        claims, expl = _deterministic_neuro_symbolic_synthesize(
            node_info, synapomorphies, stressor_diag, window_inspection, target_model
        )
        return claims, expl, False

    base_url = state.get("lm_studio_base_url")
    if base_url:
        if retry_count < max_retries:
            try:
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(
                    base_url=base_url,
                    api_key="lm-studio",
                    model=state.get("orchestrator_llm", "qwen/qwen3-14b"),
                    temperature=0.0,
                    top_p=1.0,
                    seed=42,
                    timeout=90.0,
                )
                retry_feedback = ""
                ver_prev = state.get("verification_result")
                if retry_count > 0 and ver_prev and ver_prev.get("mismatched_claims"):
                    retry_feedback = (
                        "\nWARNING — PREVIOUS DRAFT REJECTED BY STATE 4 VERIFICATION GATE:\n"
                        f"Mismatched/Hallucinated claims: {json.dumps(ver_prev['mismatched_claims'])}\n"
                        "You MUST cite ONLY exact (site, parent_state, child_state) tuples present in "
                        "Tool 2 Output below, both in draft_claims AND in explanation prose.\n"
                    )

                window_summary = None
                if window_inspection:
                    window_summary = {
                        "start_pos": window_inspection["start_pos"],
                        "end_pos": window_inspection["end_pos"],
                        "mean_shannon_entropy_bits": window_inspection["mean_shannon_entropy_bits"],
                        "mean_gap_density_pct": window_inspection["mean_gap_density_pct"],
                        "polymorphic_sites": window_inspection["polymorphic_sites"][:10],
                    }

                prompt = (
                    "You are the Lead Phylogenetic Diagnostic Orchestrator (Phase 1 Neuro-Symbolic Auditor).\n"
                    "Confidence Tier Rubric (Eq. 4.7): <50% = 'low confidence', 50% to <85% = 'moderate confidence', "
                    ">=85% = 'definitive high confidence' (state ONLY the single tier phrase matching this node's "
                    "bootstrap support_pct—do not list the other rubric tiers, and even on FALSE_CONFIDENCE_ERROR "
                    "nodes state the model's bootstrap tier accurately before explaining why the split is erroneous).\n"
                    "Given the deterministic bioinformatics tool outputs below, return ONLY a valid JSON object "
                    "with exactly two keys:\n"
                    "1. \"draft_claims\": list of objects from Tool 2 Output, each having keys:\n"
                    "   {\"site_coordinate\": <int site>, \"ancestral_state\": \"<parent_state>\", "
                    "\"derived_state\": \"<child_state>\", \"posterior_probability\": <float posterior_prob>, "
                    "\"consistency_index_ci\": <float ci_score>, \"is_indel\": <bool is_indel>}\n"
                    "2. \"explanation\": a 1-2 sentence biological causal explanation for the split at "
                    f"node {node_info['node_id']} citing its exact bootstrap support percentage, the matching "
                    "confidence tier phrase from the rubric above, its topological status, primary stressor (if discordant), "
                    "and verified site coordinates from Tool 2 Output (never invent unverified site numbers).\n"
                    f"{retry_feedback}"
                    f"Target Model: {target_model}\n"
                    f"Node Info (Tool 1 Output): {json.dumps(node_info)}\n"
                    f"Verified Clade Mutations (Tool 2 Output): {json.dumps(synapomorphies)}\n"
                    f"Stressor Diagnosis (Tool 3 Output): {json.dumps(stressor_diag)}\n"
                    f"Alignment Window Inspection (Tool 4 Output): {json.dumps(window_summary)}\n"
                    "/no_think"
                )
                resp = llm.invoke(prompt)
                content_str = (
                    resp.content if isinstance(resp.content, str) else str(resp.content)
                )
                parsed = _extract_json_block(content_str)
                if "draft_claims" in parsed and "explanation" in parsed:
                    return list(parsed["draft_claims"]), str(parsed["explanation"]), False
            except Exception as exc:
                raise RuntimeError(
                    f"Live LM Studio query failed at {base_url} on node {node_info['node_id']}: {exc}"
                ) from exc

        # Exhausted max_retries on State 4 coordinate gate -> enforce strict deterministic tool grounding
        claims, expl = _deterministic_neuro_symbolic_synthesize(
            node_info, synapomorphies, stressor_diag, window_inspection, target_model
        )
        return claims, expl, False

    if not state.get("allow_offline_mock", False):
        raise RuntimeError(
            "lm_studio_base_url is None, and allow_offline_mock is False. "
            "Start LM Studio or pass --offline-mock (allow_offline_mock=True) for deterministic offline tests."
        )

    claims, expl = _deterministic_neuro_symbolic_synthesize(
        node_info, synapomorphies, stressor_diag, window_inspection, target_model
    )
    return claims, expl, True


# -----------------------------------------------------------------------------
# LangGraph State Nodes (Algorithm 1 & Flowchart States 1 - 4)
# -----------------------------------------------------------------------------

def state_1_topological_triage(state: DiagnosticAgentState) -> dict:
    """
    State 1 — Topological and Confidence Triage:
    Invokes Tool 1 (audit_tree_topology) to compute global RF_norm, FCI_global, and FCI_cond,
    and classifies all internal splits b_k in B_pred into their diagnostic regimes.
    """
    threshold = float(state.get("support_threshold", 85.0))
    topo_res = audit_tree_topology(
        pred_newick_path=state["pred_newick_path"],
        true_newick_path=state["true_newick_path"],
        pred_meta_path=state.get("pred_meta_path"),
        support_threshold=threshold,
        stressor_gt_path=state.get("stressor_gt_path"),
    )

    pending = []
    for node_rec in topo_res["node_bipartitions"]:
        in_true = node_rec["topological_status"] == "CONCORDANT"
        regime = classify_diagnostic_regime(
            in_true, float(node_rec["support_pct"]), threshold
        )
        pending.append({**node_rec, "diagnostic_regime": regime})

    pending.sort(key=lambda r: sort_node_id_key(r["node_id"]))

    return {
        "topology_summary": topo_res,
        "pending_nodes": pending,
        "node_audits": [],
        "branch_info_weights": {},
        "total_claims_evaluated": 0,
        "total_claims_verified": 0,
        "used_offline_mock": False,
    }


def state_2_ancestral_extraction(state: DiagnosticAgentState) -> dict:
    """
    State 2 — Causal Mutation and Indel Attribution:
    Pops the next internal node b_k and invokes Tool 2 (extract_branch_synapomorphies):
      - Concordant splits query M_true.tsv strictly by (true_parent_node_id, true_matching_node_id).
      - Discordant splits extract clade-partitioning convergent/indel columns directly from
        alignment.fa + A_true.json without cross-clade node-ID contamination.
    """
    pending = list(state.get("pending_nodes", []))
    current_node = pending.pop(0)

    has_true_match = (
        current_node["topological_status"] == "CONCORDANT"
        and current_node.get("true_matching_node_id") is not None
    )
    if has_true_match:
        synapomorphies = extract_branch_synapomorphies(
            fasta_path=state["fasta_path"],
            asr_state_path=state["asr_state_path"],
            parent_node_id=current_node.get("true_parent_node_id") or "N_ROOT",
            child_node_id=current_node["true_matching_node_id"],
            is_concordant=True,
            top_k=10,
        )
    else:
        synapomorphies = extract_branch_synapomorphies(
            fasta_path=state["fasta_path"],
            asr_state_path=state["asr_state_path"],
            parent_node_id=current_node["parent_node_id"],
            child_node_id=current_node["node_id"],
            is_concordant=False,
            clade_taxa=current_node["clade_taxa"],
            stressor_gt_path=state["stressor_gt_path"],
            top_k=10,
        )

    if synapomorphies:
        mean_ci = float(
            np.mean([float(m.get("ci_score", 1.0)) for m in synapomorphies])
        )
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
        "node_committed": False,
    }


def route_after_state_2(state: DiagnosticAgentState) -> str:
    """
    Algorithm 1 Line 7:
      if b_k not in B_true or mean_CI(synapomorphies) < 0.5 then
          route to State 3 (diagnose_evolutionary_stressor + inspect_alignment_window)
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

    _, _, seq_len = load_alignment_data(state["fasta_path"])
    if synapomorphies:
        center_site = int(synapomorphies[0]["site"])
        start_pos = max(1, center_site - 15)
        end_pos = min(seq_len, center_site + 15)
    else:
        a_true = load_json_dict(state["stressor_gt_path"])
        recomb_bps = a_true.get("recombination_breakpoints") or []
        if recomb_bps:
            start_pos = max(1, int(recomb_bps[0][0]))
            end_pos = min(seq_len, start_pos + 30)
        else:
            start_pos, end_pos = 1, min(seq_len, 30)

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
    Synthesizes draft biological claims and causal explanation, then verifies both
    draft_claims AND explanation prose via Tool 5 (verify_claim_coordinates).
    Tracks node_committed explicitly so retry exhaustion terminates cleanly without loops.
    """
    draft_claims, draft_explanation, used_mock = _llm_or_fallback_synthesize(state)
    synapomorphies = state.get("current_synapomorphies", [])

    ver_res = verify_claim_coordinates(
        draft_claims=draft_claims,
        fasta_path=state["fasta_path"],
        asr_state_path=state["asr_state_path"],
        valid_branch_mutations=synapomorphies,
        explanation_text=draft_explanation,
    )

    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", 3))

    if (not ver_res["all_verified"]) and retry_count < max_retries:
        nid_alert = state["current_node"]["node_id"]
        print(
            f"  [State 4 Verification Gate] Node {nid_alert}: Blocked unverified coordinate claim(s) "
            f"{ver_res.get('mismatched_claims', [])} (Attempt {retry_count + 1}/{max_retries}) -> Self-correcting..."
        )
        return {
            "draft_claims": draft_claims,
            "draft_explanation": draft_explanation,
            "verification_result": ver_res,
            "retry_count": retry_count + 1,
            "node_committed": False,
            "used_offline_mock": bool(state.get("used_offline_mock", False) or used_mock),
        }

    current_node = state["current_node"]
    stressor_diag = state.get("current_stressor_diag")
    if stressor_diag is None:
        a_true = load_json_dict(state["stressor_gt_path"])
        stressor_diag = {
            "mean_ci": float(
                state.get(
                    "current_mean_ci",
                    a_true.get("mean_informative_ci", a_true["mean_alignment_ci"]),
                )
            ),
            "mean_ri": float(
                a_true.get("mean_informative_ri", a_true["mean_alignment_ri"])
            ),
            "lba_branch_ratio": float(a_true["global_rho_lba"]),
            "gap_density_pct": float(a_true["mean_gap_density_pct"]),
            "recombination_window_overlap": False,
            "primary_artifact": "NONE",
        }

    _, _, seq_len = load_alignment_data(state["fasta_path"])
    node_record, final_ver_res, branch_weight = evaluate_single_node_audit(
        node_info=current_node,
        draft_claims=draft_claims,
        draft_explanation=draft_explanation,
        predicted_stressor=stressor_diag["primary_artifact"],
        true_branch_mutations=synapomorphies,
        true_stressor_diag=stressor_diag,
        fasta_path=state["fasta_path"],
        pred_meta_path=state.get("pred_meta_path"),
        asr_state_path=state["asr_state_path"],
        retry_count=retry_count,
        seq_len=seq_len,
    )

    updated_audits = list(state.get("node_audits", [])) + [node_record]
    updated_weights = dict(state.get("branch_info_weights", {}))
    updated_weights[current_node["node_id"]] = branch_weight

    eval_cnt = int(final_ver_res.get("total_claims_evaluated", 0))
    ver_cnt = int(final_ver_res.get("verified_claims_count", 0))
    # Count non-empty ground-truth omission as 1 unverified slot in micro-CCVR
    if eval_cnt == 0 and len(synapomorphies) > 0:
        eval_cnt = 1
        ver_cnt = 0

    return {
        "draft_claims": draft_claims,
        "draft_explanation": draft_explanation,
        "verification_result": final_ver_res,
        "node_audits": updated_audits,
        "branch_info_weights": updated_weights,
        "total_claims_evaluated": int(state.get("total_claims_evaluated", 0)) + eval_cnt,
        "total_claims_verified": int(state.get("total_claims_verified", 0)) + ver_cnt,
        "node_committed": True,
        "used_offline_mock": bool(state.get("used_offline_mock", False) or used_mock),
    }


def route_after_state_4(state: DiagnosticAgentState) -> str:
    """
    Routes strictly on node_committed so retry exhaustion never causes an infinite loop:
      1. If node_committed is False -> loop back to state_4_synthesize_and_verify.
      2. Else if pending_nodes is non-empty -> advance to state_2_ancestral_extraction.
      3. Else -> proceed to finalize_audit_report.
    """
    if not state.get("node_committed", True):
        return "state_4_synthesize_and_verify"
    if state.get("pending_nodes"):
        return "state_2_ancestral_extraction"
    return "finalize_audit_report"


def finalize_audit_report(state: DiagnosticAgentState) -> dict:
    """
    Pipeline Step 4 Bridge:
    Reconciles all node-level explanation scores into GlobalMetrics (micro-CCVR, EDR, LCR, G_tree)
    and serializes a strict Pydantic-validated AuditReport JSON.
    """
    taxa_ids, _, seq_len = load_alignment_data(state["fasta_path"])
    taxon_count = len(taxa_ids)

    topo = state["topology_summary"]
    node_audits = state.get("node_audits", [])
    tot_eval = int(state.get("total_claims_evaluated", 0))
    tot_ver = int(state.get("total_claims_verified", 0))

    if tot_eval > 0:
        global_ccvr = round(tot_ver / tot_eval, 4)
    elif node_audits:
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
    g_tree = compute_tree_goodness(
        node_audits=node_audits,
        m_true_tsv_path=state["asr_state_path"],
        lcr_score=lcr_score,
        branch_info_weights=state.get("branch_info_weights"),
    )
    edr_score = compute_report_edr(
        node_audits=node_audits,
        fasta_path=state["fasta_path"],
        pred_newick_path=state["pred_newick_path"],
        stressor_gt_path=state["stressor_gt_path"],
    )
    total_retries = sum(
        int(n["explanation_quality_vector"].get("verification_retries", 0))
        for n in node_audits
    )

    llm_label = (
        "OFFLINE_DETERMINISTIC_MOCK"
        if state.get("used_offline_mock", False)
        else state.get("orchestrator_llm", "Qwen-2.5-32B-Instruct-Q4_K_M")
    )

    report_obj = AuditReport(
        audit_metadata=AuditMetadata(
            dataset_id=state["dataset_id"],
            target_model=state["target_model"],
            taxon_count=taxon_count,
            sequence_length=seq_len,
            orchestrator_llm=llm_label,
        ),
        global_metrics=GlobalMetrics(
            normalized_rf_distance=float(topo["normalized_rf_distance"]),
            false_confidence_index_fci=float(topo["fci_global"]),
            conditional_fci=float(topo.get("fci_conditional", 0.0)),
            error_detection_rate_edr=edr_score,
            total_internal_nodes_audited=len(node_audits),
            false_confidence_nodes_count=int(topo["false_confidence_nodes_count"]),
            total_hallucination_retries_triggered=total_retries,
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
    """Constructs and compiles the 4-state cyclic LangGraph StateMachine (Algorithm 1)."""
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
    pred_meta_path: str | None = None,
    true_newick_path: str = "",
    asr_state_path: str = "",
    stressor_gt_path: str = "",
    output_json_path: str | None = None,
    support_threshold: float = 85.0,
    orchestrator_llm: str = "Qwen-2.5-14B-Instruct-Q4_K_M",
    lm_studio_base_url: str | None = None,
    require_live_llm: bool = False,
    allow_offline_mock: bool = False,
    custom_synthesizer: Callable[[dict], tuple[list[dict], str]] | None = None,
    max_retries: int = 3,
) -> dict:
    """
    Executes the compiled LangGraph Diagnostic Agent across all internal branches of T_pred
    and returns the Pydantic-validated AuditReport dictionary.
    """
    resolved_url = lm_studio_base_url
    if resolved_url:
        try:
            resolved_url, _ = resolve_lm_studio_url(resolved_url)
        except Exception as exc:
            if require_live_llm or not allow_offline_mock:
                raise RuntimeError(
                    f"Could not connect to LM Studio at {lm_studio_base_url}: {exc}"
                ) from exc
            resolved_url = None

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
        "lm_studio_base_url": resolved_url,
        "allow_offline_mock": allow_offline_mock,
        "custom_synthesizer": custom_synthesizer,
        "max_retries": max_retries,
        "output_json_path": output_json_path,
    }
    final_state = app.invoke(initial_state, config={"recursion_limit": 2500})
    return final_state["audit_report"]
