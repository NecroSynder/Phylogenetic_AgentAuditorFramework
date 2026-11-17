"""
AI Baseline Comparison Models (Manuscript Section 4.3.1, Pages 19–21 / PDF Pages 23–25):
  - AI Baseline 1 (Zero-Shot LLM — No Tools):
      Receives raw Newick strings (T_pred, T_true), clade leaf membership + raw support %,
      and raw FASTA sequence text in a single pass WITHOUT access to the 5 deterministic Python
      tools (does NOT receive pre-computed CONCORDANT/DISCORDANT labels, M_true, or A_true)
      and WITHOUT the State 4 coordinate verification gate.
  - AI Baseline 2 (Generic Unconstrained Agent — Basic Window Inspection Only):
      Receives basic tree topology info + raw alignment window inspection (inspect_alignment_window)
      WITHOUT pre-computed ASR matrices (M_true), homoplasy/stressor calculators (A_true),
      conditional State 3 stressor routing, or the State 4 verify_claim_coordinates gate.

All outputs are evaluated post-hoc using the exact same `evaluate_single_node_audit` and
`compute_report_edr` equations as the Proposed 4-State Agent, guaranteeing zero scoring bias.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.agent.diagnostic_agent import _extract_json_block
from src.evaluation.metrics import (
    VALID_STRESSOR_LABELS,
    compute_lcr,
    compute_report_edr,
    compute_tree_goodness,
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
)
from src.utils.io_cache import (
    load_alignment_data,
    resolve_lm_studio_url,
    sort_node_id_key,
)

VALID_STATUSES = {"CONCORDANT", "DISCORDANT_LOW_SUPPORT", "FALSE_CONFIDENCE_ERROR"}


def _build_fasta_excerpt(fasta_path: str, clade_taxa: list[str], max_cols: int = 250) -> str:
    taxa_ids, char_mat, seq_len = load_alignment_data(fasta_path)
    cols = min(seq_len, max_cols)
    clade_set = set(clade_taxa)
    in_indices = [i for i, tid in enumerate(taxa_ids) if tid in clade_set]
    out_indices = [i for i, tid in enumerate(taxa_ids) if tid not in clade_set][:4]
    ordered_indices = (in_indices + out_indices)[:12]
    lines = []
    for idx in ordered_indices:
        snippet = "".join(char_mat[idx, :cols])
        lines.append(f">{taxa_ids[idx]}\n{snippet}")
    return "\n".join(lines)


def _fetch_ground_truth_for_node(
    node_info: dict,
    fasta_path: str,
    pred_newick_path: str,
    asr_state_path: str,
    stressor_gt_path: str,
) -> tuple[list[dict], dict]:
    """Retrieves ground-truth branch mutations and stressor diagnosis purely for post-hoc scoring."""
    has_true_match = (
        node_info["topological_status"] == "CONCORDANT"
        and node_info.get("true_matching_node_id") is not None
    )
    if has_true_match:
        true_mutations = extract_branch_synapomorphies(
            fasta_path=fasta_path,
            asr_state_path=asr_state_path,
            parent_node_id=node_info.get("true_parent_node_id") or "N_ROOT",
            child_node_id=node_info["true_matching_node_id"],
            is_concordant=True,
            top_k=10,
        )
    else:
        true_mutations = extract_branch_synapomorphies(
            fasta_path=fasta_path,
            asr_state_path=asr_state_path,
            parent_node_id=node_info["parent_node_id"],
            child_node_id=node_info["node_id"],
            is_concordant=False,
            clade_taxa=node_info["clade_taxa"],
            stressor_gt_path=stressor_gt_path,
            top_k=10,
        )

    if node_info["topological_status"] == "CONCORDANT":
        true_stressor_diag = diagnose_evolutionary_stressor(
            fasta_path=fasta_path,
            pred_newick_path=pred_newick_path,
            node_id=node_info["node_id"],
            stressor_gt_path=stressor_gt_path,
        )
        true_stressor_diag = {**true_stressor_diag, "primary_artifact": "NONE"}
    else:
        true_stressor_diag = diagnose_evolutionary_stressor(
            fasta_path=fasta_path,
            pred_newick_path=pred_newick_path,
            node_id=node_info["node_id"],
            stressor_gt_path=stressor_gt_path,
        )
    return true_mutations, true_stressor_diag


def _finalize_baseline_report(
    dataset_id: str,
    target_model: str,
    llm_header: str,
    taxon_count: int,
    seq_len: int,
    topo_res: dict,
    node_audits: list[dict],
    branch_info_weights: dict[str, float],
    agent_claimed_statuses: dict[str, str],
    total_claims_evaluated: int,
    total_claims_verified: int,
    fasta_path: str,
    pred_newick_path: str,
    asr_state_path: str,
    stressor_gt_path: str,
    output_json_path: str | None,
) -> dict:
    if total_claims_evaluated > 0:
        global_ccvr = round(total_claims_verified / total_claims_evaluated, 4)
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
        global_ccvr = 0.0

    lcr_score = compute_lcr(
        node_audits=node_audits,
        pred_newick_path=pred_newick_path,
        agent_claimed_statuses=agent_claimed_statuses,
    )
    g_tree = compute_tree_goodness(
        node_audits=node_audits,
        m_true_tsv_path=asr_state_path,
        lcr_score=lcr_score,
        branch_info_weights=branch_info_weights,
    )
    edr_score = compute_report_edr(
        node_audits=node_audits,
        fasta_path=fasta_path,
        pred_newick_path=pred_newick_path,
        stressor_gt_path=stressor_gt_path,
    )

    report_obj = AuditReport(
        audit_metadata=AuditMetadata(
            dataset_id=dataset_id,
            target_model=target_model,
            taxon_count=taxon_count,
            sequence_length=seq_len,
            orchestrator_llm=llm_header,
        ),
        global_metrics=GlobalMetrics(
            normalized_rf_distance=float(topo_res["normalized_rf_distance"]),
            false_confidence_index_fci=float(topo_res["fci_global"]),
            conditional_fci=float(topo_res.get("fci_conditional", 0.0)),
            error_detection_rate_edr=edr_score,
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


def run_zero_shot_llm_baseline(
    dataset_id: str,
    target_model: str,
    fasta_path: str,
    pred_newick_path: str,
    pred_meta_path: str | None,
    true_newick_path: str,
    asr_state_path: str,
    stressor_gt_path: str,
    output_json_path: str | None = None,
    support_threshold: float = 85.0,
    zeroshot_llm: str = "qwen/qwen3-14b",
    lm_studio_base_url: str | None = None,
    require_live_llm: bool = False,
    allow_offline_mock: bool = False,
    seed: int = 42,
) -> dict:
    """
    AI Baseline 1 (Zero-Shot LLM — No Tools):
    Prompts the LLM with raw Newick strings and raw FASTA sequence text without any Python
    bioinformatics tool execution, pre-computed ASR/homoplasy tables, or State 4 verification gate.
    """
    rng = np.random.default_rng(seed)
    topo_res = audit_tree_topology(
        pred_newick_path=pred_newick_path,
        true_newick_path=true_newick_path,
        pred_meta_path=pred_meta_path,
        support_threshold=support_threshold,
        stressor_gt_path=stressor_gt_path,
    )
    pred_nwk_str = Path(pred_newick_path).read_text().strip()
    true_nwk_str = Path(true_newick_path).read_text().strip()
    taxa_ids, char_mat, seq_len = load_alignment_data(fasta_path)
    taxon_count = len(taxa_ids)

    nodes_sorted = sorted(
        topo_res["node_bipartitions"],
        key=lambda r: sort_node_id_key(r["node_id"]),
    )

    resolved_url = lm_studio_base_url
    if resolved_url:
        try:
            resolved_url, _ = resolve_lm_studio_url(resolved_url)
        except Exception as exc:
            if require_live_llm or not allow_offline_mock:
                raise RuntimeError(
                    f"Zero-Shot Baseline could not connect to LM Studio at {lm_studio_base_url}: {exc}"
                ) from exc
            resolved_url = None

    if resolved_url is None and not allow_offline_mock:
        raise RuntimeError(
            "Zero-Shot Baseline requires a live LM Studio URL unless allow_offline_mock=True (--offline-mock) is set."
        )

    llm = None
    if resolved_url:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            base_url=resolved_url,
            api_key="lm-studio",
            model=zeroshot_llm,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
            timeout=90.0,
        )

    node_audits = []
    branch_info_weights: dict[str, float] = {}
    agent_claimed_statuses: dict[str, str] = {}
    tot_eval = 0
    tot_ver = 0
    used_mock = resolved_url is None

    for node_info in nodes_sorted:
        nid = node_info["node_id"]
        clade_taxa = node_info["clade_taxa"]
        sup = float(node_info["support_pct"])

        draft_claims = []
        draft_explanation = ""
        pred_stressor = "NONE"
        claimed_status = "CONCORDANT" if sup >= support_threshold else "DISCORDANT_LOW_SUPPORT"

        if llm is not None:
            fasta_excerpt = _build_fasta_excerpt(fasta_path, clade_taxa, max_cols=250)
            prompt = (
                "You are a Zero-Shot LLM Phylogenetic Auditor (AI Baseline 1 — no external tools).\n"
                "Confidence Tier Rubric: <50% = 'low confidence', 50% to <85% = 'moderate confidence', "
                ">=85% = 'definitive high confidence' (state ONLY the single tier phrase matching this node's "
                "bootstrap support—do not list the other rubric tiers).\n"
                "Inspect the raw Newick trees and raw FASTA alignment excerpt below, determine on your own "
                f"whether clade {nid} (taxa: {clade_taxa}, bootstrap support: {sup}%) is concordant or discordant "
                "with the Reference Tree Newick, and return ONLY a valid JSON object with keys:\n"
                "1. \"claimed_topological_status\": one of [\"CONCORDANT\", \"DISCORDANT_LOW_SUPPORT\", \"FALSE_CONFIDENCE_ERROR\"]\n"
                "2. \"primary_stressor\": one of [\"NONE\", \"LBA\", \"INDEL_GAP\", \"RECOMBINATION\", \"SHORT_BRANCH_HOMOPLASY\", \"SHORT_BRANCH_HOMOPLASY_AND_LBA\"]\n"
                "3. \"draft_claims\": list of 1 to 3 observed site mutations from the FASTA text supporting this split, each with keys:\n"
                "   {\"site_coordinate\": <int 1-based column>, \"ancestral_state\": \"<char>\", \"derived_state\": \"<char>\", "
                "\"posterior_probability\": 0.8, \"consistency_index_ci\": 0.8, \"is_indel\": <bool>}\n"
                f"4. \"explanation\": a 1-2 sentence explanation for node {nid} ({sup}% support).\n"
                f"Target Model: {target_model}\n"
                f"Predicted Tree Newick: {pred_nwk_str}\n"
                f"Reference Tree Newick: {true_nwk_str}\n"
                f"Raw FASTA Excerpt (first 250 columns):\n{fasta_excerpt}\n"
                "/no_think"
            )
            try:
                resp = llm.invoke(prompt)
                content_str = resp.content if isinstance(resp.content, str) else str(resp.content)
                parsed = _extract_json_block(content_str)
                draft_claims = list(parsed.get("draft_claims", []))
                draft_explanation = str(parsed.get("explanation", f"Zero-shot audit of node {nid}."))
                pred_stressor = str(parsed.get("primary_stressor", "NONE"))
                raw_st = str(parsed.get("claimed_topological_status", claimed_status))
                if raw_st in VALID_STATUSES:
                    claimed_status = raw_st
            except Exception as exc:
                raise RuntimeError(
                    f"Zero-Shot LLM query failed at {resolved_url} on node {nid}: {exc}"
                ) from exc
        else:
            # Offline deterministic mock (unassisted visual column guess from first 120 cols without ASR tools)
            guessed_site = int(rng.integers(1, max(2, min(seq_len, 120))))
            col_vals = char_mat[:, guessed_site - 1]
            s_u = str(col_vals[0]) if col_vals[0] != "-" else "A"
            s_v = "G" if s_u != "G" else "T"
            draft_claims = [
                {
                    "site_coordinate": guessed_site,
                    "ancestral_state": s_u,
                    "derived_state": s_v,
                    "posterior_probability": 0.7,
                    "consistency_index_ci": 0.6,
                    "is_indel": False,
                }
            ]
            # Without topological tools, unassisted zero-shot relies on support heuristic
            claimed_status = "CONCORDANT" if sup >= 50.0 else "DISCORDANT_LOW_SUPPORT"
            pred_stressor = "NONE" if claimed_status == "CONCORDANT" else "SHORT_BRANCH_HOMOPLASY"
            draft_explanation = (
                f"Zero-shot inspection of {target_model} at {nid} ({sup}% support) "
                f"claims {claimed_status.lower()} split with {s_u}{guessed_site}{s_v} at site {guessed_site}."
            )

        agent_claimed_statuses[nid] = claimed_status
        true_mutations, true_stressor_diag = _fetch_ground_truth_for_node(
            node_info=node_info,
            fasta_path=fasta_path,
            pred_newick_path=pred_newick_path,
            asr_state_path=asr_state_path,
            stressor_gt_path=stressor_gt_path,
        )
        record, ver_res, b_weight = evaluate_single_node_audit(
            node_info=node_info,
            draft_claims=draft_claims,
            draft_explanation=draft_explanation,
            predicted_stressor=pred_stressor,
            true_branch_mutations=true_mutations,
            true_stressor_diag=true_stressor_diag,
            fasta_path=fasta_path,
            pred_meta_path=pred_meta_path,
            asr_state_path=asr_state_path,
            retry_count=0,
            seq_len=seq_len,
        )
        node_audits.append(record)
        branch_info_weights[nid] = b_weight

        e_c = int(ver_res.get("total_claims_evaluated", 0))
        v_c = int(ver_res.get("verified_claims_count", 0))
        if e_c == 0 and len(true_mutations) > 0:
            e_c = 1
        tot_eval += e_c
        tot_ver += v_c

    llm_header = (
        "Baseline1_ZeroShot:OFFLINE_DETERMINISTIC_MOCK"
        if used_mock
        else f"Baseline1_ZeroShot:{zeroshot_llm}"
    )
    return _finalize_baseline_report(
        dataset_id=dataset_id,
        target_model=target_model,
        llm_header=llm_header,
        taxon_count=taxon_count,
        seq_len=seq_len,
        topo_res=topo_res,
        node_audits=node_audits,
        branch_info_weights=branch_info_weights,
        agent_claimed_statuses=agent_claimed_statuses,
        total_claims_evaluated=tot_eval,
        total_claims_verified=tot_ver,
        fasta_path=fasta_path,
        pred_newick_path=pred_newick_path,
        asr_state_path=asr_state_path,
        stressor_gt_path=stressor_gt_path,
        output_json_path=output_json_path,
    )


def run_generic_coding_agent_baseline(
    dataset_id: str,
    target_model: str,
    fasta_path: str,
    pred_newick_path: str,
    pred_meta_path: str | None,
    true_newick_path: str,
    asr_state_path: str,
    stressor_gt_path: str,
    output_json_path: str | None = None,
    support_threshold: float = 85.0,
    generic_agent_llm: str = "qwen/qwen3-14b",
    lm_studio_base_url: str | None = None,
    require_live_llm: bool = False,
    allow_offline_mock: bool = False,
    seed: int = 42,
) -> dict:
    """
    AI Baseline 2 (Generic Unconstrained Coding Agent):
    Has basic topology & raw alignment window inspection (audit_tree_topology + inspect_alignment_window),
    WITHOUT pre-computed ASR matrices (M_true), homoplasy/stressor calculators (A_true),
    or the State 4 verify_claim_coordinates self-correction gate (Manuscript Section 4.3.1, Page 23).
    """
    rng = np.random.default_rng(seed)
    topo_res = audit_tree_topology(
        pred_newick_path=pred_newick_path,
        true_newick_path=true_newick_path,
        pred_meta_path=pred_meta_path,
        support_threshold=support_threshold,
        stressor_gt_path=stressor_gt_path,
    )
    taxa_ids, char_mat, seq_len = load_alignment_data(fasta_path)
    taxon_count = len(taxa_ids)

    nodes_sorted = sorted(
        topo_res["node_bipartitions"],
        key=lambda r: sort_node_id_key(r["node_id"]),
    )

    resolved_url = lm_studio_base_url
    if resolved_url:
        try:
            resolved_url, _ = resolve_lm_studio_url(resolved_url)
        except Exception as exc:
            if require_live_llm or not allow_offline_mock:
                raise RuntimeError(
                    f"Generic Agent Baseline could not connect to LM Studio at {lm_studio_base_url}: {exc}"
                ) from exc
            resolved_url = None

    if resolved_url is None and not allow_offline_mock:
        raise RuntimeError(
            "Generic Agent Baseline requires a live LM Studio URL unless allow_offline_mock=True (--offline-mock) is set."
        )

    llm = None
    if resolved_url:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            base_url=resolved_url,
            api_key="lm-studio",
            model=generic_agent_llm,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
            timeout=90.0,
        )

    node_audits = []
    branch_info_weights: dict[str, float] = {}
    agent_claimed_statuses: dict[str, str] = {}
    tot_eval = 0
    tot_ver = 0
    used_mock = resolved_url is None

    for node_info in nodes_sorted:
        nid = node_info["node_id"]
        clade_taxa = node_info["clade_taxa"]
        sup = float(node_info["support_pct"])

        # Generic agent inspects a fixed raw window [1..80] without M_true ASR or A_true homoplasy tools
        win_info = inspect_alignment_window(
            fasta_path=fasta_path,
            taxa_subset=clade_taxa,
            start_pos=1,
            end_pos=min(80, seq_len),
        )
        # Compact node summary without leaking internal T_true node IDs
        basic_node = {
            "node_id": nid,
            "clade_taxa": clade_taxa,
            "support_pct": sup,
            "topological_status": node_info["topological_status"],
        }

        draft_claims = []
        draft_explanation = ""
        pred_stressor = "NONE"

        if llm is not None:
            prompt = (
                "You are a Generic ReAct Coding Agent auditing a phylogenetic split (AI Baseline 2 — "
                "basic tree & raw FASTA window inspection only; no ASR matrices, homoplasy calculators, or verification gate).\n"
                "Confidence Tier Rubric: <50% = 'low confidence', 50% to <85% = 'moderate confidence', "
                ">=85% = 'definitive high confidence' (state ONLY the single tier phrase matching this node's "
                "bootstrap support—do not list the other rubric tiers).\n"
                "Given the node info and raw FASTA window inspection [1..80] below, return ONLY a JSON object with keys:\n"
                "1. \"primary_stressor\": one of [\"NONE\", \"LBA\", \"INDEL_GAP\", \"RECOMBINATION\", \"SHORT_BRANCH_HOMOPLASY\", \"SHORT_BRANCH_HOMOPLASY_AND_LBA\"]\n"
                "2. \"draft_claims\": list of mutation objects observed in the window, each with keys "
                "{\"site_coordinate\": <int>, \"ancestral_state\": \"<char>\", \"derived_state\": \"<char>\", "
                "\"posterior_probability\": 0.85, \"consistency_index_ci\": 0.8, \"is_indel\": <bool>}\n"
                f"3. \"explanation\": a 1-2 sentence summary for node {nid} ({sup}% support).\n"
                f"Node Info: {json.dumps(basic_node)}\n"
                f"Window Inspection [1..80]: {json.dumps({'window_gap_pct': win_info['window_gap_pct'], 'polymorphic_sites': win_info['polymorphic_sites'][:10], 'window_slice': win_info['window_slice']})}\n"
                "/no_think"
            )
            try:
                resp = llm.invoke(prompt)
                content_str = resp.content if isinstance(resp.content, str) else str(resp.content)
                parsed = _extract_json_block(content_str)
                draft_claims = list(parsed.get("draft_claims", []))
                draft_explanation = str(parsed.get("explanation", f"Generic agent audit of {nid}."))
                pred_stressor = str(parsed.get("primary_stressor", "NONE"))
            except Exception as exc:
                raise RuntimeError(
                    f"Generic Agent LLM query failed at {resolved_url} on node {nid}: {exc}"
                ) from exc
        else:
            # Offline deterministic mock: inspects polymorphic columns in raw window [1..80] without ASR posteriors
            poly_sites = win_info.get("polymorphic_sites", [])
            chosen_site = (
                int(poly_sites[0])
                if poly_sites
                else int(rng.integers(1, max(2, min(seq_len, 80))))
            )
            clade_set = set(clade_taxa)
            in_chars = [
                char_mat[i, chosen_site - 1]
                for i, tid in enumerate(taxa_ids)
                if tid in clade_set
            ]
            out_chars = [
                char_mat[i, chosen_site - 1]
                for i, tid in enumerate(taxa_ids)
                if tid not in clade_set
            ]
            s_v = in_chars[0] if in_chars else "T"
            s_u = out_chars[0] if (out_chars and out_chars[0] != s_v) else ("C" if s_v != "C" else "A")
            draft_claims = [
                {
                    "site_coordinate": chosen_site,
                    "ancestral_state": str(s_u),
                    "derived_state": str(s_v),
                    "posterior_probability": 0.75,
                    "consistency_index_ci": 0.5,
                    "is_indel": "-" in {str(s_u), str(s_v)},
                }
            ]
            if node_info["topological_status"] == "CONCORDANT":
                pred_stressor = "NONE"
            else:
                pred_stressor = "INDEL_GAP" if win_info["window_gap_pct"] >= 15.0 else "SHORT_BRANCH_HOMOPLASY"
            draft_explanation = (
                f"Generic agent inspected window [1..80] for {nid} ({sup}% support, {node_info['topological_status']}) "
                f"and flagged polymorphic site {chosen_site} ({s_u}->{s_v}) under {pred_stressor}."
            )

        agent_claimed_statuses[nid] = node_info["topological_status"]
        true_mutations, true_stressor_diag = _fetch_ground_truth_for_node(
            node_info=node_info,
            fasta_path=fasta_path,
            pred_newick_path=pred_newick_path,
            asr_state_path=asr_state_path,
            stressor_gt_path=stressor_gt_path,
        )
        record, ver_res, b_weight = evaluate_single_node_audit(
            node_info=node_info,
            draft_claims=draft_claims,
            draft_explanation=draft_explanation,
            predicted_stressor=pred_stressor,
            true_branch_mutations=true_mutations,
            true_stressor_diag=true_stressor_diag,
            fasta_path=fasta_path,
            pred_meta_path=pred_meta_path,
            asr_state_path=asr_state_path,
            retry_count=0,
            seq_len=seq_len,
        )
        node_audits.append(record)
        branch_info_weights[nid] = b_weight

        e_c = int(ver_res.get("total_claims_evaluated", 0))
        v_c = int(ver_res.get("verified_claims_count", 0))
        if e_c == 0 and len(true_mutations) > 0:
            e_c = 1
        tot_eval += e_c
        tot_ver += v_c

    llm_header = (
        "Baseline2_GenericAgent:OFFLINE_DETERMINISTIC_MOCK"
        if used_mock
        else f"Baseline2_GenericAgent:{generic_agent_llm}"
    )
    return _finalize_baseline_report(
        dataset_id=dataset_id,
        target_model=target_model,
        llm_header=llm_header,
        taxon_count=taxon_count,
        seq_len=seq_len,
        topo_res=topo_res,
        node_audits=node_audits,
        branch_info_weights=branch_info_weights,
        agent_claimed_statuses=agent_claimed_statuses,
        total_claims_evaluated=tot_eval,
        total_claims_verified=tot_ver,
        fasta_path=fasta_path,
        pred_newick_path=pred_newick_path,
        asr_state_path=asr_state_path,
        stressor_gt_path=stressor_gt_path,
        output_json_path=output_json_path,
    )
