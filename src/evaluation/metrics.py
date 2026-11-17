"""
================================================================================
PIPELINE STEP 4: MULTI-DIMENSIONAL EVALUATION & STATISTICAL VALIDATION
(Manuscript Section 4.3.1 & Section 4.3.2, Equations 4.1 – 4.15, Pages 19–27)
================================================================================
Implements all mathematical equations, unbiased shared node/report evaluators,
and statistical significance tests:

  Part A — Target Phylogenetic Model Metrics (Section 4.3.2.A):
    - Eq. 4.1: Normalized Robinson-Foulds Distance (RF_norm) [in bio_tools.py]
    - Eq. 4.2 & 4.3: Global & Conditional False-Confidence Index (FCI) [in bio_tools.py]
    - Eq. 4.4: Agent-Audited Error Detection Rate (EDR) against A_true ground truth

  Part B — Biological Accuracy, Faithfulness & Hallucination Metrics (Section 4.3.2.B):
    - Eq. 4.5 & 4.6: Feature Identification Precision(b), Recall(b), and F1(b)
    - Eq. 4.7: Bootstrap and Node Support Fidelity (BSF, negation-aware)
    - Eq. 4.8: Claim-to-Coordinate Verification Rate (CCVR, micro- & node-level)
    - Eq. 4.9: Faithfulness / Context Grounding Score (FGS, anchor-required)
    - Eq. 4.10: Logical Consistency Rate (LCR, structural & prose predicates)

  Part C — Bayesian Network Transition Information Gain & Unified Goodness (Section 4.3.2.C):
    - Eq. 4.11: CTMC Branch Transition Probability P = exp(Q * l_k) (4-state DNA & 20-state AA)
    - Eq. 4.12: Normalized Transition Information Gain IG(b_k) & Shannon Entropy Reduction
    - Eq. 4.13 & 4.14: 5D Explanation Quality Vector q(b_k) & Per-Node Goodness Score G_node(b_k)
    - Eq. 4.15: Reconciled Tree-Level Explanation Goodness Score G_tree(T_pred)

  Part D — Non-Parametric Statistical Significance Testing (Section 4.3.1):
    - Paired Two-Sided Wilcoxon Signed-Rank Test & McNemar's Contingency Test
    - Replicate-level multi-metric suite with Benjamini-Hochberg FDR correction at alpha = 0.05
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy.linalg import expm
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from src.schemas.audit_schema import (
    BayesianTransitionMetrics,
    CausalSequenceEvidence,
    DiagnosedFailureMode,
    ExplanationQualityVector,
    NodeAuditRecord,
)
from src.tools.bio_tools import (
    diagnose_evolutionary_stressor,
    extract_prose_coordinate_claims,
    verify_claim_coordinates,
)
from src.utils.io_cache import (
    load_alignment_data,
    load_ete_tree,
    load_json_dict,
    load_m_true_df,
)

DNA_STATES = {"A": 0, "C": 1, "G": 2, "T": 3}
AA_STATES = {aa: idx for idx, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}
VALID_MUTATION_CLASSES = {
    "SYNAPOMORPHY",
    "INDEL",
    "HOMOPLASY",
    "HOMOPLASTIC_CONVERGENCE",
}
VALID_STRESSOR_LABELS = {
    "NONE",
    "LBA",
    "INDEL_GAP",
    "RECOMBINATION",
    "SHORT_BRANCH_HOMOPLASY",
    "SHORT_BRANCH_HOMOPLASY_AND_LBA",
}


def classify_mutation_record(m: dict) -> str:
    """Maps a mutation dictionary to CausalSequenceEvidence.mutation_class."""
    if bool(m.get("is_indel", False)) or "-" in {
        str(m.get("parent_state", m.get("ancestral_state", ""))),
        str(m.get("child_state", m.get("derived_state", ""))),
    }:
        return "INDEL"
    ci_val = float(m.get("ci_score", m.get("consistency_index_ci", 1.0)))
    if ci_val >= 0.8:
        return "SYNAPOMORPHY"
    elif ci_val >= 0.5:
        return "HOMOPLASY"
    return "HOMOPLASTIC_CONVERGENCE"


# ==============================================================================
# PART A & B: FEATURE F1, BOOTSTRAP SUPPORT FIDELITY (BSF), FAITHFULNESS (FGS)
# ==============================================================================

def confidence_tier_label(support_pct: float) -> str:
    """
    Deterministic confidence tier mapping function phi(Support(b_k)) (Section 4.3.2.B.2, Eq. 4.7):
      - Low: Support(b_k) < 50%
      - Moderate: 50% <= Support(b_k) < 85%
      - Definitive/High: Support(b_k) >= 85%
    """
    if support_pct < 50.0:
        return "Low"
    elif support_pct < 85.0:
        return "Moderate"
    return "Definitive/High"


def _canonical_mutation_key(
    site: int, s_u: str, s_v: str, is_indel: bool = False
) -> tuple[Any, ...]:
    if is_indel or "-" in {s_u, s_v}:
        return (int(site), "GAP" if s_u == "-" else "NON_GAP", "GAP" if s_v == "-" else "NON_GAP")
    return (int(site), str(s_u).upper(), str(s_v).upper())


def compute_feature_precision_recall_f1(
    cited_sites: set[Any], true_sites: set[Any]
) -> tuple[float, float, float]:
    """
    Computes Feature Identification Precision(b), Recall(b), and F1(b) (Eqs. 4.5-4.6).
    Accepts either sets of integer site coordinates or sets of canonical (site, s_u, s_v) tuples.
    """
    if not true_sites and not cited_sites:
        return 1.0, 1.0, 1.0
    if not cited_sites or not true_sites:
        return 0.0, 0.0, 0.0

    tp = len(cited_sites & true_sites)
    precision = tp / len(cited_sites)
    recall = tp / len(true_sites)
    if precision + recall == 0.0:
        return 0.0, 0.0, 0.0
    f1 = (2.0 * precision * recall) / (precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def _detect_confidence_tiers_in_text(text: str) -> tuple[bool, bool, bool]:
    """Negation-aware detection of High, Moderate, and Low confidence descriptors in prose."""
    text_lower = text.lower()
    # Strip negated descriptors (e.g., 'does not have high support', 'not high confidence', 'lacks strong support')
    cleaned = re.sub(
        r"\b(?:not|no|never|without|lacks?|neither|non-)(?:\s+\w+){0,2}\s+"
        r"(?:definitive(?:\s+high)?|high[- ]confidence|high[- ]support|strong[- ]support|"
        r"moderate[- ]confidence|moderate[- ]support|moderate|"
        r"low[- ]confidence|low[- ]support|weak[- ]support|poor[- ]support|weak\s+split)\b",
        " ",
        text_lower,
    )

    has_high = bool(
        re.search(
            r"\b(?:definitive(?:\s+high)?|high[- ]confidence|high[- ]support|strong[- ]support)\b",
            cleaned,
        )
    )
    has_moderate = bool(
        re.search(
            r"\b(?:moderate[- ]confidence|moderate[- ]support|moderate)\b",
            cleaned,
        )
    )
    has_low = bool(
        re.search(
            r"\b(?:low[- ]confidence|low[- ]support|weak[- ]support|poor[- ]support|weak\s+split)\b",
            cleaned,
        )
    )
    return has_high, has_moderate, has_low


def compute_bsf_local(support_pct: float, explanation_text: str) -> int:
    """
    Computes binary Node Support Tier Fidelity BSF(b_k) in {0, 1} (Section 4.3.2.B.2, Eq. 4.7).
    Verifies that the natural-language confidence descriptor matches phi(Support(b_k)).
    """
    if not explanation_text or not explanation_text.strip():
        return 0
    expected_tier = confidence_tier_label(support_pct)
    has_high, has_moderate, has_low = _detect_confidence_tiers_in_text(explanation_text)

    if expected_tier == "Definitive/High":
        return 1 if (has_high and not has_low) else 0
    elif expected_tier == "Moderate":
        return 1 if (has_moderate and not has_high and not has_low) else 0
    else:
        return 1 if (has_low and not has_high) else 0


def _stressor_mentions_contradict(clause_lower: str, valid_artifact: str) -> bool:
    """Checks whether prose asserts an evolutionary stressor that contradicts O_tools(b)."""
    # Explicit 'stressor NONE' or 'no stressor' is consistent with valid_artifact == 'NONE'
    claims_lba = bool(
        re.search(r"\b(?:long[- ]branch\s+attraction|artifact\s+of\s+lba|stressor\s+lba|under\s+lba)\b", clause_lower)
    )
    claims_recomb = bool(
        re.search(r"\b(?:recombination|recombinant|horizontal\s+gene\s+transfer|stressor\s+recombination)\b", clause_lower)
    )
    claims_indel_stressor = bool(
        re.search(r"\b(?:indel[- ]gap|gap[- ]masking|stressor\s+indel_gap|under\s+indel_gap)\b", clause_lower)
    )
    claims_short_homoplasy = bool(
        re.search(r"\b(?:short[- ]branch[_\s]+homoplasy|radiation\s+burst)\b", clause_lower)
    )

    if valid_artifact == "NONE":
        return claims_lba or claims_recomb or claims_indel_stressor or claims_short_homoplasy
    if claims_recomb and valid_artifact != "RECOMBINATION":
        return True
    if claims_lba and valid_artifact not in ("LBA", "SHORT_BRANCH_HOMOPLASY_AND_LBA"):
        return True
    if claims_indel_stressor and valid_artifact != "INDEL_GAP":
        return True
    return False


def compute_fgs_local(
    explanation_text: str,
    node_id: str,
    support_pct: float,
    synapomorphies: list[dict],
    stressor_diag: dict | None,
    clade_taxa: list[str] | None = None,
) -> float:
    """
    Computes Faithfulness / Context Grounding Score FGS(b) in [0, 1] (Section 4.3.2.B.4, Eq. 4.9):
      FGS(b) = |{s in S(b) | O_tools(b) |= s}| / |S(b)|
    Verifies both singular/plural coordinate citations, mutation tokens, support numbers,
    and stressor assertions, AND requires each clause to contain at least one grounded context
    anchor so irrelevant ungrounded prose cannot score 1.0 vacuously.
    """
    if not explanation_text or not explanation_text.strip():
        return 0.0

    clauses = [
        c.strip()
        for c in re.split(r";\s*|\.\s+(?=[A-Z0-9]|$)", explanation_text.strip().rstrip("."))
        if c.strip()
    ]
    if not clauses:
        return 0.0

    valid_sites = {int(m["site"]) for m in synapomorphies}
    valid_pairs = {
        (int(m["site"]), str(m["parent_state"]), str(m["child_state"]))
        for m in synapomorphies
    }
    indel_sites = {int(m["site"]) for m in synapomorphies if bool(m.get("is_indel", False))}
    valid_artifact = stressor_diag["primary_artifact"] if stressor_diag else "NONE"
    taxa_lower = [t.lower() for t in (clade_taxa or [])]

    entailed_count = 0
    for clause in clauses:
        clause_ok = True
        c_lower = clause.lower()

        # 1. Check all singular & plural site citations and mutation tokens in the clause
        prose_sites, prose_muts = extract_prose_coordinate_claims(clause)
        for s_num in prose_sites:
            if s_num not in valid_sites:
                clause_ok = False
                break

        if clause_ok:
            for s_pos, s_u, s_v in prose_muts:
                is_indel_ok = s_pos in indel_sites and "-" in {s_u, s_v} and s_u != s_v
                if (s_pos, s_u, s_v) not in valid_pairs and not is_indel_ok:
                    clause_ok = False
                    break

        # 2. Check numeric support claims (catching '% support', '% bootstrap', '% posterior')
        if clause_ok:
            pct_tokens = re.findall(
                r"(\d+(?:\.\d+)?)%\s*(?:resampled\s+|bootstrap\s+|model\s+|posterior\s+)?(?:support|bootstrap|probability)",
                clause,
                flags=re.IGNORECASE,
            )
            for pct_str in pct_tokens:
                if abs(float(pct_str) - support_pct) > 1.0:
                    clause_ok = False
                    break

        # 3. Check stressor contradictions
        if clause_ok and _stressor_mentions_contradict(c_lower, valid_artifact):
            clause_ok = False

        # 4. Require at least one grounded context anchor from O_tools(b) (closes vacuous filler loophole)
        has_anchor = (
            node_id.lower() in c_lower
            or bool(prose_sites)
            or any(t in c_lower for t in taxa_lower)
            or any(
                kw in c_lower
                for kw in (
                    "concordant",
                    "discordant",
                    "false_confidence",
                    "false-confidence",
                    "synapomorph",
                    "homoplas",
                    "indel",
                    "support",
                    "confidence",
                    "reference tree",
                    "window",
                    "entropy",
                    "gap",
                    valid_artifact.lower(),
                )
            )
        )
        if clause_ok and has_anchor:
            entailed_count += 1

    return round(entailed_count / len(clauses), 4)


# ==============================================================================
# PART C: BAYESIAN TRANSITION INFO GAIN (IG) & UNIFIED EXPLANATION GOODNESS
# ==============================================================================

def compute_bayesian_transition_metrics(
    cited_mutations: list[dict],
    true_branch_mutations: list[dict],
    branch_length: float,
    clade_taxa: list[str],
    fasta_path: str,
) -> dict:
    """
    Computes Bayesian Network Transition Probability log-likelihood (Eq. 4.11),
    clade Shannon entropy reduction in bits, and Normalized Information Gain IG(b_k) (Eq. 4.12).
    Uses mtime-cached alignment matrix (zero redundant disk parses).
    """
    eff_len = max(float(branch_length), 1e-4)

    is_protein = any(
        str(m.get("parent_state", "A")) not in DNA_STATES
        and str(m.get("parent_state", "A")) in AA_STATES
        for m in true_branch_mutations
    )
    if is_protein:
        state_map = AA_STATES
        n_states = 20
    else:
        state_map = DNA_STATES
        n_states = 4

    rate_off = 1.0 / (n_states - 1.0)
    Q = np.full((n_states, n_states), rate_off)
    np.fill_diagonal(Q, -1.0)
    P_mat = expm(Q * eff_len)

    loglik_sum = 0.0
    for m in true_branch_mutations:
        u_s = str(m.get("parent_state", "A"))
        v_s = str(m.get("child_state", "C"))
        if u_s in state_map and v_s in state_map:
            prob = max(float(P_mat[state_map[u_s], state_map[v_s]]), 1e-12)
            loglik_sum += float(np.log(prob))
        else:
            loglik_sum += float(np.log(max(eff_len / (1.0 + eff_len), 1e-6)))

    if not true_branch_mutations:
        loglik_sum = round(float(np.log(max(float(P_mat[0, 0]), 1e-6))), 4)

    # 2. Clade Shannon entropy reduction (bits) across causal sites using cached alignment
    taxa_ids, char_mat, seq_len = load_alignment_data(fasta_path)
    n_total = len(taxa_ids)
    clade_set = set(clade_taxa)
    in_mask = np.array([tid in clade_set for tid in taxa_ids], dtype=bool)
    out_mask = ~in_mask

    def _col_entropy(chars: np.ndarray) -> float:
        if chars.size == 0:
            return 0.0
        _, counts = np.unique(chars, return_counts=True)
        probs = counts / chars.size
        return float(-np.sum(probs * np.log2(probs)))

    entropy_reductions = []
    for m in true_branch_mutations:
        col_idx = int(m["site"]) - 1
        if 0 <= col_idx < seq_len:
            all_chars = char_mat[:, col_idx]
            in_chars = all_chars[in_mask]
            out_chars = all_chars[out_mask]
            h_total = _col_entropy(all_chars)
            h_cond = (in_chars.size / n_total) * _col_entropy(in_chars) + (
                out_chars.size / n_total
            ) * _col_entropy(out_chars)
            entropy_reductions.append(max(h_total - h_cond, 0.0))

    entropy_red_bits = (
        round(float(np.sum(entropy_reductions)), 4) if entropy_reductions else 0.0
    )

    # 3. Normalized Information Gain IG(b_k) (Eq. 4.12) matching verified (site, s_u, s_v) transitions
    true_sls_by_key = {
        _canonical_mutation_key(
            int(m["site"]),
            str(m["parent_state"]),
            str(m["child_state"]),
            bool(m.get("is_indel", False)),
        ): max(float(m.get("delta_sls", 0.0)), 0.0)
        for m in true_branch_mutations
    }
    total_sls = sum(true_sls_by_key.values())
    cited_keys = {
        _canonical_mutation_key(
            int(c.get("site_coordinate", c.get("site", -1))),
            str(c.get("ancestral_state", c.get("parent_state", ""))),
            str(c.get("derived_state", c.get("child_state", ""))),
            bool(c.get("is_indel", False)),
        )
        for c in cited_mutations
    }
    captured_sls = sum(
        sls for k, sls in true_sls_by_key.items() if k in cited_keys
    )

    if total_sls > 0.0:
        ig_score = min(round(captured_sls / total_sls, 4), 1.0)
    else:
        ig_score = 1.0 if not cited_mutations else 0.0

    return {
        "ctmc_branch_transition_loglik": round(loglik_sum, 4),
        "clade_shannon_entropy_reduction_bits": entropy_red_bits,
        "transition_information_gain_ig": ig_score,
    }


def compute_node_goodness(
    ccvr_local: float,
    bsf_local: int,
    feature_f1: float,
    information_gain_ig: float,
    faithfulness_fgs: float,
    alphas: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
) -> float:
    """
    Computes Per-Node Explanation Goodness Score G_node(b_k) in [0, 1] (Section 4.3.2.C.3, Eq. 4.14):
      G_node(b_k) = CCVR(b_k) * BSF(b_k) * (alpha_1 * F1(b_k) + alpha_2 * IG(b_k) + alpha_3 * FGS(b_k))
    """
    a1, a2, a3 = alphas
    weighted_sum = (
        (a1 * feature_f1) + (a2 * information_gain_ig) + (a3 * faithfulness_fgs)
    )
    score = float(ccvr_local) * float(bsf_local) * weighted_sum
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def compute_lcr(
    node_audits: list[dict],
    pred_newick_path: str,
    agent_claimed_statuses: dict[str, str] | None = None,
) -> float:
    """
    Computes global Logical Consistency Rate LCR(T_pred) in [0, 1] (Section 4.3.2.B.5, Eq. 4.10):
    Evaluates structural clade-hierarchy compatibility, support-vs-status predicates,
    and prose-vs-topology consistency across all audited internal nodes.
    """
    if not node_audits:
        return 1.0

    t_pred = load_ete_tree(pred_newick_path, fmt=1)
    true_tree_clades = {
        n.name: set(n.get_leaf_names())
        for n in t_pred.traverse("postorder")
        if not n.is_leaf() and not n.is_root()
    }

    total_predicates = 0
    contradictions = 0

    for rec in node_audits:
        nid = rec["node_id"]
        claimed_clade = set(rec["predicted_clade_taxa"])
        sup = float(rec["model_support_pct"])
        true_status = rec["topological_status"]
        asserted_status = (
            agent_claimed_statuses.get(nid, true_status)
            if agent_claimed_statuses
            else true_status
        )
        prose = rec.get("diagnosed_failure_mode", {}).get("agent_causal_explanation", "")
        prose_lower = prose.lower()

        # Predicate 1: Clade membership matches T_pred topology
        total_predicates += 1
        if nid in true_tree_clades and claimed_clade != true_tree_clades[nid]:
            contradictions += 1

        # Predicate 2: Topological status vs support threshold consistency
        total_predicates += 1
        if asserted_status == "FALSE_CONFIDENCE_ERROR" and sup < 85.0:
            contradictions += 1
        elif asserted_status == "DISCORDANT_LOW_SUPPORT" and sup >= 85.0:
            contradictions += 1
        elif asserted_status != true_status:
            contradictions += 1

        # Predicate 3: Prose status & confidence-tier internal consistency
        total_predicates += 1
        claims_concordant_prose = bool(
            re.search(r"(?<!not\s)(?<!non-)(?<!in)\bconcordant\b", prose_lower)
        ) and ("discordant" not in prose_lower)
        claims_discordant_prose = bool(
            re.search(r"\b(?:discordant|false[_\s-]confidence|erroneous\s+split)\b", prose_lower)
        )
        has_high, _, has_low = _detect_confidence_tiers_in_text(prose)

        prose_contradiction = False
        if true_status == "CONCORDANT" and claims_discordant_prose:
            prose_contradiction = True
        elif true_status != "CONCORDANT" and claims_concordant_prose:
            prose_contradiction = True
        elif sup >= 85.0 and has_low:
            prose_contradiction = True
        elif sup < 50.0 and has_high:
            prose_contradiction = True

        if prose_contradiction:
            contradictions += 1

    # Predicate 4: Pairwise clade Laminar / Hierarchical Compatibility on a tree
    n_records = len(node_audits)
    for i in range(n_records):
        for j in range(i + 1, n_records):
            c_i = set(node_audits[i]["predicted_clade_taxa"])
            c_j = set(node_audits[j]["predicted_clade_taxa"])
            total_predicates += 1
            intersect = c_i & c_j
            if intersect and not (c_i.issubset(c_j) or c_j.issubset(c_i)):
                contradictions += 1

    if total_predicates == 0:
        return 1.0
    return round(max(1.0 - (contradictions / total_predicates), 0.0), 4)


def compute_tree_goodness(
    node_audits: list[dict],
    m_true_tsv_path: str,
    lcr_score: float,
    epsilon: float = 1e-6,
    branch_info_weights: dict[str, float] | None = None,
) -> float:
    """
    Computes Reconciled Tree-Level Explanation Goodness Score G_tree(T_pred) in [0, 1]
    (Section 4.3.2.C.4, Eq. 4.15):
      G_tree(T_pred) = LCR(T_pred) * sum_{k=1}^K w_k * G_node(b_k)
    Uses true matched branch information weights when provided so discordant node IDs
    never collide with unrelated child_node IDs in M_true.tsv.
    """
    if not node_audits:
        return 0.0

    m_df = load_m_true_df(m_true_tsv_path) if branch_info_weights is None else None
    weights = []
    node_scores = []

    for rec in node_audits:
        nid = rec["node_id"]
        if branch_info_weights is not None and nid in branch_info_weights:
            i_branch = max(float(branch_info_weights[nid]), 0.0) + epsilon
        elif m_df is not None and not m_df.empty:
            sub = m_df[m_df["child_node"] == nid]
            i_branch = float(sub["delta_sls"].clip(lower=0.0).sum()) + epsilon
        else:
            i_branch = epsilon
        weights.append(i_branch)
        node_scores.append(float(rec["node_goodness_score"]))

    total_weight = sum(weights)
    if total_weight <= 0.0:
        norm_weights = [1.0 / len(node_audits)] * len(node_audits)
    else:
        norm_weights = [w / total_weight for w in weights]

    weighted_avg = sum(w * g for w, g in zip(norm_weights, node_scores))
    return round(float(np.clip(lcr_score * weighted_avg, 0.0, 1.0)), 4)


def compute_edr(flagged_anomalies: set[str], true_anomalies: set[str]) -> float:
    """
    Computes Agent-Audited Error Detection Rate (EDR) (Section 4.3.2.A, Eq. 4.4):
      EDR = |A_flagged intersect A_true| / |A_true|
    """
    if not true_anomalies:
        return 1.0
    return round(len(flagged_anomalies & true_anomalies) / len(true_anomalies), 4)


def compute_report_edr(
    node_audits: list[dict],
    fasta_path: str,
    pred_newick_path: str,
    stressor_gt_path: str,
) -> float:
    """
    Evaluates Agent-Audited Error Detection Rate (Eq. 4.4) strictly against deterministic
    ground-truth stressor diagnoses on discordant splits (eliminating self-referential EDR bias).
    """
    true_anomalies: set[str] = set()
    flagged_anomalies: set[str] = set()

    for rec in node_audits:
        if rec["topological_status"] == "CONCORDANT":
            continue
        nid = rec["node_id"]
        gt_diag = diagnose_evolutionary_stressor(
            fasta_path=fasta_path,
            pred_newick_path=pred_newick_path,
            node_id=nid,
            stressor_gt_path=stressor_gt_path,
        )
        gt_stressor = gt_diag["primary_artifact"]
        pred_stressor = rec["diagnosed_failure_mode"]["primary_stressor"]

        if gt_stressor != "NONE":
            true_anomalies.add(f"{nid}:{gt_stressor}")
            if pred_stressor == gt_stressor or (
                gt_stressor == "SHORT_BRANCH_HOMOPLASY_AND_LBA"
                and pred_stressor in ("LBA", "SHORT_BRANCH_HOMOPLASY")
            ):
                # Exact match required; partial credit NOT added to set unless exact tag matches
                if pred_stressor == gt_stressor:
                    flagged_anomalies.add(f"{nid}:{gt_stressor}")
        else:
            # Topological error with unstructured noise: auditor must flag discordant status without inventing wrong stressor
            true_anomalies.add(f"{nid}:DISCORDANT")
            if pred_stressor == "NONE":
                flagged_anomalies.add(f"{nid}:DISCORDANT")

    return compute_edr(flagged_anomalies, true_anomalies)


# ==============================================================================
# SHARED UNBIASED NODE AUDITOR EVALUATOR (USED BY AGENT & BASELINES)
# ==============================================================================

def evaluate_single_node_audit(
    node_info: dict,
    draft_claims: list[dict],
    draft_explanation: str,
    predicted_stressor: str,
    true_branch_mutations: list[dict],
    true_stressor_diag: dict,
    fasta_path: str,
    pred_meta_path: str | None,
    asr_state_path: str,
    retry_count: int = 0,
    seq_len: int | None = None,
) -> tuple[dict, dict, float]:
    """
    Single canonical node evaluation pipeline shared across Proposed 4-State Agent,
    Baseline 1 (Zero-Shot LLM), and Baseline 2 (Generic Agent) to guarantee zero
    scoring bias across methods.
    Returns (node_audit_record_dict, verification_result_dict, branch_info_weight).
    """
    nid = node_info["node_id"]
    clade_taxa = node_info["clade_taxa"]
    sup_pct = float(node_info["support_pct"])

    if seq_len is None:
        _, _, seq_len = load_alignment_data(fasta_path)

    ver_res = verify_claim_coordinates(
        draft_claims=draft_claims,
        fasta_path=fasta_path,
        asr_state_path=asr_state_path,
        valid_branch_mutations=true_branch_mutations,
        explanation_text=draft_explanation,
    )

    branch_len = 0.05
    if pred_meta_path:
        pred_meta = load_json_dict(pred_meta_path)
        branch_len = float(pred_meta.get("nodes", {}).get(nid, {}).get("branch_length", 0.05))

    cited_keys = {
        _canonical_mutation_key(
            int(c.get("site_coordinate", c.get("site", -1))),
            str(c.get("ancestral_state", c.get("parent_state", ""))),
            str(c.get("derived_state", c.get("child_state", ""))),
            bool(c.get("is_indel", False)),
        )
        for c in draft_claims
        if 1 <= int(c.get("site_coordinate", c.get("site", -1))) <= seq_len
    }
    true_keys = {
        _canonical_mutation_key(
            int(m["site"]),
            str(m["parent_state"]),
            str(m["child_state"]),
            bool(m.get("is_indel", False)),
        )
        for m in true_branch_mutations
    }
    _, _, f1_score = compute_feature_precision_recall_f1(cited_keys, true_keys)

    ccvr_local = float(ver_res["ccvr_score"])
    bsf_local = compute_bsf_local(sup_pct, draft_explanation)
    fgs_local = compute_fgs_local(
        explanation_text=draft_explanation,
        node_id=nid,
        support_pct=sup_pct,
        synapomorphies=true_branch_mutations,
        stressor_diag=true_stressor_diag,
        clade_taxa=clade_taxa,
    )
    bayes_metrics = compute_bayesian_transition_metrics(
        cited_mutations=draft_claims,
        true_branch_mutations=true_branch_mutations,
        branch_length=branch_len,
        clade_taxa=clade_taxa,
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
        raw_site = int(c.get("site_coordinate", c.get("site", 1)))
        site_coord = max(1, raw_site)
        s_u = str(c.get("ancestral_state", c.get("parent_state", "A"))).strip() or "A"
        s_v = str(c.get("derived_state", c.get("child_state", "C"))).strip() or "C"
        p_prob = float(np.clip(float(c.get("posterior_probability", c.get("posterior_prob", 0.8))), 0.0, 1.0))
        ci_val = float(np.clip(float(c.get("consistency_index_ci", c.get("ci_score", 0.5))), 0.0, 1.0))
        m_class = str(c.get("mutation_class", classify_mutation_record(c)))
        if m_class not in VALID_MUTATION_CLASSES:
            m_class = classify_mutation_record(c)
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

    stressor_clean = (
        predicted_stressor if predicted_stressor in VALID_STRESSOR_LABELS else "NONE"
    )
    node_record = NodeAuditRecord(
        node_id=nid,
        predicted_clade_taxa=clade_taxa,
        model_support_pct=sup_pct,
        topological_status=node_info["topological_status"],
        bayesian_transition_metrics=BayesianTransitionMetrics(**bayes_metrics),
        explanation_quality_vector=ExplanationQualityVector(
            ccvr_local=ccvr_local,
            bsf_local=bsf_local,
            feature_f1=f1_score,
            information_gain_ig=ig_local,
            faithfulness_fgs=fgs_local,
            verification_retries=retry_count,
        ),
        node_goodness_score=g_node,
        causal_sequence_evidence=evidence_list,
        diagnosed_failure_mode=DiagnosedFailureMode(
            primary_stressor=stressor_clean,
            lba_branch_ratio=max(float(true_stressor_diag["lba_branch_ratio"]), 0.0),
            local_gap_density_pct=float(
                np.clip(float(true_stressor_diag["gap_density_pct"]), 0.0, 100.0)
            ),
            agent_causal_explanation=draft_explanation,
        ),
    ).model_dump()

    branch_info_weight = sum(
        max(float(m.get("delta_sls", 0.0)), 0.0) for m in true_branch_mutations
    )
    return node_record, ver_res, branch_info_weight


# ==============================================================================
# PART D: PAIRED STATISTICAL SIGNIFICANCE TESTING (SECTION 4.3.1)
# ==============================================================================

def run_statistical_validation(
    agent_continuous: list[float],
    baseline_continuous: list[float],
    agent_binary: list[int],
    baseline_binary: list[int],
    alpha: float = 0.05,
) -> dict:
    """
    Executes paired two-sided Wilcoxon signed-rank test (for continuous metrics)
    and McNemar's test (for binary verification proportions) with Benjamini-Hochberg
    FDR correction at alpha = 0.05 (Section 4.3.1, Page 22 / PDF Page 26).
    """
    diffs = np.asarray(agent_continuous, dtype=float) - np.asarray(
        baseline_continuous, dtype=float
    )
    if diffs.size == 0 or np.allclose(diffs, 0.0):
        w_stat, w_pval = 0.0, 1.0
    else:
        res_w = wilcoxon(
            agent_continuous,
            baseline_continuous,
            zero_method="zsplit",
            alternative="two-sided",
        )
        w_stat, w_pval = float(res_w.statistic), float(res_w.pvalue)

    a_bin = np.asarray(agent_binary, dtype=int)
    b_bin = np.asarray(baseline_binary, dtype=int)
    n11 = int(np.sum((a_bin == 1) & (b_bin == 1)))
    n10 = int(np.sum((a_bin == 1) & (b_bin == 0)))
    n01 = int(np.sum((a_bin == 0) & (b_bin == 1)))
    n00 = int(np.sum((a_bin == 0) & (b_bin == 0)))
    table = [[n11, n10], [n01, n00]]

    if n10 + n01 == 0:
        m_stat, m_pval = 0.0, 1.0
    else:
        res_m = mcnemar(table, exact=(n10 + n01 < 25))
        m_stat, m_pval = float(res_m.statistic), float(res_m.pvalue)

    reject, pvals_corrected, _, _ = multipletests(
        [w_pval, m_pval], alpha=alpha, method="fdr_bh"
    )
    return {
        "wilcoxon_statistic": round(w_stat, 4),
        "wilcoxon_p_raw": round(w_pval, 6),
        "wilcoxon_p_fdr_bh": round(float(pvals_corrected[0]), 6),
        "wilcoxon_significant": bool(reject[0]),
        "mcnemar_statistic": round(m_stat, 4),
        "mcnemar_p_raw": round(m_pval, 6),
        "mcnemar_p_fdr_bh": round(float(pvals_corrected[1]), 6),
        "mcnemar_significant": bool(reject[1]),
    }


def run_replicate_statistical_suite(
    proposed_rows: list[dict],
    baseline_rows: list[dict],
    alpha: float = 0.05,
) -> dict:
    """
    Runs paired replicate-level hypothesis tests across multiple independent benchmark datasets
    (Manuscript Section 4.3.1):
      - Paired Wilcoxon signed-rank across replicates for G_tree, Mean_F1, Mean_FGS, LCR, EDR
      - McNemar's test on replicate-level full coordinate verification (CCVR == 1.0)
      - Family-wise Benjamini-Hochberg FDR correction across all tested metrics.
    """
    if len(proposed_rows) != len(baseline_rows) or not proposed_rows:
        return {}

    continuous_keys = ["g_tree", "mean_feature_f1", "mean_fgs", "lcr", "edr"]
    raw_pvals: list[float] = []
    metric_names: list[str] = []
    stats_map: dict[str, float] = {}

    for k in continuous_keys:
        a_vals = np.array([float(r[k]) for r in proposed_rows], dtype=float)
        b_vals = np.array([float(r[k]) for r in baseline_rows], dtype=float)
        if np.allclose(a_vals - b_vals, 0.0):
            stat_v, p_v = 0.0, 1.0
        else:
            res = wilcoxon(a_vals, b_vals, zero_method="zsplit", alternative="two-sided")
            stat_v, p_v = float(res.statistic), float(res.pvalue)
        metric_names.append(f"wilcoxon_{k}")
        stats_map[f"wilcoxon_{k}"] = round(stat_v, 4)
        raw_pvals.append(p_v)

    a_ccvr_bin = np.array([1 if float(r["ccvr"]) >= 0.999 else 0 for r in proposed_rows], dtype=int)
    b_ccvr_bin = np.array([1 if float(r["ccvr"]) >= 0.999 else 0 for r in baseline_rows], dtype=int)
    n11 = int(np.sum((a_ccvr_bin == 1) & (b_ccvr_bin == 1)))
    n10 = int(np.sum((a_ccvr_bin == 1) & (b_ccvr_bin == 0)))
    n01 = int(np.sum((a_ccvr_bin == 0) & (b_ccvr_bin == 1)))
    n00 = int(np.sum((a_ccvr_bin == 0) & (b_ccvr_bin == 0)))
    if n10 + n01 == 0:
        m_stat, m_pval = 0.0, 1.0
    else:
        res_m = mcnemar([[n11, n10], [n01, n00]], exact=(n10 + n01 < 25))
        m_stat, m_pval = float(res_m.statistic), float(res_m.pvalue)
    metric_names.append("mcnemar_ccvr_perfect")
    stats_map["mcnemar_ccvr_perfect"] = round(m_stat, 4)
    raw_pvals.append(m_pval)

    reject, p_fdr, _, _ = multipletests(raw_pvals, alpha=alpha, method="fdr_bh")
    out: dict[str, Any] = {"n_replicates": len(proposed_rows)}
    for idx, name in enumerate(metric_names):
        out[name] = {
            "statistic": stats_map[name],
            "p_raw": round(float(raw_pvals[idx]), 6),
            "p_fdr_bh": round(float(p_fdr[idx]), 6),
            "significant": bool(reject[idx]),
        }
    return out
