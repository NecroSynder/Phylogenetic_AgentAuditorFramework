"""
Multi-Dimensional Evaluation Metrics for Phase 1 Neuro-Symbolic Auditing
(Manuscript Section 4.3.2, Equations 4.1 - 4.15, and Statistical Validation Section 4.3.1).
"""
import re
import numpy as np
import pandas as pd
from Bio import AlignIO
from ete3 import Tree
from scipy.linalg import expm
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


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


def compute_feature_precision_recall_f1(
    cited_sites: set[int], true_sites: set[int]
) -> tuple[float, float, float]:
    """
    Computes Feature Identification Precision(b), Recall(b), and F1(b) (Section 4.3.2.B.1, Eqs. 4.5-4.6).
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


def compute_bsf_local(support_pct: float, explanation_text: str) -> int:
    """
    Computes binary Node Support Tier Fidelity BSF(b_k) in {0, 1} (Section 4.3.2.B.2, Eq. 4.7).
    Verifies that the natural-language confidence descriptor in explanation_text matches phi(Support(b_k)).
    """
    expected_tier = confidence_tier_label(support_pct)
    text_lower = explanation_text.lower()

    has_high = any(w in text_lower for w in ("definitive", "high confidence", "high support", "strong support", "high-confidence"))
    has_moderate = any(w in text_lower for w in ("moderate confidence", "moderate support", "moderate"))
    has_low = any(w in text_lower for w in ("low confidence", "low support", "weak support", "low-support"))

    if expected_tier == "Definitive/High":
        return 1 if (has_high and not has_low) else 0
    elif expected_tier == "Moderate":
        return 1 if (has_moderate and not has_high and not has_low) else 0
    else:
        return 1 if (has_low and not has_high) else 0


def compute_fgs_local(
    explanation_text: str,
    node_id: str,
    support_pct: float,
    synapomorphies: list[dict],
    stressor_diag: dict | None,
) -> float:
    """
    Computes Faithfulness / Context Grounding Score FGS(b) in [0, 1] (Section 4.3.2.B.4, Eq. 4.9):
      FGS(b) = |{s in S(b) | O_tools(b) |= s}| / |S(b)|
    Extracts atomic biological statements/citations from explanation_text and verifies each
    against the active tool output dictionary O_tools(b).
    """
    if not explanation_text or not explanation_text.strip():
        return 0.0

    # Split explanation into atomic clauses/sentences without breaking decimal numbers (e.g., 100.0%, CI = 1.0)
    clauses = [
        c.strip()
        for c in re.split(r";\s*|\.\s+(?=[A-Z]|$)", explanation_text.strip().rstrip("."))
        if c.strip()
    ]
    if not clauses:
        return 0.0

    valid_sites = {int(m["site"]) for m in synapomorphies}
    valid_Pairs = {(int(m["site"]), str(m["parent_state"]), str(m["child_state"])) for m in synapomorphies}
    valid_artifact = stressor_diag["primary_artifact"] if stressor_diag else "NONE"

    entailed_count = 0
    for clause in clauses:
        clause_ok = True
        # Check any explicit site numbers cited via "site <int>" or "<State><int><State>"
        cited_sites_in_clause = [int(x) for x in re.findall(r"\bsite\s+(\d+)\b", clause, flags=re.IGNORECASE)]
        for s_num in cited_sites_in_clause:
            if s_num not in valid_sites:
                clause_ok = False
                break

        # Check compact mutation tokens like G4C or C128A
        mut_tokens = re.findall(r"\b([ACGT])-?(\d+)-?>?([ACGT])\b", clause)
        for s_u, s_pos, s_v in mut_tokens:
            if (int(s_pos), s_u, s_v) not in valid_Pairs:
                clause_ok = False
                break

        # Check support percentage if cited
        pct_tokens = re.findall(r"(\d+(?:\.\d+)?)%\s*(?:resampled\s+)?support", clause, flags=re.IGNORECASE)
        for pct_str in pct_tokens:
            if abs(float(pct_str) - support_pct) > 1.0:
                clause_ok = False
                break

        # Ensure no contradicted artifact label is claimed when valid_artifact is NONE
        if valid_artifact == "NONE" and "artifact of lba" in clause.lower():
            clause_ok = False

        if clause_ok:
            entailed_count += 1

    return round(entailed_count / len(clauses), 4)


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
    """
    # 1. Continuous-Time Markov Chain (JC69 / symmetric rate matrix Q) transition matrix P = exp(Q * l_k)
    eff_len = max(float(branch_length), 1e-4)
    rate_off = 1.0 / 3.0
    Q = np.full((4, 4), rate_off)
    np.fill_diagonal(Q, -1.0)
    P_mat = expm(Q * eff_len)
    state_idx = {"A": 0, "C": 1, "G": 2, "T": 3}

    loglik_sum = 0.0
    for m in true_branch_mutations:
        u_s = str(m.get("parent_state", "A"))
        v_s = str(m.get("child_state", "C"))
        if u_s in state_idx and v_s in state_idx:
            prob = max(float(P_mat[state_idx[u_s], state_idx[v_s]]), 1e-12)
            loglik_sum += float(np.log(prob))
        else:
            # Indel transition log-likelihood approximation
            loglik_sum += float(np.log(max(eff_len / (1.0 + eff_len), 1e-6)))

    if not true_branch_mutations:
        loglik_sum = round(float(np.log(max(float(P_mat[0, 0]), 1e-6))), 4)

    # 2. Clade Shannon entropy reduction (bits) across causal sites
    alignment = AlignIO.read(fasta_path, "fasta")
    n_total = len(alignment)
    clade_set = set(clade_taxa)

    def _col_entropy(chars: list[str]) -> float:
        if not chars:
            return 0.0
        _, counts = np.unique(chars, return_counts=True)
        probs = counts / len(chars)
        return float(-np.sum(probs * np.log2(probs)))

    entropy_reductions = []
    for m in true_branch_mutations:
        col_idx = int(m["site"]) - 1
        if 0 <= col_idx < alignment.get_alignment_length():
            all_chars = [rec.seq[col_idx] for rec in alignment]
            in_chars = [rec.seq[col_idx] for rec in alignment if rec.id in clade_set]
            out_chars = [rec.seq[col_idx] for rec in alignment if rec.id not in clade_set]
            h_total = _col_entropy(all_chars)
            h_cond = (len(in_chars) / n_total) * _col_entropy(in_chars) + (
                len(out_chars) / n_total
            ) * _col_entropy(out_chars)
            entropy_reductions.append(max(h_total - h_cond, 0.0))

    entropy_red_bits = round(float(np.sum(entropy_reductions)), 4) if entropy_reductions else 0.0

    # 3. Normalized Information Gain IG(b_k) (Eq. 4.12)
    true_sls_by_site = {
        int(m["site"]): max(float(m.get("delta_sls", 0.0)), 0.0)
        for m in true_branch_mutations
    }
    total_sls = sum(true_sls_by_site.values())
    cited_sites = {
        int(c.get("site_coordinate", c.get("site", -1))) for c in cited_mutations
    }
    captured_sls = sum(
        sls for site, sls in true_sls_by_site.items() if site in cited_sites
    )

    if total_sls > 0.0:
        ig_score = min(round(captured_sls / total_sls, 4), 1.0)
    else:
        ig_score = 1.0 if not cited_sites else 0.0

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
    weighted_sum = (a1 * feature_f1) + (a2 * information_gain_ig) + (a3 * faithfulness_fgs)
    score = float(ccvr_local) * float(bsf_local) * weighted_sum
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def compute_lcr(node_audits: list[dict], pred_newick_path: str) -> float:
    """
    Computes global Logical Consistency Rate LCR(T_pred) in [0, 1] (Section 4.3.2.B.5, Eq. 4.10):
      LCR = 1 - |{(p_a, p_b) in P_tree x P_tree | p_a ^ p_b => False}| / |P_tree|
    Checks cross-node clade hierarchy predicates and topological status vs support consistency.
    """
    if not node_audits:
        return 1.0

    t_pred = Tree(pred_newick_path, format=1)
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
        status = rec["topological_status"]

        # Predicate 1: MonophyleticClade(nid, claimed_clade) matches actual tree split
        total_predicates += 1
        if nid in true_tree_clades and claimed_clade != true_tree_clades[nid]:
            contradictions += 1

        # Predicate 2: Topological status vs support threshold consistency
        total_predicates += 1
        if status == "FALSE_CONFIDENCE_ERROR" and sup < 85.0:
            contradictions += 1
        elif status == "DISCORDANT_LOW_SUPPORT" and sup >= 85.0:
            contradictions += 1

    # Pairwise Laminar Tree Hierarchy Predicate check: any two clades on a rooted tree
    # must be either disjoint or one a subset of the other (never partially overlapping)
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
) -> float:
    """
    Computes Reconciled Tree-Level Explanation Goodness Score G_tree(T_pred) in [0, 1]
    (Section 4.3.2.C.4, Eq. 4.15):
      G_tree(T_pred) = LCR(T_pred) * sum_{k=1}^K w_k * G_node(b_k)
      where w_k = I_branch(b_k) / sum_j I_branch(b_j), I_branch(b_k) = sum_{i in M_true(b_k)} delta_SLS_i + epsilon
    """
    if not node_audits:
        return 0.0

    m_df = pd.read_csv(m_true_tsv_path, sep="\t", keep_default_na=False)
    weights = []
    node_scores = []

    for rec in node_audits:
        nid = rec["node_id"]
        sub = m_df[m_df["child_node"] == nid]
        i_branch = float(sub["delta_sls"].astype(float).clip(lower=0.0).sum()) + epsilon
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


def run_statistical_validation(
    agent_continuous: list[float],
    baseline_continuous: list[float],
    agent_binary: list[int],
    baseline_binary: list[int],
    alpha: float = 0.05,
) -> dict:
    """
    Executes paired two-sided Wilcoxon signed-rank test (for continuous metrics: RF_norm, FCI, F1, FGS)
    and McNemar's test (for binary verification proportions: CCVR, BSF) with Benjamini-Hochberg
    FDR correction at alpha = 0.05 (Section 4.3.1, page 26).
    """
    diffs = np.asarray(agent_continuous, dtype=float) - np.asarray(baseline_continuous, dtype=float)
    if np.allclose(diffs, 0.0):
        w_stat, w_pval = 0.0, 1.0
    else:
        res_w = wilcoxon(agent_continuous, baseline_continuous, zero_method="zsplit", alternative="two-sided")
        w_stat, w_pval = float(res_w.statistic), float(res_w.pvalue)

    # Build 2x2 contingency table for McNemar's test
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

    reject, pvals_corrected, _, _ = multipletests([w_pval, m_pval], alpha=alpha, method="fdr_bh")
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
