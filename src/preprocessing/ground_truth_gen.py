import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import AlignIO
from ete3 import Tree

def fitch_parsimony_steps(tree: Tree, site_states: dict) -> tuple[int, int, int]:
    """
    Computes observed parsimony steps (s_i), minimum steps (m_i),
    and maximum star-tree steps (g_i) for a single alignment column.
    """
    unique_states = set(site_states.values()) - {"-", "?", "N"}
    if len(unique_states) <= 1:
        return 0, 0, 0
    m_i = len(unique_states) - 1
    state_counts = [list(site_states.values()).count(s) for s in unique_states]
    g_i = sum(state_counts) - max(state_counts)

    postorder_sets = {}
    steps = 0
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            ch = site_states.get(node.name, "-")
            postorder_sets[node] = {ch} if ch not in {"-", "?", "N"} else unique_states.copy()
        else:
            child_sets = [postorder_sets[c] for c in node.children]
            intersect = set.intersection(*child_sets)
            if intersect:
                postorder_sets[node] = intersect
            else:
                postorder_sets[node] = set.union(*child_sets)
                steps += 1
    return steps, m_i, g_i

def fitch_binary_gap_states(tree: Tree, binary_states: dict[str, int]) -> tuple[dict[str, int], int, int, int]:
    """
    Runs two-pass Fitch Maximum Parsimony (post-order bottom-up + pre-order top-down)
    on a binary gap-coded column (0 = present character, 1 = alignment gap '-')
    per Section 4.1.1 Item 2.2. Returns deterministic internal/leaf node states {0, 1}
    alongside (s_i, m_i, g_i).
    """
    vals = list(binary_states.values())
    unique_states = set(vals)
    if len(unique_states) <= 1:
        only = next(iter(unique_states)) if unique_states else 0
        return {n.name: only for n in tree.traverse() if n.name}, 0, 0, 0

    m_i = len(unique_states) - 1
    state_counts = [vals.count(s) for s in unique_states]
    g_i = sum(state_counts) - max(state_counts)

    postorder_sets = {}
    steps = 0
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            postorder_sets[node] = {binary_states.get(node.name, 0)}
        else:
            child_sets = [postorder_sets[c] for c in node.children]
            intersect = set.intersection(*child_sets)
            if intersect:
                postorder_sets[node] = intersect
            else:
                postorder_sets[node] = set.union(*child_sets)
                steps += 1

    assigned: dict[str, int] = {}
    for node in tree.traverse("preorder"):
        cand = postorder_sets[node]
        if node.is_root():
            state = 0 if 0 in cand else min(cand)
        else:
            parent_state = assigned[node.up.name]
            state = parent_state if parent_state in cand else min(cand)
        if node.name:
            assigned[node.name] = state

    return assigned, steps, m_i, g_i

def shannon_mutual_information_gap_bits(binary_states: dict[str, int], clade_taxa: set[str]) -> float:
    """
    Computes Shannon mutual information I(X_i; b_k) = H(X_i) - H(X_i | b_k) in bits
    for a binary gap-coded alignment column X_i partitioned by branch b_k (Section 4.3.2.C.1).
    """
    all_taxa = list(binary_states.keys())
    n_total = len(all_taxa)
    if n_total == 0:
        return 0.0

    def _entropy(vals: list[int]) -> float:
        if not vals:
            return 0.0
        n = len(vals)
        p1 = sum(vals) / n
        p0 = 1.0 - p1
        h = 0.0
        if p0 > 0.0:
            h -= p0 * np.log2(p0)
        if p1 > 0.0:
            h -= p1 * np.log2(p1)
        return float(h)

    all_vals = [binary_states[t] for t in all_taxa]
    in_vals = [binary_states[t] for t in all_taxa if t in clade_taxa]
    out_vals = [binary_states[t] for t in all_taxa if t not in clade_taxa]

    h_x = _entropy(all_vals)
    h_cond = (len(in_vals) / n_total) * _entropy(in_vals) + (len(out_vals) / n_total) * _entropy(out_vals)
    return max(round(h_x - h_cond, 4), 0.0)

def run_ground_truth_pipeline(
    fasta_path: str,
    true_tree_path: str,
    out_dir: str,
    outgroup: str = "Taxon_01",
    recombination_breakpoints: list | None = None,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Universal Outgroup Rooting & Deterministic Post-Order Node Labeling on T_true
    t_true = Tree(true_tree_path, format=1)
    t_true.set_outgroup(t_true & outgroup)
    k_idx = 1
    for node in t_true.traverse("postorder"):
        if not node.is_leaf():
            node.name = f"N{k_idx}"
            k_idx += 1
    rooted_true_path = out_path / "T_true_rooted.nwk"
    t_true.write(format=1, outfile=str(rooted_true_path))

    # 2. Run IQ-TREE 2 Empirical Bayesian ASR (--ancestral) and site log-likelihoods (-wsl)
    prefix = out_path / "iqtree_asr"
    iqtree_bin = (
        shutil.which("iqtree2", path=str(Path(sys.executable).parent))
        or shutil.which("iqtree2")
        or "iqtree2"
    )
    cmd = [
        iqtree_bin, "-s", fasta_path, "-te", str(rooted_true_path),
        "-m", "GTR+G4", "--ancestral", "-wsl", "-pre", str(prefix), "-redo", "-quiet"
    ]
    subprocess.run(cmd, check=True)

    # 3. Parse .state file for marginal posterior probabilities P(X_{u,i} = s | T_true, D) >= 0.90
    state_df = pd.read_csv(f"{prefix}.state", sep="\t", comment="#")
    state_probs = [c for c in state_df.columns if c.startswith("p_")]
    state_df["max_posterior"] = state_df[state_probs].max(axis=1)

    # 4. Compute Fitch Parsimony CI_i, RI_i and Indel Gap Density across columns
    alignment = AlignIO.read(fasta_path, "fasta")
    seq_len = alignment.get_alignment_length()
    n_taxa = len(alignment)

    site_metrics = {}
    gap_node_states_per_site = {}
    gap_binary_per_site = {}
    for col_idx in range(seq_len):
        col_str = alignment[:, col_idx]
        site_1based = col_idx + 1
        gap_pct = (col_str.count("-") / n_taxa) * 100.0
        char_map = {rec.id: rec.seq[col_idx] for rec in alignment}
        s_i, m_i, g_i = fitch_parsimony_steps(t_true, char_map)
        ci_i = (m_i / s_i) if s_i > 0 else 1.0
        ri_i = ((g_i - s_i) / (g_i - m_i)) if (g_i > m_i and s_i > 0) else 1.0

        # Binary gap-coded matrix (0 = present nucleotide/AA, 1 = alignment gap '-')
        bin_map = {rec.id: (1 if rec.seq[col_idx] == "-" else 0) for rec in alignment}
        gap_assigned, gap_s, gap_m, _ = fitch_binary_gap_states(t_true, bin_map)
        gap_ci = (gap_m / gap_s) if gap_s > 0 else 1.0
        gap_node_states_per_site[site_1based] = gap_assigned
        gap_binary_per_site[site_1based] = bin_map

        site_metrics[site_1based] = {
            "ci_i": round(ci_i, 4),
            "ri_i": round(ri_i, 4),
            "gap_pct": round(gap_pct, 2),
            "gap_ci_i": round(gap_ci, 4)
        }

    # 5. Build M_true (Causal Mutations per branch u -> v: ASR substitutions + Fitch Indel synapomorphies)
    node_state_lookup = state_df.set_index(["Node", "Site"]).to_dict(orient="index")
    m_true_records = []
    indel_boundaries_by_branch = {}

    for node in t_true.traverse("postorder"):
        if node.is_root() or node.is_leaf():
            continue
        u_name = node.up.name
        v_name = node.name
        clade_taxa = set(node.get_leaf_names())

        # 5a. Contiguous indel boundaries [i_start, i_end] supporting branch u -> v (Section 4.1.1 Item 2.2)
        indel_sites = []
        for site_i in range(1, seq_len + 1):
            g_u = gap_node_states_per_site[site_i].get(u_name, 0)
            g_v = gap_node_states_per_site[site_i].get(v_name, 0)
            if g_u != g_v:
                indel_sites.append((site_i, g_u, g_v))

        # Group contiguous indel sites into [i_start, i_end] tracts
        site_to_tract = {}
        branch_tracts = []
        idx = 0
        while idx < len(indel_sites):
            start_s, g_u0, g_v0 = indel_sites[idx]
            end_s = start_s
            while (
                idx + 1 < len(indel_sites)
                and indel_sites[idx + 1][0] == end_s + 1
                and indel_sites[idx + 1][1] == g_u0
                and indel_sites[idx + 1][2] == g_v0
            ):
                idx += 1
                end_s = indel_sites[idx][0]
            tract_label = f"{start_s}-{end_s}"
            branch_tracts.append({
                "i_start": start_s,
                "i_end": end_s,
                "tract_length": end_s - start_s + 1,
                "event": "DELETION" if (g_u0 == 0 and g_v0 == 1) else "INSERTION"
            })
            for s_pos in range(start_s, end_s + 1):
                site_to_tract[s_pos] = (g_u0, g_v0, tract_label)
            idx += 1
        if branch_tracts:
            indel_boundaries_by_branch[f"{u_name}->{v_name}"] = branch_tracts

        for site_i in range(1, seq_len + 1):
            # Indel synapomorphy takes precedence at gap-transition columns
            if site_i in site_to_tract:
                g_u, g_v, tract_str = site_to_tract[site_i]
                mi_bits = shannon_mutual_information_gap_bits(gap_binary_per_site[site_i], clade_taxa)
                m_true_records.append({
                    "parent_node": u_name,
                    "child_node": v_name,
                    "site": site_i,
                    "parent_state": "-" if g_u == 1 else "CHAR",
                    "child_state": "-" if g_v == 1 else "CHAR",
                    "posterior_prob": 1.0,
                    "delta_sls": max(mi_bits, 1e-4),
                    "is_indel": True,
                    "ci_score": site_metrics[site_i]["gap_ci_i"],
                    "indel_boundary": tract_str
                })
                continue

            u_row = node_state_lookup.get((u_name, site_i))
            v_row = node_state_lookup.get((v_name, site_i))
            if u_row and v_row:
                s_u, p_u = u_row["State"], u_row["max_posterior"]
                s_v, p_v = v_row["State"], v_row["max_posterior"]
                if s_u != s_v and p_u >= 0.90 and p_v >= 0.90:
                    m_true_records.append({
                        "parent_node": u_name,
                        "child_node": v_name,
                        "site": site_i,
                        "parent_state": s_u,
                        "child_state": s_v,
                        "posterior_prob": round(min(p_u, p_v), 4),
                        "delta_sls": round(float(np.log(min(p_u, p_v) / 0.25)), 4),
                        "is_indel": False,
                        "ci_score": site_metrics[site_i]["ci_i"],
                        "indel_boundary": ""
                    })

    m_true_df = pd.DataFrame(m_true_records, columns=[
        "parent_node", "child_node", "site", "parent_state", "child_state",
        "posterior_prob", "delta_sls", "is_indel", "ci_score", "indel_boundary"
    ])
    m_true_df.to_csv(out_path / "M_true.tsv", sep="\t", index=False)

    # 6. Build A_true (Homoplasy, LBA, Indel Boundaries & Recombination Stressor Ground Truth)
    term_dists = [leaf.dist for leaf in t_true.iter_leaves()]
    int_dists = [n.dist for n in t_true.traverse() if not n.is_leaf() and not n.is_root() and n.dist > 0]
    rho_lba = (max(term_dists) / min(int_dists)) if int_dists else 1.0

    # Contiguous alignment-wide gap tracts L_gap (Table 4.1)
    gap_cols = [s for s, m in site_metrics.items() if m["gap_pct"] > 0.0]
    gap_tracts = []
    g_idx = 0
    while g_idx < len(gap_cols):
        g_start = gap_cols[g_idx]
        g_end = g_start
        while g_idx + 1 < len(gap_cols) and gap_cols[g_idx + 1] == g_end + 1:
            g_idx += 1
            g_end = gap_cols[g_idx]
        gap_tracts.append({"i_start": g_start, "i_end": g_end, "l_gap": g_end - g_start + 1})
        g_idx += 1

    a_true = {
        "global_rho_lba": round(rho_lba, 3),
        "mean_alignment_ci": round(float(np.mean([v["ci_i"] for v in site_metrics.values()])), 4),
        "mean_alignment_ri": round(float(np.mean([v["ri_i"] for v in site_metrics.values()])), 4),
        "mean_gap_density_pct": round(float(np.mean([v["gap_pct"] for v in site_metrics.values()])), 2),
        "max_gap_tract_length": max((t["l_gap"] for t in gap_tracts), default=0),
        "gap_tracts": gap_tracts,
        "branch_indel_synapomorphies": indel_boundaries_by_branch,
        "recombination_breakpoints": list(recombination_breakpoints) if recombination_breakpoints else [],
        "site_metrics": site_metrics
    }
    with open(out_path / "A_true.json", "w") as f:
        json.dump(a_true, f, indent=2)

if __name__ == "__main__":
    run_ground_truth_pipeline(
        "data/00_golden_toy/toy_alignment.fa",
        "data/00_golden_toy/T_true.nwk",
        "data/02_ground_truths/toy"
    )