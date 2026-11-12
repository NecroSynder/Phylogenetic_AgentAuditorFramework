import json
import subprocess
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

def run_ground_truth_pipeline(
    fasta_path: str, true_tree_path: str, out_dir: str, outgroup: str = "Taxon_01"
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
    cmd = [
        "iqtree2", "-s", fasta_path, "-te", str(rooted_true_path),
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
    for col_idx in range(seq_len):
        col_str = alignment[:, col_idx]
        site_1based = col_idx + 1
        gap_pct = (col_str.count("-") / n_taxa) * 100.0
        char_map = {rec.id: rec.seq[col_idx] for rec in alignment}
        s_i, m_i, g_i = fitch_parsimony_steps(t_true, char_map)
        ci_i = (m_i / s_i) if s_i > 0 else 1.0
        ri_i = ((g_i - s_i) / (g_i - m_i)) if (g_i > m_i and s_i > 0) else 1.0
        site_metrics[site_1based] = {
            "ci_i": round(ci_i, 4),
            "ri_i": round(ri_i, 4),
            "gap_pct": round(gap_pct, 2)
        }

    # 5. Build M_true (Causal Mutations per branch u -> v)
    node_state_lookup = state_df.set_index(["Node", "Site"]).to_dict(orient="index")
    m_true_records = []

    for node in t_true.traverse("postorder"):
        if node.is_root() or node.is_leaf():
            continue
        u_name = node.up.name
        v_name = node.name
        for site_i in range(1, seq_len + 1):
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
                        "ci_score": site_metrics[site_i]["ci_i"]
                    })

    m_true_df = pd.DataFrame(m_true_records)
    m_true_df.to_csv(out_path / "M_true.tsv", sep="\t", index=False)

    # 6. Build A_true (Homoplasy & LBA Stressor Ground Truth)
    term_dists = [leaf.dist for leaf in t_true.iter_leaves()]
    int_dists = [n.dist for n in t_true.traverse() if not n.is_leaf() and not n.is_root() and n.dist > 0]
    rho_lba = (max(term_dists) / min(int_dists)) if int_dists else 1.0

    a_true = {
        "global_rho_lba": round(rho_lba, 3),
        "mean_alignment_ci": round(float(np.mean([v["ci_i"] for v in site_metrics.values()])), 4),
        "mean_alignment_ri": round(float(np.mean([v["ri_i"] for v in site_metrics.values()])), 4),
        "mean_gap_density_pct": round(float(np.mean([v["gap_pct"] for v in site_metrics.values()])), 2),
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