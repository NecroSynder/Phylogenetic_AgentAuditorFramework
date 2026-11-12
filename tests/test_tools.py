import numpy as np
from src.preprocessing.ground_truth_gen import run_ground_truth_pipeline
from src.preprocessing.normalizer_wrapper import (
    standardize_and_root_tree,
    distance_matrix_to_bionj_newick
)
from src.tools.bio_tools import audit_tree_topology, verify_claim_coordinates

def test_pipeline_and_tools():
    # 1. Generate Ground Truths (T_true, M_true, A_true)
    run_ground_truth_pipeline(
        "data/00_golden_toy/toy_alignment.fa",
        "data/00_golden_toy/T_true.nwk",
        "data/02_ground_truths/toy"
    )
    with open("data/02_ground_truths/toy/T_true_rooted.nwk") as f:
        nwk_str = f.read().strip()

    # 2. Test Fusang Normalizer Wrapper
    standardize_and_root_tree(
        nwk_str, [nwk_str]*10, "Taxon_01", "Fusang", "data/03_normalized_preds/toy_fusang"
    )

    # 3. Test Phyloformer BIONJ Distance Matrix Normalizer
    dummy_dist = np.array([
        [0.0, 0.1, 0.4, 0.5],
        [0.1, 0.0, 0.4, 0.5],
        [0.4, 0.4, 0.0, 0.1],
        [0.5, 0.5, 0.1, 0.0]
    ])
    bionj_nwk = distance_matrix_to_bionj_newick(dummy_dist, ["T1", "T2", "T3", "T4"])
    assert "T1:" in bionj_nwk and ";" in bionj_nwk

    # 4. Test Tool 1: audit_tree_topology
    res = audit_tree_topology(
        "data/03_normalized_preds/toy_fusang_normalized.nwk",
        "data/02_ground_truths/toy/T_true_rooted.nwk",
        "data/03_normalized_preds/toy_fusang_metadata.json"
    )
    assert res["normalized_rf_distance"] == 0.0
    print("✅ All Ground-Truth, Normalizer, BIONJ, and Tool 1 checks passed!")

if __name__ == "__main__":
    test_pipeline_and_tools()