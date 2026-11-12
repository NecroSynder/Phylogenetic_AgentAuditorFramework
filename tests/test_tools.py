from src.preprocessing.ground_truth_gen import run_ground_truth_pipeline
from src.preprocessing.normalizer_wrapper import standardize_and_root_tree
from src.tools.bio_tools import audit_tree_topology, verify_claim_coordinates

def test_pipeline_and_tools():
    run_ground_truth_pipeline(
        "data/00_golden_toy/toy_alignment.fa",
        "data/00_golden_toy/T_true.nwk",
        "data/02_ground_truths/toy"
    )
    with open("data/02_ground_truths/toy/T_true_rooted.nwk") as f:
        nwk_str = f.read().strip()

    standardize_and_root_tree(
        nwk_str, [nwk_str]*10, "Taxon_01", "Fusang", "data/03_normalized_preds/toy_fusang"
    )
    res = audit_tree_topology(
        "data/03_normalized_preds/toy_fusang_normalized.nwk",
        "data/02_ground_truths/toy/T_true_rooted.nwk",
        "data/03_normalized_preds/toy_fusang_metadata.json"
    )
    assert res["normalized_rf_distance"] == 0.0