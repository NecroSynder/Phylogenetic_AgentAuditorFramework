"""
Validation of the NumPy BIONJ implementation (Gascuel, 1997) used by the Phyloformer
normalizer (manuscript Section 4.1.1, item 4, Phyloformer Normalization).

1. Additive (tree-metric) matrices: BIONJ must recover the generating topology and
   reproduce every pairwise patristic distance exactly (Gascuel, 1997).
2. Reference comparison: topology (and, on additive data, branch lengths) must match the
   reference FastME BIONJ implementation (Lefort et al., 2015). Skipped if `fastme` is
   not installed (conda install -c bioconda fastme).
3. Input validation through scikit-bio DistanceMatrix (Aton et al., 2025).
"""
import random
import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from ete3 import Tree
from skbio import DistanceMatrix
from skbio.stats.distance import DissimilarityMatrixError  # parent of DistanceMatrixError

from src.preprocessing.normalizer_wrapper import distance_matrix_to_bionj_newick

FASTME = shutil.which("fastme", path=str(Path(sys.executable).parent)) or shutil.which("fastme")
requires_fastme = pytest.mark.skipif(
    FASTME is None, reason="FastME not installed (conda install -c bioconda fastme)"
)


# ----------------------------------------------------------------------------- helpers
def random_tree(n_taxa: int, seed: int) -> Tree:
    """Random binary tree with strictly positive branch lengths (deterministic per seed)."""
    random.seed(seed)
    t = Tree()
    t.populate(
        n_taxa,
        names_library=[f"T{i:02d}" for i in range(1, n_taxa + 1)],
        random_branches=True,
        branch_range=(0.02, 0.5),
    )
    return t


def patristic_matrix(tree: Tree, labels: list[str]) -> np.ndarray:
    """Pairwise path-length (additive) distance matrix of a tree."""
    n = len(labels)
    D = np.zeros((n, n))
    for a, b in combinations(range(n), 2):
        D[a, b] = D[b, a] = tree.get_distance(labels[a], labels[b])
    return D


def unrooted_rf(t1: Tree, t2: Tree) -> int:
    return t1.robinson_foulds(t2, unrooted_trees=True)[0]


def write_phylip_matrix(dm: DistanceMatrix, path) -> None:
    lines = [str(dm.shape[0])]
    for name, row in zip(dm.ids, dm.data):
        lines.append(name + " " + " ".join(f"{x:.10f}" for x in row))
    path.write_text("\n".join(lines) + "\n")


def run_fastme_bionj(dm: DistanceMatrix, workdir) -> Tree:
    """Runs the reference FastME BIONJ (-mB) on a PHYLIP distance matrix."""
    write_phylip_matrix(dm, workdir / "dm.phy")
    subprocess.run(
        [FASTME, "-idm.phy", "-oref.nwk", "-mB"],
        cwd=workdir, check=True, capture_output=True, text=True,
    )
    return Tree((workdir / "ref.nwk").read_text().strip(), format=1)


def noisy_matrix(D: np.ndarray, seed: int, sd: float = 0.02) -> np.ndarray:
    """Symmetric, non-additive perturbation of D (keeps a zero diagonal, positive entries)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sd, size=D.shape)
    noise = np.triu(noise, 1)
    noise = noise + noise.T
    return np.clip(D + noise, 1e-4, None) * (1 - np.eye(len(D)))


# ------------------------------------------------------------- 1. additive exactness
@pytest.mark.parametrize("n_taxa,seed", [(4, 1), (8, 2), (15, 3), (30, 4)])
def test_bionj_recovers_additive_tree_exactly(n_taxa, seed):
    true_tree = random_tree(n_taxa, seed)
    labels = sorted(true_tree.get_leaf_names())
    D = patristic_matrix(true_tree, labels)

    pred = Tree(distance_matrix_to_bionj_newick(DistanceMatrix(D, ids=labels)), format=1)

    assert unrooted_rf(true_tree, pred) == 0
    np.testing.assert_allclose(patristic_matrix(pred, labels), D, atol=1e-4)


# ---------------------------------------------------------- 2. FastME reference check
@requires_fastme
@pytest.mark.parametrize("n_taxa,seed", [(8, 11), (15, 12), (30, 13), (50, 14)])
def test_bionj_matches_fastme_on_additive_matrices(n_taxa, seed, tmp_path):
    true_tree = random_tree(n_taxa, seed)
    labels = sorted(true_tree.get_leaf_names())
    dm = DistanceMatrix(patristic_matrix(true_tree, labels), ids=labels)

    ours = Tree(distance_matrix_to_bionj_newick(dm), format=1)
    ref = run_fastme_bionj(dm, tmp_path)

    assert unrooted_rf(ours, ref) == 0
    np.testing.assert_allclose(
        patristic_matrix(ours, labels), patristic_matrix(ref, labels), atol=1e-4
    )


@requires_fastme
@pytest.mark.parametrize("n_taxa,seed", [(8, 21), (15, 22), (30, 23), (50, 24)])
def test_bionj_matches_fastme_topology_on_noisy_matrices(n_taxa, seed, tmp_path):
    """Non-additive matrices (as Phyloformer produces): topology must match FastME."""
    true_tree = random_tree(n_taxa, seed)
    labels = sorted(true_tree.get_leaf_names())
    dm = DistanceMatrix(noisy_matrix(patristic_matrix(true_tree, labels), seed), ids=labels)

    ours = Tree(distance_matrix_to_bionj_newick(dm), format=1)
    ref = run_fastme_bionj(dm, tmp_path)

    assert unrooted_rf(ours, ref) == 0


# ------------------------------------------------------- 3. scikit-bio input checks
def test_accepts_numpy_matrix_with_labels():
    D = np.array([
        [0.0, 0.1, 0.4, 0.5],
        [0.1, 0.0, 0.4, 0.5],
        [0.4, 0.4, 0.0, 0.1],
        [0.5, 0.5, 0.1, 0.0],
    ])
    pred = Tree(distance_matrix_to_bionj_newick(D, ["T1", "T2", "T3", "T4"]), format=1)
    assert sorted(pred.get_leaf_names()) == ["T1", "T2", "T3", "T4"]


def test_negative_branches_are_clamped_to_epsilon():
    # Violates the triangle inequality: raw BIONJ length for B = 0.5*(0.1+0.1-1.0) < 0
    D = np.array([
        [0.0, 0.1, 1.0],
        [0.1, 0.0, 0.1],
        [1.0, 0.1, 0.0],
    ])
    pred = Tree(distance_matrix_to_bionj_newick(D, ["A", "B", "C"]), format=1)
    assert all(n.dist >= 1e-6 for n in pred.traverse() if not n.is_root())
    assert (pred & "B").dist == pytest.approx(1e-6)


@pytest.mark.parametrize("bad", [
    np.array([[0.0, 0.1, 0.2], [0.3, 0.0, 0.2], [0.2, 0.2, 0.0]]),  # asymmetric
    np.array([[0.5, 0.1, 0.2], [0.1, 0.0, 0.2], [0.2, 0.2, 0.0]]),  # non-zero diagonal
])
def test_rejects_invalid_matrices(bad):
    with pytest.raises(DissimilarityMatrixError):
        distance_matrix_to_bionj_newick(bad, ["A", "B", "C"])


def test_rejects_duplicate_taxon_labels():
    D = np.array([[0.0, 0.1, 0.2], [0.1, 0.0, 0.2], [0.2, 0.2, 0.0]])
    with pytest.raises(DissimilarityMatrixError):
        distance_matrix_to_bionj_newick(D, ["A", "A", "B"])
