import itertools
import pathlib
from typing import List, Tuple, Set, Dict

import matplotlib
import numpy as np
import pandas as pd
import plotnine
import pytest

import cvanmf.models
from cvanmf import models
from cvanmf.combine import combine_signatures, Cohort, Model, \
    Signature, Combiner, Cluster, split_dataframe_to_cohorts
from cvanmf.data import synthetic_blocks
from cvanmf.stability import compare_signatures, align_signatures, \
    match_signatures, signature_stability, plot_signature_stability
from cvanmf.denovo import NMFParameters, decompositions, Decomposition
from cvanmf.models import Signatures


@pytest.fixture
def small_overlap_blocks(scope="session") -> pd.DataFrame:
    """Small overlapping block diagonal matrix with k=4, for use in testing
    de-novo methods."""
    return synthetic_blocks(100, 100, 0.25, 3,
                                              scale_lognormal_params=True).data


@pytest.fixture
def small_decompositions_random(
        small_overlap_blocks,
        scope="session"
) -> Dict[int, List[Decomposition]]:
    """Get 'best' decompositions for a small dataset from random
    initialisations."""
    res = decompositions(
        x=small_overlap_blocks,
        random_starts=5,
        ranks=[3, 4, 5],
        top_n=5,
        top_criteria="cosine_similarity",
        progress_bar=False
    )
    return res


@pytest.fixture(scope="module", autouse=True)
def pyplot_backend():
    matplotlib.pyplot.switch_backend("Agg")


def test_align_signatures():
    """Makes signatures have matched indices."""

    # Start by testing with two DataFrames with simply named indices
    shared_f: List[str] = [f'shared_{x}' for x in range(20)]
    unique_a: List[str] = [f'unique_a{x}' for x in range(10)]
    unique_b: List[str] = [f'unique_b{x}' for x in range(15)]

    sigs_a: pd.DataFrame = pd.DataFrame(
        np.random.uniform(
            size=(len(shared_f) + len(unique_a), 4)
        ),
        index=shared_f + unique_a
    )
    sigs_b: pd.DataFrame = pd.DataFrame(
        np.random.uniform(
            size=(len(shared_f) + len(unique_b), 4)
        ),
        index=shared_f + unique_b
    )

    a, b = align_signatures(sigs_a, sigs_b)
    expected_dimensions: Tuple[int, int] = (
        len(shared_f) + len(unique_a) + len(unique_b), 4
    )

    # Check dimensions
    assert a.shape == expected_dimensions, "Aligned A is incorrect shape"
    assert b.shape == expected_dimensions, "Aligned B is incorrect shape"

    # Check index in same order
    assert (a.index == b.index).all(), \
        "Aligned indices are not in the same order"

    # Check index contains all required values
    expected_index_set: Set[str] = set(shared_f + unique_b + unique_a)
    assert set(a.index) == expected_index_set, "Aligned A index incorrect"
    assert set(b.index) == expected_index_set, "Aligned B index incorrect"

    # All the keys unique to a should be 0 valued in aligned b
    assert (b.loc[unique_a, :] == 0.0).all().all(), \
        "Features unique to A are not 0 valued in B."
    # and the features in B should have equal values
    assert (b.loc[shared_f + unique_b, :] ==
            sigs_b.loc[shared_f + unique_b, :]).all().all(), \
        "Features in original B don't have matching values in aligned B"
    # Vice-versa for b
    assert (a.loc[unique_b, :] == 0.0).all().all(), \
        "Features unique to B are not 0 valued in A."
    assert (a.loc[shared_f + unique_a, :] ==
            sigs_a.loc[shared_f + unique_a, :]).all().all(), \
        "Features in original A don't have matching values in aligned A"

    # Test that this will run also with an iterable of Comparables
    it_a, it_b = align_signatures([sigs_a, sigs_b])
    assert np.array_equiv(it_a, a)
    assert np.array_equiv(it_b, b)


def test_compare_signatures():
    """Compare signatures between models."""

    # Start with simple case, compare a model to itself
    five_a: Signatures = models.five_es()

    sim_self: pd.DataFrame = compare_signatures(five_a, five_a)
    assert np.isclose(np.diag(sim_self.values), 1.0).all(), \
        "Similarity of signature to itself is not 1.0."
    assert sim_self.shape == (five_a.w.shape[1], five_a.w.shape[1])


def test_match_signatures():
    """Match signatures to maximise cosine similarity."""
    # Start with simple case, compare a model to itself
    five_a: Signatures = models.five_es()

    matched: pd.DataFrame = match_signatures(five_a, five_a)

    # Add some random signatures to make sure output sensible when
    # rank does not match
    seven: pd.DataFrame = models.five_es().w
    seven['Rand_1'] = seven.sample(frac=1).iloc[:, 0].values
    seven['Rand_2'] = seven.sample(frac=1).iloc[:, 1].values
    matched_uneven: pd.DataFrame = match_signatures(
        five_a.w.sample(axis="columns", frac=1),
        seven.sample(axis="columns", frac=1)
    )


def test_signature_stability(small_decompositions_random):
    """Ensure that signature stability runs. """

    # Comparing single rank
    stability = signature_stability(
        small_decompositions_random[3],
        small_decompositions_random[3][0]
    )

    assert (
            stability['model'].nunique() ==
            len(small_decompositions_random[3]) - 1
    ), "Incorrect number of models considered in stability."
    assert not stability[['a', 'b']].isna().any().any(), \
        "NA in signature pairings."

    # Using decompositions() output
    stability_d: pd.DataFrame = signature_stability(
        small_decompositions_random)


def test_plot_signature_stability(
        small_decompositions_random,
        tmp_path
):
    """Plotting for signature stability."""

    stability: pd.DataFrame = signature_stability(
        small_decompositions_random
    )
    plt_stability: plotnine.ggplot = plot_signature_stability(
        stability,
        colors=small_decompositions_random[3][0].colors,
        geom_line=None
    )
    pth = tmp_path / "stability_fig.png"
    plt_stability.save(pth)

    assert pth.exists(), "Failed to save plot."