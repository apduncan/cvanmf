from typing import List, Tuple, Set

import numpy as np
import pandas as pd

from cvanmf import models
from cvanmf.combine import align_signatures, compare_signatures, match_signatures, combine_signatures
from cvanmf.denovo import NMFParameters, decompositions
from cvanmf.models import Signatures


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


def test_combine_signatures():
    # Temp dev testing
    # Combine signatures with themselves
    # combine_signatures([models.five_es(), models.five_es()])
    dupd: pd.DataFrame = models.five_es().w
    dupd.loc[:, "Dupd_1"] = dupd.iloc[:, 0]
    dupd.loc[:, "Merg_1"] = dupd.iloc[:, 0] + dupd.iloc[:, 1]
    dupd.loc[:, "Rand_1"] = np.random.uniform(size=(dupd.shape[0], 1))
    dupd.loc[:, "Merg_2"] = dupd.iloc[:, 1] + 4 * dupd.iloc[:, 4]
    combine_signatures([models.five_es(), dupd])

# def test_experiment():
#     # Temp function - see how splitting the DRAMA data goes
#     es_x = pd.read_csv(
#         ("https://gitlab.inria.fr/cfrioux/enterosignature-paper/-/raw/main/data/"
#          "GMR_dataset/GMR_genus_level_abundance_normalised.tsv?ref_type=heads"),
#         sep="\t").iloc[:, :]
#     cohorts_arr = np.random.randint(low=0, high=32, size=(es_x.shape[1]))
#     cohorts = []
#     for i in set(cohorts_arr):
#         cohorts.append(es_x.iloc[:, cohorts_arr == i])
#     mmodels = [decompositions(x,
#                              ranks=[5],
#                              random_starts=20,
#                              top_n=1
#                              )[5][0] for x in cohorts]
#     combi = combine_signatures(mmodels, x=es_x, low_support_threshold=5)
#     matched = match_signatures(combi, models.five_es().w)
#     from cvanmf.reapply import _reapply_model
#     from cvanmf.reapply import match_identical
#     combi_model_fit = _reapply_model(y=es_x,
#                                      w=combi,
#                                      colors=None,
#                                      input_validation=lambda x: x,
#                                      feature_match=match_identical)
#     orig_model_fit = models.five_es().reapply(es_x)
#     print(matched)
