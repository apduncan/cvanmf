import itertools
import pathlib
from typing import List, Tuple, Set, Dict

import matplotlib
import numpy as np
import pandas as pd
import pytest

import cvanmf.models
from cvanmf import models
from cvanmf.combine import combine_signatures, Cohort, Model, \
    Signature, Combiner, Cluster, split_dataframe_to_cohorts
from cvanmf.stability import compare_signatures, align_signatures, \
    match_signatures
from cvanmf.denovo import NMFParameters, decompositions, Decomposition
from cvanmf.models import Signatures


@pytest.fixture(scope="module", autouse=True)
def pyplot_backend():
    matplotlib.pyplot.switch_backend("Agg")


@pytest.fixture(scope="session")
def complex_cohort_structure() -> Dict[str, pd.DataFrame]:
    """Make 5 cohorts of data with a complex structure.

    This set of cohorts is suitable to evaluate whether we are capturing the
    different kind of situations we expect during signature combining. These
    are:
    * highly similar signatures which should be merged
    * signatures which are linear combinations of others which should be removed
    * low support signatures (present in few cohorts) which can be
    *   *   uninformative, in which case removed
    *   *   informative cohort specific, in which case retained

    To this end, we make 5 cohorts based on the 5 ES model signatures.
    1.  Contains all 5 ES
    2.  Doesn't contain ES_Bifi
    3.  Contains an extra signature IS_1 which is informative
    4.  Completely shuffled, does not fit at all
    5.  Doesn't contain ES_Bact or ES_Esch
    6.  Merged ES_Firm and ES_Prev
    """
    # C1 - All ES
    es_w: pd.DataFrame = models.five_es().w
    c1_w: pd.DataFrame = es_w.copy()
    c1_h: pd.DataFrame = pd.DataFrame(
        np.random.uniform(low=0, high=1, size=(c1_w.shape[1], 100)),
        index=c1_w.columns
    )
    c1_h: pd.DataFrame = c1_h / c1_h.sum()
    c1_x: pd.DataFrame = c1_w.dot(c1_h)

    # C2 - No ES_Bifi
    c2_w: pd.DataFrame = es_w.drop(columns=['ES_Bifi'])
    c2_h: pd.DataFrame = pd.DataFrame(
        np.random.uniform(low=0, high=1, size=(c2_w.shape[1], 200)),
        index=c2_w.columns
    )
    c2_h: pd.DataFrame = c2_h / c2_h.sum()
    c2_x: pd.DataFrame = c2_w.dot(c2_h)

    # C3 - Extra signature
    c3_w: pd.DataFrame = es_w.copy()
    c3_w.loc[:, 'IS_1'] = es_w['ES_Firm'].sample(frac=1).values
    c3_h: pd.DataFrame = pd.DataFrame(
        np.random.uniform(low=0, high=1, size=(c3_w.shape[1], 150)),
        index=c3_w.columns
    )
    c3_h: pd.DataFrame = c3_h / c3_h.sum()
    c3_x: pd.DataFrame = c3_w.dot(c3_h)

    # C4 - Randomly shuffled, which should make irrelevant signatures
    c4_w: pd.DataFrame = es_w.copy()
    c4_h: pd.DataFrame = pd.DataFrame(
        np.random.uniform(low=0, high=1, size=(c4_w.shape[1], 100)),
        index=c4_w.columns
    )
    c4_h: pd.DataFrame = c4_h / c4_h.sum()
    c4_x: pd.DataFrame = c4_w.dot(c4_h)
    for i in range(c4_x.shape[1]):
        c4_x.iloc[:, i] = c4_x.iloc[:, i].sample(frac=1)

    # C5 - No Bifi or Esch
    c5_w: pd.DataFrame = es_w.drop(columns=['ES_Bact', 'ES_Esch'])
    c5_h: pd.DataFrame = pd.DataFrame(
        np.random.uniform(low=0, high=1, size=(c5_w.shape[1], 200)),
        index=c5_w.columns
    )
    c5_h: pd.DataFrame = c5_h / c5_h.sum()
    c5_x: pd.DataFrame = c5_w.dot(c5_h)

    # C6 - Bact and Prev merged
    c6_w: pd.DataFrame = es_w.copy()
    c6_w['ES_Prev+Bact'] = c6_w['ES_Prev'] + c6_w['ES_Bact']
    c6_w = c6_w.drop(columns=['ES_Bact', 'ES_Prev'])
    c6_h: pd.DataFrame = pd.DataFrame(
        np.random.uniform(low=0, high=1, size=(c6_w.shape[1], 200)),
        index=c6_w.columns
    )
    c6_h: pd.DataFrame = c6_h / c6_h.sum()
    c6_x: pd.DataFrame = c6_w.dot(c6_h)

    return {name: x / x.sum() for name, x in zip(
        ['All', 'No_Bifi', 'IS_1', 'Random', 'No_Bact_Esch', 'Merge_Prev_Bact'],
        [c1_x, c2_x, c3_x, c4_x, c5_x, c6_x]
    )}


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


# def test_combine_signatures():
#     # Temp dev testing
#     # Combine signatures with themselves
#     # combine_signatures([models.five_es(), models.five_es()])
#     dupd: pd.DataFrame = models.five_es().w
#     dupd.loc[:, "Dupd_1"] = dupd.iloc[:, 0]
#     dupd.loc[:, "Merg_1"] = dupd.iloc[:, 0] + dupd.iloc[:, 1]
#     dupd.loc[:, "Rand_1"] = np.random.uniform(size=(dupd.shape[0], 1))
#     dupd.loc[:, "Merg_2"] = dupd.iloc[:, 1] + 4 * dupd.iloc[:, 4]
#     combine_signatures([models.five_es(), dupd])
#
#
# def test_experiment():
#     # Temp function - see how splitting the DRAMA data goes
#     es_x = pd.read_csv(
#         (
#             "https://gitlab.inria.fr/cfrioux/enterosignature-paper/-/raw/main/data/"
#             "GMR_dataset/GMR_genus_level_abundance_normalised.tsv?ref_type=heads"),
#         sep="\t").iloc[:, :]
#     cohorts_arr = np.random.randint(low=0, high=20, size=(es_x.shape[1]))
#     cohorts = []
#     for i in set(cohorts_arr):
#         cohorts.append(es_x.iloc[:, cohorts_arr == i])
#     mmodels = [decompositions(x,
#                               ranks=[8],
#                               random_starts=100,
#                               top_n=1
#                               )[8][0] for x in cohorts]
#     combi = combine_signatures(mmodels, x=es_x, low_support_threshold=0.25)
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
#
#
# def test_experiment2(complex_cohort_structure: List[pd.DataFrame]):
#     # Temp function - see how splitting the DRAMA data goes
#     x: pd.DataFrame = pd.concat(complex_cohort_structure, axis=1)
#     x.columns = [f'sampl{x}' for x in x.columns]
#     mmodels = [decompositions(x,
#                               ranks=[7],
#                               random_starts=20,
#                               top_n=1
#                               )[7][0] for x in complex_cohort_structure]
#     combi = combine_signatures(mmodels, x=x, low_support_threshold=0.2)
#     matched = match_signatures(combi, models.five_es().w)
#     from cvanmf.reapply import _reapply_model
#     from cvanmf.reapply import match_identical
#     combi_model_fit = _reapply_model(y=x,
#                                      w=combi,
#                                      colors=None,
#                                      input_validation=lambda x: x,
#                                      feature_match=match_identical)
#     orig_model_fit = models.five_es().reapply(x)
#     print(matched)
#
#
# def test_experiment3():
#     # Temp function - see how splitting the DRAMA data goes
#     es_x = pd.read_csv(
#         (
#             "https://gitlab.inria.fr/cfrioux/enterosignature-paper/-/raw/main/data/"
#             "GMR_dataset/GMR_genus_level_abundance_normalised.tsv?ref_type=heads"),
#         sep="\t").iloc[:, :]
#     mmodels = decompositions(es_x,
#                              ranks=[5],
#                              random_starts=100,
#                              top_n=1
#                              )[5]
#     combi = combine_signatures(mmodels, x=es_x, low_support_threshold=0.25)
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
#
#
# def test_rewrite():
#     es_x = pd.read_csv(
#         ("/Users/pez23lof/Documents/postdrama_es/MGS.matL5.20240116.rel.txt"),
#         sep="\t", index_col=0).iloc[:, :3000]
#     cohorts_arr = np.random.randint(low=0, high=5, size=(es_x.shape[1]))
#     cohorts = []
#     for i in set(cohorts_arr):
#         cohorts.append(es_x.iloc[:, cohorts_arr == i])
#     mmodels = [decompositions(x,
#                               ranks=[5],
#                               random_starts=10,
#                               top_n=10,
#                               seed=4298
#                               )[5] for x in cohorts]
#     cohort_obj = Cohort(name="COH_1")
#     model_objs = [
#         Cohort(f"COH_{i}").add_models(
#             [
#                 Model().add_signatures(Signature.from_comparable(m)) for m in c
#             ])
#         for i, c in enumerate(mmodels)]
#     for i in range(len(cohorts)):
#         model_objs[i].x = cohorts[i]
#     c = Combiner(model_objs)
#     c.merge_similar(0.9)
#     print("After Merge")
#     print(c.clusters)
#     aa = c.clusters[0].support(type="model")
#     c.remove_low_support(retain_alpha=0.05, support_required=0.5)
#     print("After Support")
#     print(c.clusters)
#     c.remove_linear_combinations_2()
#     print("After Linear Comb")
#     print(c.clusters)
#     c.label_by_match(models.five_es())
#     plotter = Signature.mds(c.clusters[0].signatures)
#     plotto = c.clusters[0].plot_mds()
#     plotto = c.plot_mds()
#     match = match_signatures(models.five_es(), Cluster.as_dataframe(c.clusters))
#     plt_featvar = c.clusters[0].plot_feature_variance(
#         top_n=20, split_cohort=True)
#     plt_membersim = c.clusters[0].plot_member_similarity(split_cohort=True)
#
#     def end_tax(tax_list):
#         return [
#             x.split(";")[-1] for x in tax_list
#         ]
#
#     plt_featclust = c.plot_feature_variance(True, 20, end_tax)
#     plt_featclust.save("/Users/pez23lof/test_clustfeat_postdrama_k5.png",
#                        height=12,
#                        width=20, dpi=200)
#     plt_clustsi = c.plot_member_similarity(True)
#     plt_clustsi.save("/Users/pez23lof/test_clustsi_postdrama_k5.png", height=12,
#                      width=20, dpi=200)
#     print(match)
#     foo = 'bar'
#
#
# def test_rewrite_rank_sloppy():
#     # How sloppy can we be about rank selection?
#     es_x = pd.read_csv(
#         ("~/Downloads/GMR_genus_level_abundance_normalised.tsv"),
#         sep="\t").iloc[:, :]
#     mmodels = decompositions(es_x,
#                              ranks=[6],
#                              random_starts=100,
#                              top_n=50,
#                              seed=4298
#                              )
#     amodels = [Model().add_signatures(Signature.from_comparable(c)) for
#                c in itertools.chain.from_iterable(mmodels.values())]
#     cohort_obj = Cohort(name="COH_1", x=es_x)
#     cohort_obj.add_models(amodels)
#     c = Combiner([cohort_obj])
#     c.merge_similar(0.9)
#     aa = c.clusters[0].support(type="model")
#     c.remove_low_support(retain_alpha=0.05, support_required=0.5,
#                          only_cohort_data=False)
#     c.remove_linear_combinations_2(min_small_support_ratio=0.5)
#     match = match_signatures(models.five_es(), Cluster.as_dataframe(c.clusters))
#     # Someting to spot "fragments" - parts of larger signatures
#     # c.clusters[0].plot_feature_variance(top_n=20, split_cohort=True)
#     # c.clusters[0].member_similarity(utri=True)
#     # c.clusters[0].plot_member_similarity(split_cohort=False)
#     c.label_by_match(models.five_es())
#     plt_feat = c.plot_feature_variance(top_n=20,
#                                        label_fn=lambda x: [y.split(";")[-1] for
#                                                            y in x])
#     plt_sim = c.plot_member_similarity()
#     print(match)
#     foo = 'bar'
#
#
# def test_rewrite_synthetic(complex_cohort_structure: Dict[str, pd.DataFrame]):
#     # Decompositions
#     models = {name: decompositions(x, ranks=[4, 5, 6], random_starts=40,
#                                    top_n=30,
#                                    progress_bar=False, seed=4298) for name, x in
#               complex_cohort_structure.items()}
#
#     # Prepare cohort structure
#     model_objs = [
#         Cohort(name).add_models(
#             [
#                 Model().add_signatures(Signature.from_comparable(m)) for m in
#                 itertools.chain.from_iterable(decomps.values())
#             ])
#         for name, decomps in models.items()]
#     for name, data in complex_cohort_structure.items():
#         next(x for x in model_objs if x.name == name).x = data
#
#     # Merge
#     c = Combiner(model_objs)
#     c.merge_similar(cosine_threshold=0.9)
#     spec = c.identify_cohort_specific()
#     c.remove_low_support(support_required=0.2, retain_alpha=0.05,
#                          only_cohort_data=False, exempt_clusters=spec)
#     c.remove_linear_combinations_2(cosine_threshold=0.9)
#     c.label_by_match(cvanmf.models.five_es())
#     plot = c.plot_mds()
#     foo = 'bar'
#
#
# def test_split_dataframe_to_cohorts():
#     smpl_names: List[str] = (
#             [f'CA_S{i}' for i in range(10)] +
#             [f'CB_S{i}' for i in range(20)] +
#             [f'CC_S{i}' for i in range(3)]
#     )
#     smpl_cohorts: List[str] = [x[1] for x in smpl_names]
#     cohort_series: pd.Series = pd.Series(smpl_cohorts, index=smpl_names)
#     df: pd.DataFrame = pd.DataFrame(
#         np.random.uniform(size=(100, len(smpl_names))),
#         columns=smpl_names
#     )
#     cohort_dict: Dict = split_dataframe_to_cohorts(df, cohort_series,
#                                                    min_size=5)
#     assert len(cohort_dict) == 2

