import collections.abc
import itertools
import logging
import math
from typing import Union, Tuple, NamedTuple, Iterable, List, Set, Literal

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal
from sklearn.metrics.pairwise import cosine_similarity

from cvanmf.denovo import Decomposition, NMFParameters
from cvanmf.models import Signatures
from cvanmf.reapply import FeatureMatch, FeatureMapping, match_identical, _reapply_model

# Alias for types which hold signatures that can be compared
Comparable = Union[Signatures, Decomposition, pd.DataFrame]


def _get_signatures(c: Comparable) -> pd.DataFrame:
    """Get the signature matrix from any of the comparable types."""

    match c:
        case pd.DataFrame():
            return c
        case Decomposition():
            return c.w
        case Signatures():
            return c.w
        case _:
            raise TypeError(
                f"Cannot find signatures in type {type(c)}, please provide"
                "one of the supported types (Signatures, Decomposition, "
                "DataFrame")


def compare_signatures(a: Comparable,
                       b: Comparable
                       ) -> pd.DataFrame:
    """Compare how similar signatures are between two models.

    For models learnt on similar data, signatures recovered may also be similar.
    We can characterise the similarity by the angle between the signature
    vectors. This functions aligns W matrices (so the features are the union of
    those in a and b), and calculate pairwise cosine similarity between
    signatures.

    This returns a DataFrame with signatures of A on rows, and B on columns
    with entry i,j being the cosine of the angle between aligned signature
    vectors A[,i] and B[,j].

    :param a: Set of signatures to be on rows
    :param b: Set of signatures to be on columns
    :return: Pairwise cosine of angle between signatures
    """

    al_a, al_b = align_signatures(a, b)
    cosine_sim: np.ndarray = cosine_similarity(al_a.T, al_b.T)
    cosine_df: pd.DataFrame = pd.DataFrame(
        cosine_sim,
        index=al_a.columns,
        columns=al_b.columns
    )
    return cosine_df


def align_signatures(
        *args
) -> List[pd.DataFrame]:
    """Give signature matrices matching indices.

    Signatures from different matrices potentially will have some different
    features (species observed in one matrix and not another). For the
    comparisons we use, need to have signatures have the same set of features
    in the same order.

    Pass any number of Comparable type objects, or a single iterable of
    Comparable type objects.

    Uses exact matching of feature name strings from the index of the W
    matrix DataFrame.
    """

    sigs: Iterable[Comparable]
    if len(args) < 1:
        raise ValueError("No signatures provided in arguments.")
    if len(args) == 1:
        if not isinstance(args[0], collections.abc.Iterable):
            raise ValueError(
                "Single argument must be an Iterable of signatures.")
        sigs = args[0]
    else:
        sigs = args

    sig_mats: Iterable[pd.DataFrame] = (_get_signatures(x) for x in sigs)
    index_union: List[str] = list(set(
        itertools.chain.from_iterable(
            (x.index for x in sig_mats)
        )
    ))
    aligned_mats: List[pd.DataFrame] = [
        x.reindex(index_union, fill_value=0) for x in
        (_get_signatures(x) for x in sigs)
    ]
    return aligned_mats


def match_signatures(a: Comparable, b: Comparable) -> pd.DataFrame:
    """Match signatures between two models maximising cosine similarity.

    Find the pairing of signatures which are most similar. More technically,
    this finds the pairing of signatures which maximises the total cosine
    similarity using the Hungarian algorithm. It is possible that a
    signature gets paired with another for which the cosine similarity is
    not highest, suggesting a potentially bad match between some signatures
    in the model.

    The return is a dataframe with columns a and b for which signatures
    are paired, the cosine similarity of the pairing, and the maximum
    'off-target' cosine value for any of the signatures which it was not
    assigned to. The intention for the off-target score is that ideally
    this would be low, and the paired similarity high: signatures match
    well their paired one, while being dissimilar to all others.

    This is a convenince method which calls
    :func:`combine.match_signatures`.

    :param a: Signature matrix, or object with signature matrix
    :param b: Signature matrix, or object with signature matrix
    :returns: DataFrame with pairing and scores"""

    mat_a, mat_b = _get_signatures(a), _get_signatures(b)
    # Swap so a is always the comparable with more signatures
    mat_a, mat_b, order = (mat_b, mat_a, ['b', 'a']) \
        if mat_b.shape[1] > mat_a.shape[1] \
        else (mat_a, mat_b, ['a', 'b'])

    cos: pd.DataFrame = compare_signatures(mat_a, mat_b)

    # Assign using Hungarian algorithm
    assign_row, assign_col = linear_sum_assignment(cos, maximize=True)
    paired: List[Tuple[str, str]] = list(zip(
        (mat_a.columns[i] for i in assign_row),
        (mat_b.columns[i] for i in assign_col)
    ))
    paired_df: pd.DataFrame = pd.DataFrame(paired, columns=order)
    paired_df['pair_cosine'] = [cos.iloc[i, j]
                                for i, j in zip(assign_row, assign_col)]

    # Add entries for non-matched signatures
    missing_indices: List[str] = [x for x in mat_a.columns
                                  if x not in list(paired_df.iloc[:, 0])]
    paired_df = pd.concat(
        [paired_df,
         pd.DataFrame(
             ((x, None) for x in missing_indices),
             columns=order
         )]
    )

    # Determine the maximum cosine of angle between row and its non-assigned
    # signatures
    mask: np.ndarray = np.zeros(shape=cos.shape)
    for i, j in zip(assign_row, assign_col):
        mask[i, j] = 1

    max_off: pd.Series = pd.Series(
        np.ma.masked_array(cos, mask).max(axis=1),
        index=cos.index)

    # Reorder to match table ordering, cosine matrix does not match this
    # currently
    paired_df['max_off_cosine'] = max_off.loc[paired_df.iloc[:, 0]].values
    return paired_df


def combine_signatures(
        signatures: Iterable[Comparable],
        x: pd.DataFrame = None,
        merge_threshold: float = 0.9,
        split_threshold: float = 0.9,
        prune_low_support: bool = True,
        low_support_threshold: Union[int, float] = 0.1,
        low_support_alpha: float = 0.05
) -> Decomposition:
    """Combine signatures into a non-redundant set."""

    # 1. Cosine similarity between all
    logging.info("Concatenating signatures from models")
    aligned_sigs: List[pd.DataFrame] = align_signatures(signatures)
    cat_signatures: pd.DataFrame = pd.concat(
        aligned_sigs,
        axis=1
    )
    merged_sigs: pd.DataFrame = cat_signatures
    # Keep identifying and merging cliques until no cliques |c| > 1 remain
    while True:
        cos: pd.DataFrame = cosine_similarity(merged_sigs.T)

        # 2. Maximal clique detection
        # This might benefit from being replaced with some near-clique methods
        # Could replace with some clustering method instead if this seems too
        # crude
        # Apply threshold to get adjacency matrix, convert to graph, B-K to find
        # cliques
        logging.info("Identifying cliques in similarity graph")
        adj: pd.DataFrame = cos > merge_threshold
        g = nx.from_numpy_array(adj)
        cliques = list(nx.find_cliques(g))
        logging.info("Found %s cliques, merging to single signatures",
                     len(cliques))

        # If all cliques are a single node, terminate merging
        if max(len(x) for x in cliques) < 2:
            break

        # 3. Merging
        # Make a new dataframe with the merged signatures
        # After making, remove all deleted signatures
        # Concatenate on merged signatures
        merged_sigs_i: pd.DataFrame = pd.concat(
            [merged_sigs.iloc[:, c].mean(axis=1) for c in cliques],
            axis=1
        )
        merged_sigs = pd.concat(
            [merged_sigs_i,
             merged_sigs.iloc[:,
             [x for x in range(merged_sigs.shape[1])
              if x not in list(itertools.chain.from_iterable(cliques))]
             ]],
            axis=1
        )

    # 4a. Fit each signature S_a to S_{-a} to and see if good similarity
    # (meaning it can be described well as a mix as some other signatures)
    # and can probably be removed
    logging.info("Pruning signatures which are adequately described as a mix of"
                 "other signatures")
    sig_remove: List[int] = []
    for i in range(merged_sigs.shape[1]):
        retain_sigs: List[int] = [x for x in range(merged_sigs.shape[1])
                                  if x not in sig_remove and x != i]
        reduced_sigs: pd.DataFrame = merged_sigs.iloc[:, retain_sigs]
        reduced_model: Decomposition = _reapply_model(
            y=merged_sigs.iloc[:, i].to_frame(),
            w=reduced_sigs,
            input_validation=lambda x: x,
            feature_match=match_identical,
            colors=None
        )
        if reduced_model.model_fit.iloc[0] > split_threshold:
            sig_remove.append(i)
    logging.info("Removed %s signatures which are linear combinations"
                 "of others", len(sig_remove))
    merged_sigs = merged_sigs.iloc[:,
                  [x for x in range(merged_sigs.shape[1])
                   if x not in sig_remove]]

    if not prune_low_support:
        return merged_sigs

    # 5. Remove signatures with low support which do not significantly
    # change model fit when removed
    # Determine number of models which contained something similar to
    # this signature, using the merge threshold
    support: pd.Series = pd.concat(
        [pd.Series(
            (cosine_similarity(merged_sigs.T, x.T) > merge_threshold
             ).any(axis=1)
        )
            for x in aligned_sigs],
        axis=1
    ).sum(axis=1)
    # support is now a series with the number of models which contain a
    # signature similar to this
    req_support: int  # If signature in this or more models, is well supported
    match low_support_threshold:
        case int():
            req_support = low_support_threshold
        case float():
            req_support = math.ceil(len(aligned_sigs) * low_support_threshold)
        case _:
            raise TypeError('low_support_threshold should be int or float')
    req_support = max(1, min(req_support, len(aligned_sigs)))
    low_support_sigs: pd.DataFrame = support[support < req_support]
    # Test the effect of removing the low support signature from a model
    # containing only the well supported signatures and this one
    good_sigs: pd.DataFrame = merged_sigs.iloc[:,
                              [x for x in support.index
                               if x not in low_support_sigs.index]]
    reject_low: List[int] = []
    for poor_sig, support in low_support_sigs.items():
        i_sigs: pd.DataFrame = pd.concat(
            [merged_sigs.iloc[:, poor_sig],
            good_sigs],
            axis=1,
        )
        # Rename to avoid undesired behaviour with dropping column by index
        i_sigs.columns = [f's{i}' for i in i_sigs.columns]
        loo_modelfit: List[pd.Series] = [
            _reapply_model(y=x,
                           w=i_sigs.drop(columns=[i]),
                           colors=None,
                           input_validation=lambda x: x,
                           feature_match=match_identical).model_fit
            for i in i_sigs.columns
        ]
        h, p = kruskal(loo_modelfit[0],
                       pd.concat(loo_modelfit[1:]))
        if (loo_modelfit[0].mean() > pd.concat(loo_modelfit[1:]).mean() and
            p <= low_support_alpha):
            # Reject this signature
            reject_low.append(poor_sig)
    merged_sigs = merged_sigs.iloc[:, [x for x in range(merged_sigs.shape[1])
                                       if x not in reject_low]]

    return merged_sigs
