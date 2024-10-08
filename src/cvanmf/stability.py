"""Describe how stable signatures are across initialisations and ranks.

It can be useful to look at how similar signatures are across multiple random
intialisations. When a signatures is not frequently repeated across iterations,
we can consider this as potentially as poor selection of rank, or place less
confidence in those signatures. This can also serve as a method of rank
selection, by looking for ranks where signatures show high similarity across
ranks.

Functions in this module are primarily concerned with looking at stability
of each signature, rather than rank selection.

The main functions are :func:`signature_stability` and
:func:`plot_signature_stability`.

For rank selection using stability, see instead
:func:`cvanmf.denovo.signature_similarity`.
"""

from __future__ import annotations

import collections.abc
import itertools
import logging

from typing import Union, List, Iterable, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import plotnine
from multimethod import multimethod
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

from cvanmf.denovo import Decomposition, decompositions
from cvanmf.models import Signatures

Comparable = Union[Signatures, Decomposition, pd.DataFrame]

logger: logging.Logger = logging.getLogger(__name__)


def compare_signatures(a: Comparable,
                       b: Comparable
                       ) -> pd.DataFrame:
    """Compare how similar signatures are between two models.

    For models learnt on similar data, signatures recovered may also be similar.
    We can characterise the similarity by the angle between the signature
    vectors. This functions aligns W matrices (so the features are the union of
    those in a and b), and calculates pairwise cosine similarity between
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

    sig_mats: Iterable[pd.DataFrame] = (
        get_signatures_from_comparable(x) for x in sigs)
    index_union: List[str] = list(set(
        itertools.chain.from_iterable(
            (x.index for x in sig_mats)
        )
    ))
    aligned_mats: List[pd.DataFrame] = [
        x.reindex(index_union, fill_value=0) for x in
        (get_signatures_from_comparable(x) for x in sigs)
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

    :param a: Signature matrix, or object with signature matrix
    :param b: Signature matrix, or object with signature matrix
    :returns: DataFrame with pairing and scores"""

    mat_a, mat_b = (
        get_signatures_from_comparable(a), get_signatures_from_comparable(b))
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


def get_signatures_from_comparable(c: Comparable) -> pd.DataFrame:
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


@multimethod
def signature_stability(
        decomps: List[Decomposition],
        to: Comparable = None
) -> pd.DataFrame:
    """Characterise how similar signatures are across random initialisations.

    Compare the signatures in ``to`` to those in ``decomps``. If
    ``to`` is None, the first value in ``decomps`` will be used.

    Each model in ``decomps`` is compared to reference ``to`` using
    ``match_signatures``.

    Note that if you compare signatures for multiple ranks, the orders
    of signatures are not related across ranks, i.e. S1 in k=2 and S1 in k=3
    are not related.

    :param decomps: Decompositions to compare to. If this is a list,
        all decompositions must have the same rank.
    :param to: Reference decompositions to compare each of decomps to. If
        this None then the first item in the list will be used.
    """

    if to is None:
        logger.info("Using first decomposition as reference for comparison.")
        to = decomps[0]
    ref: pd.DataFrame = get_signatures_from_comparable(to)

    matched_sigs: List[pd.DataFrame] = [
        match_signatures(ref, x).assign(model=i, k=ref.shape[1])
        for i, x in enumerate(decomps) if x is not to
    ]
    matched_df: pd.DataFrame = pd.concat(matched_sigs)

    return matched_df


@multimethod
def signature_stability(
        decomps: Dict[int, List[Decomposition]],
        to: Comparable = None
) -> pd.DataFrame:
    """Characterise how similar signatures are across random initialisations.

    This versions accepts a dictionary of results, with keys being rank, and
    values list of decompositions (the output format of
    :func:`cvanmf.denovo.decompositions`). For each rank, the first (best)
    decomposition is compared to the others.
    """

    if to is not None:
        logger.warning(
            "'to' not used when given a dictionary of Decompositions."
            "For each rank, the first Decomposition is used at the reference."
        )

    return pd.concat([
        signature_stability(x) for x in decomps.values()
    ])


def plot_signature_stability(
        stability: pd.DataFrame,
        colors: Optional[List[str]] = None,
        ncol: int = 6,
        geom_boxplot: Dict[str, Any] = None,
        geom_line: Optional[Union[bool, Dict[str, Any]]] = None
) -> plotnine.ggplot:
    """Plot the similarity of signatures across multiple decompositions.

    The distribution of how similar (measured by cosine similarity) the
    paired signature in each other model is will be represented as boxplots.
    Each panel is the distribution for one rank.

    :param stability: DataFrame in format returned by
        :func:`signature_stability`.
    :param colors: Colours to be applied to signatures. If list is shorter
        than the number of signatures, excess will be grey.
    :param ncol: Number of columns in the plot.
    :param geom_boxplot: Arguments to pass to plotnine's geom_boxplot class.
    :param geom_line: Arguments to pass to plotnine's geom_line class. If set
        to True, will draw lines connecting signatures from each model with
        default styling; pass dictionary to alter styling of lines.
    """

    fill_scale: plotnine.scale_fill_discrete = (
        plotnine.scale_fill_manual(
            values=colors,
            limits=stability['a'].unique()
        )
        if colors is not None else
        plotnine.scale_fill_discrete()
    )
    geom_line = dict() if geom_line == True else geom_line
    l_args = dict(linewidth=0.1, color='grey') | (dict() if geom_line is None
                                                  else geom_line)
    bp_args = dict() | (dict() if geom_boxplot is None else geom_boxplot)
    plt: plotnine.ggplot = (
        plotnine.ggplot(
            stability,
            plotnine.aes(x='a', y='pair_cosine', fill='a')
        ) +
        plotnine.geom_boxplot(**bp_args) +
        plotnine.facet_wrap(facets="k", ncol=ncol) +
        plotnine.xlab('Signature') +
        plotnine.ylab('Cosine Similarity') +
        plotnine.guides(fill=plotnine.guide_legend(title='Signature')) +
        fill_scale
    )
    if geom_line is not None:
        plt = plt + plotnine.geom_line(**l_args)
    return plt