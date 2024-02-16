"""
Generate new Enterosignature models using NMF decomposition

It is likely that additional Enterosignatures exist in data with different
populations, or new Enterosignatures might become evident with ever-increasing
taxonomic resolution. This module provides functions to generate new models
from data, which encompasses three main steps: rank selection, regularisation
selection, and model inspection. The first of these two steps involves
running decompositions multiple times for a range of values, and can be
time-consuming. Methods are provided to run the whole process on a single
machine, but also for running individual decompositions, which are used by the
accompanying nextflow pipeline to allow spreading the computation across
multiple nodes in an HPC environment.
"""
from __future__ import annotations

import collections
import glob
import hashlib
import inspect
import itertools
import logging
import math
import os
import pathlib
import re
from functools import cached_property
from typing import (Optional, NamedTuple, List, Iterable, Union, Tuple, Set,
                    Any, Dict, Callable)

import click
import numpy as np
import pandas as pd
# import patchworklib as pw
import plotnine
import sklearn
from scipy.spatial.distance import pdist, squareform
from sklearn import manifold
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.decomposition._nmf import _beta_divergence
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Type aliases
# BicvSplit = List[pd.DataFrame]
Numeric = Union[int, float]
"""Alias for a python numeric types"""


class BicvFold(NamedTuple):
    """One fold from a shuffled matrix

    The 9 submatrices have been joined into the structure shown below

    .. code-block:: text

        A B B
        C D D
        C D D

    from which A will be estimated as A' using only B, C, D.
    ```
    """
    A: pd.DataFrame
    B: pd.DataFrame
    C: pd.DataFrame
    D: pd.DataFrame


class BicvSplit:
    """Shuffled matrix for bi-cross validation, split into 9 matrices in a 3x3
    pattern. To shuffle and split an existing matrix, use the static method
    :method:`BicvSplit.from_matrix`"""

    FOLD_PERMUTATIONS = [
        [0, 1, 2],
        [1, 0, 2],
        [2, 1, 0]
    ]
    """Distinct permutations of submatrix rows/columns for generating folds."""

    # TODO: Consider generalising this to other folds rather than just 9

    def __init__(self, mx: List[pd.DataFrame], i: Optional[int] = None) -> None:
        """Create a shuffled matrix containing the 9 split matrices. These
        should be in the order

        .. code-block:: text

            0, 1, 2
            3, 4, 5
            6, 7, 8

        :param mx: Split matrices
        :type mx: List[pd.DataFrame]
        :param i: Index of the split if one of many
        :type i: int
        """

        # Some validations
        if len(mx) != 9:
            raise ValueError("Must provide 9 sub-matrices to BicvSplit")
        self.__mx: List[List[pd.DataFrame]] = [mx[i:i + 3] for i in range(0, 9, 3)]
        self.__i: Optional[int] = i

    # TODO: Read up on how to implement slicing properly
    @property
    def mx(self) -> List[List[pd.DataFrame]]:
        """Submatrices as a 2d list."""
        return self.__mx

    @property
    def x(self) -> pd.DataFrame:
        """The input matrix

        This reproduces the input matrix by concatenating the submatrices.

        :returns: Input matrix
        """
        return pd.concat([self.row(i, join=True) for i in range(3)], axis=0)

    @property
    def i(self) -> int:
        return self.__i

    @i.setter
    def i(self, i: int) -> None:
        self.__i = i

    @property
    def shape(self) -> Tuple[int, int]:
        """Dimensions of the input matrix."""
        return (self.col(0, join=True).shape[0],
                self.row(0, join=True).shape[1])

    @property
    def size(self) -> int:
        """Size of input matrix."""
        shape: Tuple[int, int] = self.shape
        return shape[0] * shape[1]

    def row(self,
            row: int,
            join: bool = False
            ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """Get a row of the submatrices by index. Convenience method for
        readability.
        
        :param row: Row index to get
        :type row: int
        :param join: Join into a single matrix
        :type join: bool
        :returns: List of submatrices making up the row, or these submatrices
            joined if requested.
        :rtype: Union[List[pd.DataFrame], pd.DataFrame]
        """
        subs: List[pd.DataFrame] = self.mx[row]
        if join:
            return pd.concat(subs, axis=1)
        return subs

    def col(self,
            col: int,
            join: bool = False
            ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """Get a column of the submatrices by index. Convenience method for
        readability.

        :param col: Column index to get
        :type col: int
        :param join: Join into a single matrix
        :type join: bool
        :returns: List of submatrices making up the column, or these submatrices
            joined if requested.
        :rtype: Union[List[pd.DataFrame], pd.DataFrame]
        """
        subs: List[pd.DataFrame] = [x[col] for x in self.mx]
        if join:
            return pd.concat(subs, axis=0)
        return subs

    def save_npz(self,
                 path: pathlib.Path,
                 compress: bool = True,
                 force: bool = False) -> None:
        """Save these splits to file.

        Write the splits to a numpy format file. This will lose the row
        and column names, however this is unimportant for rank selection.
        Compression is enabled by default, as sparse data such as microbiome
        counts tends to create large files.

        :param path: Path to write to. If passed a directory, will output
            with filename `shuffle_{i}.npz`. If `i` is not set, cause an error.
        :param compress: Use compression.
        :param force: Overwrite existing files.
        """
        if path.is_dir():
            if self.i is None:
                raise ValueError(
                    ("When writing BicvSplit to directory i cannot be None"
                     " to allow a unique filename to be constructed.")
                )
            path = path / f"shuffle_{self.i}.npz"
        if force:
            path.unlink(missing_ok=True)
        if path.exists():
            raise FileExistsError(
                (f"File {path} already exists when saving splits. Use "
                 f"force=True to overwrite existing files")
            )
        save_fn = np.savez_compressed if compress else np.savez
        save_fn(path, *[
            x.values for x in itertools.chain.from_iterable(self.mx)])

    def fold(self,
             i: int) -> BicvFold:
        """Construct a fold of the data

        There are 9 possible folds of the data, this function constructs the
        i-th fold.

        :param i: Index of the fold to construct, from 0 to 8
        :returns: A, B, C, and D matrices for this fold
        """
        if i > len(self.FOLD_PERMUTATIONS) ** 2:
            raise IndexError(f"Fold index out of range.")
        row, col = divmod(i, len(self.FOLD_PERMUTATIONS))
        row_idx, col_idx = (self.FOLD_PERMUTATIONS[row],
                            self.FOLD_PERMUTATIONS[col])
        # Make a rearranged mx list
        mx_i: List[List[pd.DataFrame]] = [
            [self.mx[ri][ci] for ci in col_idx] for ri in row_idx
        ]
        # Make submatrices
        a: pd.DataFrame = mx_i[0][0]
        b: pd.DataFrame = pd.concat([mx_i[0][1], mx_i[0][2]], axis=1)
        c: pd.DataFrame = pd.concat([mx_i[1][0], mx_i[2][0]], axis=0)
        d: pd.DataFrame = pd.concat(
            [pd.concat([mx_i[1][1], mx_i[1][2]], axis=1),
             pd.concat([mx_i[2][1], mx_i[2][2]], axis=1)],
            axis=0
        )
        return BicvFold(a, b, c, d)

    @property
    def folds(self) -> List[BicvFold]:
        """List of the 9 possible folds of these submatrices."""
        return [self.fold(i) for i in range(9)]

    @staticmethod
    def save_all_npz(
            splits: Iterable[BicvSplit],
            path: pathlib.Path,
            fix_i: bool = False,
            force: bool = False,
            compress: bool = True
    ) -> None:
        """Save a collection of splits to a directory as npz files.

        :param splits: Iterable of BicvSplit objects
        :param path: Directory to write to
        :param fix_i: Where values of i for object in splits are not unique,
            renumber them so that they are.
        :param compress: Use compression
        :param force: Overwrite existing files
        """
        # Correct numbering if needed
        i_vals: Set[Optional[int]] = set(
            x.i for x in splits
        )
        if len(i_vals) != len(i_vals) or None in i_vals:
            # Some non-unique values of i, so all will be renumbered
            logging.warning("Non-unique values of i while saving shuffles")
            if fix_i:
                logging.warning("Renumbering all shuffles")
                for i, x in enumerate(splits):
                    x.i = i
            else:
                raise ValueError("Non-unique values of i while saving shuffles")

        # Write all
        _ = [x.save_npz(path=path, compress=compress, force=force)
             for x in splits]
        logging.info("%s shuffles saved to %s", len(list(splits)),
                     path)

    @staticmethod
    def load_npz(path: pathlib.Path,
                 allow_pickle: bool = True,
                 i: Optional[int] = None) -> BicvSplit:
        """Load splits from file.

        Load splits from an npz file. This will mean don't have the column
        and row names anymore, but this is unimportant for cross-validation.
        Pickling is required for loading compressed files, but is not secure,
        so the option is provided to turn it off if you don't need it.

        :param path: File to load
        :param allow_pickle: Allow unpickling when loading; necessary for
            compressed files
        :param i: Shuffle number. Will attempt to parse from filename
            if blank. Only parses files like prefix_1.npz, which will be i=1,
            prefix only alphanumeric.
        """
        use_i: Optional[int] = i
        if i is None:
            # Try to infer i using regexp
            rei: re.Match = re.match(r'[A-Za-z\d]*_(\d*).npz', path.name)
            if rei is not None:
                use_i = rei.group(1)
        with np.load(path, allow_pickle=allow_pickle) as mats:
            return BicvSplit([pd.DataFrame(x) for x in mats.values()], i=use_i)

    @staticmethod
    def load_all_npz(
            path: Union[pathlib.Path, str],
            allow_pickle: bool = True,
            fix_i: bool = False
    ) -> List[BicvSplit]:
        """Read shuffles from files.

        Reads either all the npz files in a directory, or those specified by a
        glob. The expectation is the filenames are in format prefix_i.npz,
        where i is the number of this shuffle. If not, use fix_i to renumber
        in order loaded.

        :param path: Directory with .npz files, or glob identifying .npz files
        :param allow_pickle: Allow unpickling when loading; necessary for
            compressed files.
        :param fix_i: Renumber shuffles
        """
        npz_glob: str = path
        if isinstance(path, pathlib.Path):
            npz_glob = str(path / "*.npz")
        logging.info("Loading shuffles from %s", str(npz_glob))
        shuffles: List[BicvSplit] = [
            BicvSplit.load_npz(path=pathlib.Path(x),
                               allow_pickle=allow_pickle)
            for x in glob.glob(npz_glob)
        ]
        logging.info("%s shuffles loaded", len(shuffles))
        if fix_i:
            logging.info("Reindexing all loaded shuffles")
            for i, x in shuffles:
                x.i = i
        return sorted(shuffles, key=lambda x: x.i)

    @staticmethod
    def from_matrix(
            df: pd.DataFrame,
            n: int = 1,
            random_state: Optional[Union[int, np.random.Generator]] = None
    ) -> List[BicvSplit]:
        """Create random shuffles and splits of a matrix

        :param df: Matrix to shuffle and split
        :param n: Number of shuffles
        :param random_state: Random state, either int seed or numpy Generator;
            None for default numpy random Generator.
        """
        rnd: np.random.Generator = np.random.default_rng(random_state)
        logging.info(
            "Generating %s splits with state %s", str(n), str(rnd))

        shuffles: Iterable[pd.DataFrame] = list(
            map(BicvSplit.__bicv_shuffle, (df for _ in range(n)))
        )
        splits: List[List[pd.DataFrame]] = list(
            map(BicvSplit.__bicv_split, shuffles)
        )
        return [BicvSplit(x, i) for i, x in enumerate(splits)]

    @staticmethod
    def __bicv_split(df: pd.DataFrame) -> List[pd.DataFrame]:
        """Make a 3x3 split of a matrix for bi-cross validation. This is adapted
        from Clemence's work in bicv_rank.py at Adapted from bicv_rank.py at
        https://gitlab.inria.fr/cfrioux/enterosignature-paper.

        :param df: Matrix to split
        :type df: pd.DataFrame
        :return: Matrix split into 3x3 sections
        :rtype: BicvSplit, which is List[pd.DataFrame] of length 9
        """
        # Cut the h x h submatrices
        chunks_feat: int = df.shape[0] // 3
        chunks_sample: int = df.shape[1] // 3
        thresholds_feat: List[int] = [chunks_feat * i for i in range(1, 3)]
        thresholds_feat.insert(0, 0)
        thresholds_feat.append(df.shape[0] + 1)
        thresholds_sample: List[int] = [chunks_sample * i for i in range(1, 3)]
        thresholds_sample.insert(0, 0)
        thresholds_sample.append(df.shape[1] + 1)

        # Return the 9 matrices
        # TODO: Tidy this a little
        matrix_nb = [i for i in range(1, 3 * 3 + 1)]
        all_sub_matrices: List[Optional[pd.DataFrame]] = [None] * 9
        done = 0
        row = 0
        col = 0
        while row < 3:
            while col < 3:
                done += 1
                all_sub_matrices[done - 1] = df.iloc[
                                             thresholds_feat[row]:thresholds_feat[row + 1],
                                             thresholds_sample[col]: thresholds_sample[col + 1]
                                             ]
                col += 1
            row += 1
            col = 0
        assert (len(all_sub_matrices) == len(matrix_nb))
        return all_sub_matrices

    @staticmethod
    def __bicv_shuffle(
            df: pd.DataFrame,
            random_state: Optional[Union[np.random.Generator, int]] = None
    ) -> pd.DataFrame:
        """Perform a single shuffle of a matrix.

        :param df: Matrix to shuffle
        :type df: pd.DataFrame
        :param random_state: Randomisation state. Either a numpy Generator,
            an integer seed for a Generator, or None for the default generator.
        :type random_state: int
        :returns: Shuffled matrix
        :rtype: pd.DataFrame
        """
        # Get a random state - if this is already a generator it is returned
        # unaltered, if int uses this as a seed, if None gets Generator with
        # fresh entropy from OS
        rnd: np.random.Generator = np.random.default_rng(random_state)

        # Do shuffles of both rows and columns
        return (df
                .sample(frac=1.0, replace=False, random_state=rnd, axis=0)
                .sample(frac=1.0, replace=False, random_state=rnd, axis=1)
                )

    def __repr__(self) -> str:
        return f'BicvSplit[i={self.i}, shape=[{self.shape}]]'


class NMFParameters(NamedTuple):
    """Parameters for a single decomposition, or iterations of bi-cross
    validation. See sklearn NMF documentation for more detail on parameters."""
    x: Optional[Union[BicvSplit, pd.DataFrame]]
    """For a simple decomposition, a matrix as a dataframe. For a bi-cross
    validation iteration, this should be the shuffled matrix split into 9 parts.
    When returning results and keep_mats is False, this will be set to None to
    avoid passing and saving large data."""
    rank: int
    """Rank of the decomposition."""
    seed: Optional[Union[int, np.random.Generator]] = None
    """Random seed for initialising decomposition matrices; if None no seed
    used so results will not be reproducible."""
    alpha: float = 0.0
    """Regularisation parameter applied to both H and W matrices."""
    l1_ratio: float = 0.0
    """Regularisation mixing parameter. In range 0.0 <= l1_ratio <= 1.0."""
    max_iter: int = 3000
    """Maximum number of iterations during decomposition. Will terminate earlier
    if solution converges."""
    keep_mats: bool = False
    """Whether to return the H and W matrices as part of the results."""
    beta_loss: str = "kullback-leibler"
    """Beta loss function for NMF decomposition."""
    init: str = "nndsvdar"
    """Initialisation method for H and W matrices on first step. Defaults to 
    non-negative SVD with small random values added."""


class BicvResult(NamedTuple):
    """Results from a single bi-cross validation run. For each BicvSplit there
    are 9 folds, for which the top left submatrix (A) is estimated (A') using
    the other portions."""
    parameters: NMFParameters
    """Parameters used during this run."""
    a: Optional[List[np.ndarray]]
    """Reconstructed matrix A for each fold. Not included when keep_mats is
    False."""
    r_squared: np.ndarray
    """Explained variance between each A and A', with each considered as a 
    flattened vector."""
    reconstruction_error: np.ndarray
    """Reconstruction error between each A and A'."""
    sparsity_h: np.ndarray
    """Sparsity of H matrix for each A'"""
    sparsity_w: np.ndarray
    """Sparsity of W matrix for each A'"""
    rss: np.ndarray
    """Residual sum of squares between each A and A'"""
    cosine_similarity: np.ndarray
    """Cosine similarity between each A and A' considered as a flattened 
    vector."""
    l2_norm: np.ndarray
    """L2 norm between each A and A'."""

    @staticmethod
    def join_folds(results: List[BicvResult]) -> BicvResult:
        """Join results from individual folds

        Each fold returns a BicvResult with a length one array. This method
        joins these into a single object summarising all the folds. Could
        also join other sets of results.

        :param results: Results from individual folds
        :returns: Single object with individual arrays joined
        """

        # Ensure params are equivalent
        if len(set(x.parameters for x in results)) > 1:
            raise ValueError(("Trying to join results run with different "
                              "parameters"))
        return BicvResult(
            parameters=results[0].parameters,
            r_squared=np.concatenate([x.r_squared for x in results]),
            reconstruction_error=np.concatenate(
                [x.reconstruction_error for x in results]),
            sparsity_h=np.concatenate([x.sparsity_h for x in results]),
            sparsity_w=np.concatenate([x.sparsity_w for x in results]),
            rss=np.concatenate([x.rss for x in results]),
            cosine_similarity=np.concatenate(
                [x.cosine_similarity for x in results]),
            l2_norm=np.concatenate([x.l2_norm for x in results]),
            a=([x.a for x in results] if
               results[0].parameters.keep_mats else None)
        )

    def to_series(self,
                  summarise: Callable[[np.ndarray], float] = np.mean
                  ) -> pd.Series:
        """Convert bi-fold cross validation results to series

        :param summarise: Function to reduce each the measure (r_squared etc)
            to a single value for each shuffle.
        :returns: Series with entry for each non-parameter measure
        """
        exclude: Set[str] = {"a", "parameters"}
        res_dict: Dict[str, float] = {
            name: summarise(vals) for name, vals in self._asdict().items() if
            name not in exclude
        }
        # Append the parameters which aren't complex data types
        # Just include the strings and numbers
        params: Dict[str, Any] = {
            name: val for name, val in self.parameters._asdict().items() if
            any(isinstance(val, t) for t in [str, int, float])
        }
        # Also include the shuffle number from shuffle params, might be of
        # interest to users
        params['shuffle_num'] = self.parameters.x.i
        # Join these two and turn into a series
        series: pd.Series = pd.Series(
            {**res_dict, **params}
        )
        return series

    @staticmethod
    def results_to_table(
            results: Union[
                Iterable[BicvResult], Dict[Numeric, Iterable[BicvResult]]
            ],
            summarise: Callable[[np.ndarray], float] = np.mean
    ) -> pd.DataFrame:
        """Convert bi-fold crossvalidation results to a table

        For results run with the same parameters, convert the output to a
        table suitable for plotting.

        :param results: List of results for bicv runs with the same
            parameters on different shuffles of the data, or dict of runs
            across multiple values on the same shuffles.
        :param summarise: Function to reduce each the measure (r_squared etc)
            to a single value for each shuffle.
        """
        result: Iterable[BicvResult] = (
            results if not isinstance(results, dict) else
            itertools.chain.from_iterable(results.values())
        )
        df: pd.DataFrame = pd.DataFrame(
            [x.to_series(summarise=summarise) for x in result])
        return df


def rank_selection(x: pd.DataFrame,
                   ranks: Iterable[int],
                   shuffles: int = 100,
                   keep_mats: Optional[bool] = None,
                   seed: Optional[Union[int, np.random.Generator]] = None,
                   alpha: Optional[float] = None,
                   l1_ratio: Optional[float] = None,
                   max_iter: Optional[int] = None,
                   beta_loss: Optional[str] = None,
                   init: Optional[str] = None,
                   progress_bar: bool = True) -> Dict[int, List[BicvResult]]:
    """Bi-cross validation for rank selection.

    Run 9-fold bi-cross validation across a range of ranks. Briefly, the
    input matrix is shuffled `shuffles` times. Each shuffle is then split
    into 9 submatrices. The rows and columns of submatrices are permuted,
    and the top left submatrix (A) is estimated through NMF decompositions of
    the other matrices produced an estimate A'. Various measures of how well
    A' reconstructed A are provided, see :class:`BicvResult` for details
    on the measures.

    No multiprocessing is used, as a majority of build of scikit-learn
    make good use of multiple processors anyway. Decompositions are run
    from highest rank to lowest. This is to give a worst case, rather than
    over optimistic estimate, of remaining time when using the progress
    bar.

    This method returns a dictionary with each rank as a key, and a list
    containing one :class:`BicvResult` for each shuffle.

    Values other than 9 for folds are possible, currently this package only
    supports 9.

    :param x: Input matrix.
    :param ranks: Ranks of k to be searched. Iterable of unique ints.
    :param shuffles: Number of times to shuffle `x`.
    :param keep_mats: Return A' and shuffle as part of results.
    :param seed: Random value generator or seed for creation of the same.
        If not provided, will initialise with entropy from system.
    :param alpha: Regularisation coefficient
    :param l1_ratio: Ratio between L1 and L2 regularisation. L2 regularisation
        (1.0) is densifying, L1 (0.0) sparisfying.
    :param max_iter: Maximum iterations of NMF updates. Will end early if
        solution converges.
    :param beta_loss: Beta-loss function, see sklearn documentation for
        details.
    :param init: Initialisation method for H and W during decomposition.
        Used only where one of the matrices during bi-cross steps is not
        fixed. See sklearn documentation for values.
    :param progress_bar: Show a progress bar while running.
    :returns: Dictionary with entry for each rank, containing a list of
        results for each shuffle (as a :class:`BicvResult` object)
    """
    # Set up a dictionary of arguments
    args: Dict[str, Any] = {}
    if keep_mats is not None:
        args['keep_mats'] = keep_mats
    if alpha is not None:
        args['alpha'] = alpha
    if l1_ratio is not None:
        args['l1_ratio'] = l1_ratio
    if max_iter is not None:
        args['max_iter'] = max_iter
    if beta_loss is not None:
        args['beta_loss'] = beta_loss
    if init is not None:
        args['init'] = init

    # Set up random generator for this whole run
    rng: np.random.Generator = np.random.default_rng(seed)

    # Generate shuffles of data
    shuffles: List[BicvSplit] = BicvSplit.from_matrix(x,
                                                      random_state=rng,
                                                      n=shuffles)

    # Make a generator of parameter objects to pass to bicv
    params: List[NMFParameters] = list(itertools.chain.from_iterable(
        [[NMFParameters(x=x, rank=k, seed=rng, **args) for x in shuffles]
         for k in sorted(ranks, reverse=True)]
    ))

    # Get results
    # No real use in multiprocessing, sklearn implementation generally makes
    # good use of multiple cores anyway. Multiprocessing just makes processes
    # compete for resources and is slower.
    res_map: Iterable = (
        map(bicv, tqdm(params)) if progress_bar else map(bicv, params))
    results: List[BicvResult] = sorted(
        list(res_map),
        key=lambda x: x.parameters.rank
    )
    # Collect results from the ranks into lists, and place in a dictionary
    # with key = rank
    grouped_results: Dict[int, List[BicvResult]] = {
        rank_i: list(list_i) for rank_i, list_i in
        itertools.groupby(results, key=lambda z: z.parameters.rank)
    }
    # Validate that there are the same number of results for each rank.
    # Decomposition shouldn't silently fail, but best not to live in a
    # world of should. Currently deciding to warn user and still return
    # results.
    if len(set(len(y) for y in grouped_results.values())) != 1:
        logging.error(("Uneven number of results returned for each rank, "
                       "some rank selection iterations may have failed."))

    return grouped_results


def plot_rank_selection(results: Dict[int, List[BicvResult]],
                        exclude: Optional[Iterable[str]] = None,
                        geom: str = 'box',
                        summarise: str = 'mean',
                        jitter: bool = None,
                        n_col: int = None) -> plotnine.ggplot:
    """Plot rank selection results from bi-cross validation.

    Draw either box plots or violin plots showing statistics comparing
    A and A' from all bi-cross validation results across a range of ranks.
    The plotting library used is `plotnine`; the returned plot object
    can be saved or drawn using `plt_obj.save` or `plt_obj.draw` respectively.

    :param results: Dictionary of results, with rank as key and a list of
        :class:`BicvResult` for that rank as value
    :param exclude: Measures from :class:`BicvResult` not to plot.
    :param geom: Type of plot to draw. Accepts either 'box' or 'violin'
    :param summarise: How to summarise the statistics across the folds
        of a given shuffle.
    :param jitter: Draw individual points for each shuffle above the main plot.
    :param n_col: Number of columns in the plot. If blank, attempts to guess
        a sensible value.
    :return: :class:`plotnine.ggplot` instance
    """
    # Intended a user friendly interface to plot rank selection results
    # so takes string argument rather than functions etc.
    if summarise not in ['mean', 'median']:
        raise ValueError("summarise must be one of mean, median")
    exclude = {} if exclude is None else exclude
    summarise_fn: Callable[[np.ndarray], float] = (
        np.mean if summarise == "mean" else np.median)
    # What type of plot to draw
    if geom not in {"box", "violin"}:
        raise ValueError("geom must be on of box, violin")
    requested_geom = (plotnine.geom_boxplot if geom == 'box' else
                      plotnine.geom_violin)
    # Decide number of cols
    # My assumption is that many people will be viewing in a notebook,
    # so long may be preferable to wide
    if n_col is None:
        n_col = 3 if len(results) < 15 else 1

    # Determine which columns we're interested in plotting
    measures: list[str] = list(set(
        name for (name, value) in
        inspect.getmembers(
            BicvResult, lambda x: isinstance(x, collections._tuplegetter))
    ).difference({"a", "parameters", *exclude}).union({"rank"}))
    # Get results and stack so measure is a column
    df: pd.DataFrame = (
        BicvResult.results_to_table(results, summarise=summarise_fn)[measures]
    )
    # Make longer so we have measure as a column
    stacked: pd.DataFrame = (
        df
        .set_index('rank')
        .stack(dropna=False)
        .to_frame(name='value')
        .reset_index(names=["rank", "measure"])
    )

    # Plot
    plot: plotnine.ggplot = (
            plotnine.ggplot(
                data=stacked,
                mapping=plotnine.aes(
                    x="factor(rank)",
                    y="value"
                )
            )
            + plotnine.facet_wrap(facets="measure", scales="free_y",
                                  ncol=n_col)
            + requested_geom(mapping=plotnine.aes(fill="measure"))
            + plotnine.xlab("Rank")
            + plotnine.ylab("Value")
            + plotnine.scale_fill_discrete(guide=False)
    )

    # If not specifically requested, decide whether to add a jitter
    if jitter is None:
        jitter = False if len(list(results.values())[0]) > 150 else True
    if jitter:
        plot = plot + plotnine.geom_jitter()

    return plot


def bicv(params: Optional[NMFParameters] = None, **kwargs) -> BicvResult:
    """Perform a single run of bi-cross validation

    Perform one run of bi-cross validation. Parameters can either be passed
    as a :class:`BicvParameters` tuple and are documented there, or by keyword
    arguments using the same names as :class`BicvParameters`.

    :returns: Comparisons of the held out submatrix and estimate for each fold
    """

    if params is None:
        params = NMFParameters(**kwargs)

    runs: Iterable[BicvResult] = map(__bicv_single,
                                     (params for _ in range(9)),
                                     (params.x.fold(i) for i in range(9))
                                     )

    # Combine results from each fold
    logging.info("Starting bi-cross validation")
    joined: BicvResult = BicvResult.join_folds(list(runs))
    return joined


def __bicv_single(params: NMFParameters, fold: BicvFold) -> BicvResult:
    """Run bi-cross validation on a single fold. Return a results tuple with
    single entries in all the result fields, these are later joined. This
    implementation is based on scripts from
    https://gitlab.inria.fr/cfrioux/enterosignature-paper/. Quality
    measurements are only done for A and A', as intermediate steps are not
    of interest for rank or regularisation selection."""

    # Get three seeds for the decompositions
    rng: np.random.Generator = np.random.default_rng(params.seed)
    decomp_seeds: np.ndarray = rng.integers(0, 4294967295, 3)

    logging.info("Starting")

    model_D: NMF = NMF(n_components=params.rank,
                       init=params.init,
                       verbose=False,
                       solver="mu",
                       max_iter=params.max_iter,
                       random_state=decomp_seeds[0],
                       l1_ratio=params.l1_ratio,
                       alpha_W=params.alpha,
                       alpha_H="same",
                       beta_loss=params.beta_loss)

    H_d: np.ndarray = np.transpose(model_D.fit_transform(np.transpose(fold.D)))
    W_d: np.ndarray = np.transpose(model_D.components_)

    # step 2, get W_a using M_b
    # Use the function rather than object interface, as has some convenience
    # for using only one fixed initialisation
    W_a, H_d_t, _ = non_negative_factorization(
        fold.B,
        n_components=params.rank,
        init='custom',
        verbose=False,
        solver="mu",
        max_iter=params.max_iter,
        random_state=decomp_seeds[1],
        l1_ratio=params.l1_ratio,
        alpha_W=params.alpha,
        alpha_H='same',
        beta_loss=params.beta_loss,
        update_H=False,
        H=H_d
    )

    # Step 3, get H_a using M_c
    H_a, W_d_t, _ = non_negative_factorization(
        fold.C.T,
        n_components=params.rank,
        init='custom',
        verbose=False,
        solver="mu",
        max_iter=params.max_iter,
        random_state=decomp_seeds[2],
        l1_ratio=params.l1_ratio,
        alpha_W=params.alpha,
        alpha_H="same",
        beta_loss=params.beta_loss,
        update_H=False,
        H=W_d.T
    )

    # Step 4, calculate error for M_a
    Ma_calc = np.dot(W_a, H_a.T)

    return BicvResult(
        parameters=params,
        a=[Ma_calc] if params.keep_mats else None,
        r_squared=np.array([_rsquared(fold.A.values, Ma_calc)]),
        sparsity_h=np.array([_sparsity(H_a)]),
        sparsity_w=np.array([_sparsity(W_a)]),
        rss=np.array([_rss(fold.A.values, Ma_calc)]),
        cosine_similarity=np.array([
            _cosine_similarity(fold.A.values, Ma_calc)]),
        l2_norm=np.array([_l2norm_calc(fold.A.values, Ma_calc)]),
        reconstruction_error=np.array([_beta_divergence(fold.A.values,
                                                        W_a,
                                                        H_a.T,
                                                        params.beta_loss)])
    )


def _rss(A: np.ndarray, A_prime: np.ndarray) -> float:
    """Residual sum of squares

    The square of difference between values in A and A'

    :param A: Held-out matrix A
    :param A_prime: Imputed matrix A
    :returns: RSS
    """
    return ((A - A_prime) ** 2).sum().sum()


def _rsquared(A: np.ndarray, A_prime: np.ndarray) -> float:
    """Explained variance

    Consider the matrix as a flattened 1d array, and calculate the R^2. This
    calculation varies from original Enterosignatures paper in that we
    subtract the mean. In the event R^2 would be inf/-inf (when total sum of
    squares is 0), this instead returns nan to make taking mean/median simpler
    later on.

    :param A: Held-out matrix A
    :param A_prime: Imputed matrix A
    :returns: R^2
    """
    A_mean: float = np.mean(np.ravel(A))
    tss: float = ((A - A_mean) ** 2).sum().sum()
    rsq: float = 1 - (_rss(A, A_prime) / tss)
    return np.nan if abs(rsq) == np.inf else rsq


def _cosine_similarity(A: np.ndarray, A_prime: np.ndarray) -> float:
    """Cosine similarity between two flattened matrices

    Cosine angle between two matrices which are flattened to a 1d vector.

    :param A: Held-out matrix A
    :param A_prime: Imputed matrix A
    :returns: Cosine similarity
    """
    x_flat: np.array = np.ravel(A)
    wh_flat: np.array = np.ravel(A_prime)
    return wh_flat.dot(x_flat) / np.sqrt(
        x_flat.dot(x_flat) * wh_flat.dot(wh_flat)
    )


def _l2norm_calc(A: np.ndarray, A_prime: np.ndarray) -> float:
    """Calculates the L2 norm metric between two matrices

    :param A: Held-out matrix A
    :param A_prime: Imputed matrix A
    :returns: L2 norm
    """
    return np.sqrt(np.sum((np.array(A) - np.array(A_prime)) ** 2))


def _sparsity(x: np.ndarray) -> float:
    """Sparsity of a matrix

    :param x: Matrix
    :returns: Sparsity, 0 to 1
    """
    return 1 - (np.count_nonzero(x) / x.size)


@click.command()
@click.option("-i",
              "--input",
              required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="""Matrix to be decomposed, in character delimited 
              format. Use -d/--delimiter to set delimiter.""")
@click.option("-o",
              "--output_dir",
              required=False,
              type=click.Path(dir_okay=True),
              default=os.getcwd(),
              help="""Directory to write output. Defaults to current directory.
              Output is a table with a row for each shuffle and rank 
              combination, and columns for each of the rank selection measures 
              (R^2, cosine similarity, etc.)""")
@click.option("-d",
              "--delimiter",
              required=False,
              type=str,
              default="\t",
              help="""Delimiter to use for input and output tables. Defaults
              to tab.""")
@click.option("--shuffles",
              required=False,
              type=int,
              default=100,
              show_default=True,
              help="""Number of times to shuffle input matrix. Bi-cross 
              validation is run once on each shuffle, for each rank.""")
@click.option("--progress/--no-progress",
              default=True,
              show_default=True,
              help="""Display progress bar showing number of bi-cross 
              validation iterations completed and remaining.""")
@click.option("--log_warning", "verbosity", flag_value="warning",
              default=True,
              help="Log only warnings or higher.")
@click.option("--log_info", "verbosity", flag_value="info",
              help="Log progress information as well as warnings etc.")
@click.option("--log_debug", "verbosity", flag_value="debug",
              help="Log debug info as well as info, warnings, etc.")
@click.option("--seed",
              required=False,
              type=int,
              help="""Seed to initialise random state. Specify if results
              need to be reproducible.""")
@click.option("-l", "--rank_min",
              required=True,
              type=int,
              help="""Lower bound of ranks to search. Must be >= 2.""")
@click.option("-u", "--rank_max",
              required=True,
              type=int,
              help="""Upper bound of ranks to search. Must be >= 2.""")
@click.option("-s", "--rank_step",
              required=False,
              type=int,
              default=1,
              help="""Step between ranks to search.""")
@click.option("--l1_ratio",
              required=False,
              type=float,
              default=0.0,
              show_default=True,
              help="""Regularisation mixing parameter. In range 0.0 <= l1_ratio 
              <= 1.0. This controls the mix between sparsifying and densifying
              regularisation. 1.0 will encourage sparsity, 0.0 density.""")
@click.option("--alpha",
              required=False,
              type=float,
              default=0.0,
              show_default=True,
              help="""Multiplier for regularisation terms.""")
@click.option("--max_iter",
              required=False,
              type=int,
              default=3000,
              show_default=True,
              help="""Maximum number of iterations during decomposition. Will 
              terminate earlier if solution converges. Warnings will be emitted
              when the solutions fail to converge.""")
@click.option("--beta_loss",
              required=False,
              type=click.Choice(
                  ['kullback-leibler', 'frobenius', 'itakura-saito']),
              default="kullback-leibler",
              show_default=True,
              help="""Beta loss function for NMF decomposition.""")
@click.option("--init",
              required=False,
              type=click.Choice(
                  ["nndsvdar", "random", "nndsvd", "nndsvda"]),
              default="nndsvdar",
              show_default=True,
              help="""Method to use when intialising H and W for 
              decomposition.""")
def cli_rank_selection(
        input: str,
        output_dir: str,
        delimiter: str,
        shuffles: int,
        progress: bool,
        verbosity: str,
        seed: int,
        rank_min: int,
        rank_max: int,
        rank_step: int,
        l1_ratio: float,
        alpha: float,
        max_iter: int,
        beta_loss: str,
        init: str,
) -> None:
    """Rank selection for NMF using 9 fold bi-cross validation

    Attempt to identify a suitable rank k for decomposition of input matrix X.
    This is done by shuffling the matrix a number of times, and for each
    shuffle diving it into 9 submatrices. Each of these nine is held out and
    and estimate learnt from the remaining matrices, and the quality of the
    estimated matrix used to identify a suitable rank.

    The underlying NMF implementation is from scikit-learn, and there is more
    documentation available there for many of the NMF specific parameters there.
    """

    # Configure logger
    log_level: int = logging.WARNING
    match verbosity:
        case "debug":
            log_level = logging.DEBUG
        case "info":
            log_level = logging.INFO
    logging.basicConfig(
        format='%(levelname)s [%(asctime)s]: %(message)s',
        datefmt='%d/%m/%Y %I:%M:%S',
        level=log_level
    )

    # Validate parameters
    # Rank min / max in right order
    rank_min, rank_max = min(rank_min, rank_max), max(rank_min, rank_max)
    ranks: List[int] = list(range(rank_min, rank_max, rank_step))
    if len(ranks) < 2:
        logging.fatal(("Must search 2 or more ranks; ranks provided were "
                       "%s"), str(ranks))
        return None

    # Read input data
    x: pd.DataFrame = pd.read_csv(input, delimiter=delimiter, index_col=0)
    # Check reasonable dimensions
    if x.shape[0] < 2 or x.shape[1] < 2:
        logging.fatal("Loaded matrix invalid shape: (%s)", x.shape)
        return
    # TODO: Check all columns numeric

    # Log params being used
    param_str: str = (
        f"Data Locations\n"
        f"------------------------------\n"
        f"Input:            {input}\n"
        f"Output:           {output_dir}\n"
        f""
        f"Bi-cross validation parameters\n"
        f"-----------------------------\n-"
        f"Seed:             {seed}\n"
        f"Ranks:            {ranks}\n"
        f"Shuffles:         {shuffles}\n"
        f"L1 Ratio:         {l1_ratio}\n"
        f"Alpha:            {alpha}\n"
        f"Max Iterations:   {max_iter}\n"
        f"Beta Loss:        {beta_loss}\n"
        f"Initialisation:   {init}\n"
    )
    logging.info(param_str)

    # Perform rank selection
    results: Dict[int, List[BicvResult]] = rank_selection(
        x=x,
        ranks=ranks,
        shuffles=shuffles,
        l1_ratio=l1_ratio,
        alpha=alpha,
        max_iter=max_iter,
        beta_loss=beta_loss,
        init=init,
        progress_bar=progress
    )

    # Write output
    rank_tbl: pd.DataFrame = BicvResult.results_to_table(results)
    rank_plt: plotnine.ggplot = plot_rank_selection(results)
    out_path: pathlib.Path = pathlib.Path(output_dir)
    logging.info("Writing results to %s", str(out_path))
    rank_tbl.to_csv(str(out_path / "rank_selection.tsv"), sep=delimiter)
    rank_plt.save(out_path / "rank_selection.pdf")

    # Completion
    logging.info("Rank selection completed")


def decompositions(
        x: pd.DataFrame,
        ranks: Iterable[int],
        random_starts: int = 100,
        top_n: int = 5,
        top_criteria: str = "beta_divergence",
        seed: Optional[Union[int, np.random.Generator]] = None,
        alpha: Optional[float] = None,
        l1_ratio: Optional[float] = None,
        max_iter: Optional[int] = None,
        beta_loss: Optional[str] = None,
        init: Optional[str] = "random",
        progress_bar: bool = True
) -> Dict[int, List[Decomposition]]:
    """Get the best decompositions for input matrix for one or more ranks.

    The model obtained by NMF decomposition depend on the initial values of the
    two matrices W and H; different initialisations lead to different solutions.
    Two approaches to initialising H and W are to attempt multiple random
    initialisations and select the best ones based on criteria such as
    reconstructions error, or to adopt a deterministic method (such as
    nndsvd) to set initial values.

    This function provides both approaches, but defaults to multiple random
    initialisations. To use one of the deterministic methods, change the
    initialisation method using `init`.

    A dictionary with one entry for each rank of decomposition requested is
    return, with the values being a list of top_n best decompositions for that
    rank. Where a deterministic method is used, the list will only have one
    item.

    :param x: Matrix to be decomposed
    :param ranks: Rank(s) of decompositions to be produced
    :param random_starts: Number of random initialisations to be tried for
        each rank. Ignored if using a deterministic initialisations.
    :param top_n: Number of decompositions to be return for each rank.
    :param top_criteria: Criteria to use when determining which are the top
        decompositions. Can be one of beta_divergence, rss, r_squared,
        cosine_similairty, or l2_norm.
    :param seed: Seed or random generator used
    :param alpha: Regularisation parameter approach to both H and W matrices.
    :param l1_ratio: Regularisation mixing parameter. In range 0.0 <= l1_ratio
          <= 1.0. This controls the mix between sparsifying and densifying
          regularisation. 1.0 will encourage sparsity, 0.0 density
    :param max_iter: Maximum number of iterations during decomposition. Will
        terminate earlier if solution converges
    :param beta_loss: Beta loss function for NMF decomposition.
    :param init: Initialisation method for H and W matrices on first step.
        Defaults to random.
    :param progress_bar: Display progress bar.
    """

    # Set up a dictionary of arguments
    args: Dict[str, Any] = {}
    if alpha is not None:
        args['alpha'] = alpha
    if l1_ratio is not None:
        args['l1_ratio'] = l1_ratio
    if max_iter is not None:
        args['max_iter'] = max_iter
    if beta_loss is not None:
        args['beta_loss'] = beta_loss
    if init is not None:
        args['init'] = init

    # Can't return more results than random starts
    top_n = min(random_starts, top_n)

    # Only one pe rank with deterministic intialisations
    if init not in {"random", "nndsvdar"}:
        logging.info("Using deterministic initialisation; only a single"
                     "result will be returned for each rank.")
        random_starts = 1

    # Check the top_criteria is a valid property
    if not top_criteria in Decomposition.TOP_CRITERIA:
        raise ValueError(
            f"Invalid value for top_criteria: {top_criteria}. "
            f"Must be one of {set(Decomposition.TOP_CRITERIA.keys())}"
        )
    take_high: bool = Decomposition.TOP_CRITERIA[top_criteria]

    # Set up random generator for this whole run
    rng: np.random.Generator = np.random.default_rng(seed)

    # Make a generator of parameter objects to pass to bicv
    params: List[NMFParameters] = list(itertools.chain.from_iterable(
        [[NMFParameters(x=x, rank=k, seed=seed, **args) for seed in
          rng.integers(low=0, high=4294967295, size=random_starts)]
         for k in sorted(ranks, reverse=True)]
    ))

    # Get results
    # No real use in multiprocessing, sklearn implementation generally makes
    # good use of multiple cores anyway. Multiprocessing just makes processes
    # compete for resources and is slower.
    res_map: Iterable = (
        map(decompose, tqdm(params)) if progress_bar else map(bicv, params))
    results: List[Decomposition] = sorted(
        list(res_map),
        key=lambda x: x.parameters.rank
    )

    # Collect results from the ranks into lists, and place in a dictionary
    # with key = rank
    grouped_results: Dict[int, List[Decomposition]] = {
        rank_i: sorted(list_i,
                       key=lambda x: getattr(x, top_criteria),
                       reverse=take_high)[:top_n]
        for rank_i, list_i in
        itertools.groupby(results, key=lambda z: z.rank)
    }

    return grouped_results


def decompose(params: NMFParameters) -> Decomposition:
    """Perform a single decomposition of a matrix.

    :param params: Decomposition parameters as a :class:`NMFParameters` object.
    :return: A single decomposition
    """

    model: NMF = NMF(
        n_components=params.rank,
        random_state=params.seed,
        alpha_H=params.alpha,
        alpha_W=params.alpha,
        l1_ratio=params.l1_ratio,
        max_iter=params.max_iter,
        beta_loss=params.beta_loss,
        init=params.init,
        solver="mu"
    )

    # Get decomposition arrays
    # Naming in line with NMF literature: H is transformed data, W feature
    # weights
    X_t: pd.DataFrame = params.x.T
    H: np.ndarray = model.fit_transform(X_t)
    W: np.ndarray = model.components_

    # Make names for the signatures
    signature_names: List[str] = [f'S{i}' for i in range(1, params.rank + 1)]

    # Convert back to dataframes
    W_df: pd.DataFrame = pd.DataFrame(W.T,
                                      index=params.x.index,
                                      columns=signature_names)
    H_df: pd.DataFrame = pd.DataFrame(H.T,
                                      index=signature_names,
                                      columns=params.x.columns)

    return Decomposition(
        parameters=params,
        h=H_df,
        w=W_df
    )


class Decomposition:
    """Decomposition of a matrix.

    Note that we use the naming conventions and orientation common in NMF
    literature:
    * X is the input matrix, with m features on rows, and n samples on columns.
    * H is the transformed data, with k signatures on rows, and n samples on
        columns.
    * W is the feature weight matrix, with m features on rows, and m
        features on columns.
    The scikit-learn implementation has these transposed; this package
    handles transposing back and forth internally, and expects input in the
    features x samples orientation, and provides W and H inline with the
    literature rather than scikit-learn.
    """

    TOP_CRITERIA: Dict[str, bool] = dict(
        cosine_similarity=True,
        r_squared=True,
        rss=False,
        l2_norm=False,
        beta_divergence=False
    )
    DEFAULT_SCALES: List[List[str]] = [
        # Used by preference, Bang Wong's 7 distinct colours for colour
        # blindness, https://www.nature.com/articles/nmeth.1618 via
        # https://davidmathlogic.com/colorblind
        ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2',
         '#D55E00', '#CC79A7' '#000000'],
        # Sarah Trubetskoy's 20 distinct colours
        # https://sashamaps.net/docs/resources/20-colors/
        ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
         '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff',
         '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
         '#000075', '#a9a9a9']
    ]
    """Defines which criteria are available to select the best decomposition
    based on, and whether to take high values (True) or low values (False)."""

    def __init__(self,
                 parameters: NMFParameters,
                 h: pd.DataFrame,
                 w: pd.DataFrame) -> None:
        self.__h: pd.DataFrame = h
        self.__w: pd.DataFrame = w
        self.__params: NMFParameters = parameters
        self.__input_hash: Optional[int] = None
        self.__colors: List[str] = self.__default_colors(n=parameters.rank)

    @property
    def h(self) -> pd.DataFrame:
        return self.__h

    @property
    def w(self) -> pd.DataFrame:
        return self.__w

    @property
    def parameters(self) -> NMFParameters:
        return self.__params

    @property
    def cosine_similarity(self) -> float:
        return float(_cosine_similarity(self.parameters.x.values,
                                        self.w.dot(self.h).values))

    @property
    def r_squared(self) -> float:
        return _rsquared(self.parameters.x.values,
                         self.w.dot(self.h).values)

    @property
    def rss(self) -> float:
        return _rss(self.parameters.x.values,
                    self.w.dot(self.h).values)

    @property
    def l2_norm(self) -> float:
        return _l2norm_calc(self.parameters.x.values,
                            self.w.dot(self.h).values)

    @property
    def sparsity_w(self) -> float:
        return _sparsity(self.w.values)

    @property
    def sparsity_h(self) -> float:
        return _sparsity(self.h.values)

    @property
    def beta_divergence(self) -> float:
        return _beta_divergence(self.parameters.x,
                                self.w.values,
                                self.h.values,
                                beta=self.parameters.beta_loss)

    @property
    def wh(self) -> pd.DataFrame:
        """Product of decomposed matrices W and H which approximates input,"""
        return self.w.dot(self.h)

    @property
    def model_fit(self) -> pd.Series:
        """How well each sample i is described by the model, expressed by the
        cosine similarity between X_i and (WH)_i."""
        cos_sim: pd.Series = pd.Series(
            np.diag(cosine_similarity(
                self.wh.T, self.parameters.x.T)),
            index=self.wh.columns,
            name="model_fit"
        )
        return cos_sim

    @property
    def input_hash(self) -> int:
        """Hash of the input matrix. Used to validate loads where data
        was not included in the saved form."""
        # Potentially expensive, only calculate if requested
        if self.__input_hash is None:
            self.__input_hash = int(
                hashlib.sha256(
                    pd.util.hash_pandas_object(
                        self.parameters.x, index=True
                    ).values).hexdigest(),
                16)
        return self.__input_hash

    @property
    def names(self) -> List[str]:
        """Names for each of the k signatures."""
        return list(self.h.index)

    @names.setter
    def names(self, names: Iterable[str]) -> None:
        """Set names for each of the k signatures. Renames the H and W
        matrices."""
        n_lst: List[str] = list(names)
        if len(n_lst) != self.h.shape[0]:
            raise ValueError(
                (f"Names given must match number of signatures. Given "
                 f"{len(n_lst)} ({n_lst}), expected {self.h.shape[0]}")
            )
        self.h.index = n_lst
        self.w.columns = n_lst

    @property
    def primary_signature(self) -> pd.Series:
        """Signature with the highest weight for each sample.

        The primary signature for a sample is the one with the highest weight
        in the H matrix. In the unusual case where all signatures have 0
        weight for a sample, this will return NaN, and is likely a sign of
        a poor model."""
        # Replace 0s with NA, as cannot have a max when picking between 0s
        return self.h.replace(0, np.nan).idxmax()

    @property
    def colors(self) -> List[str]:
        """Color which represents each signature in plots."""
        return self.__colors

    @colors.setter
    def colors(self,
               colors: Union[Dict[str, str], Iterable[str]]) -> None:
        """Set colors to represent each signatures in plots.

        Can either set all colors at once, or set individual colors using a
        signature: color dictionary."""
        if isinstance(colors, dict):
            for signature, color in colors.items():
                try:
                    sig_idx: int = self.names.index(signature)
                    self.colors[sig_idx] = color
                except IndexError as e:
                    logging.info("Unable to set color for %s, signature"
                                 "not found", signature)
        else:
            color_list: List[str] = list(colors)
            if len(color_list) < self.parameters.rank:
                logging.info("Fewer colors than signature provided. Given %s, "
                             "expected %s", len(color_list),
                             self.parameters.rank)
                self.colors[:len(color_list)] = color_list
            elif len(color_list) > self.parameters.rank:
                logging.info("More colors than signatures provided. Given %s, "
                             "expected %s", len(color_list),
                             self.parameters.rank)
                self.colors = color_list[:self.parameters.rank]
            else:
                self.colors = color_list

    def representative_signatures(self, threshold: float=0.9) -> pd.DataFrame:
        """Which signatures describe a sample.

        Identify which signatures contribute to describing a samples.
        Represenative signatures are those for which the cumulative sum is
        equal to or lower than the threshold value.

        This is done by considering each sample in the sample scaled H matrix,
        and taking a cumulative sum of weights in descending order. Any
        signature for which the cumulative sum is less than or equal to the
        threshold is considered representative.

        :param threshold: Cumulative sum below which samples are considered
            representative.
        :return: Boolean dataframe indicating whether a signature is
            representative for a given sample.
        """
        return (
            self.scaled('h')
            .apply(Decomposition.__representative_signatures,
                   threshold=threshold)
        )

    def monodominant_samples(self,
                             threshold: float=0.9
                             ) -> pd.DataFrame:
        """Which samples have a monodominant signature.

        A monodominant signature is one which represents at least the
        threshold amount of the weight in the scaled h matrix.

        :param threshold: Proportion of the scaled H matrix weight to consider
            a signature dominnant.
        :return: Dataframe with column is_monodominant indicating if a
            sample has a monodominant signature, and signature_name indicating
            the name of the signature, or none if not."""
        mono_df: pd.DataFrame = pd.concat([
            self.scaled('h').max() >= threshold,
            self.h.idxmax()
            ],
            axis=1
        )
        mono_df.columns = ['is_monodominant', 'signature_name']
        # Replace non-monodominants signature with nan
        mono_df.loc[~mono_df['is_monodominant'], 'signature_name'] = np.nan
        return mono_df

    def scaled(self,
                   matrix: Union[pd.DataFrame, str],
                   by: str = "sample"
                   ) -> pd.DataFrame:
        """Total sum scaled version of a matrix.

        Scale a matrix to a proportion of the feature/sample total, or
        to a proportion of the signature total.

        :param matrix: Matrix to be scaled, one of H or W, or a string
            {'h', 'w'}.
        :param by: Scale to proportion of sample, feature, or signature
            total.
        :return: Scaled version of matrix.
        """

        is_h: bool
        if isinstance(matrix, str):
            if matrix.lower() not in {'h', 'w'}:
                raise ValueError(
                    f"matrix must be one of 'h' or 'w'; given {matrix}"
                )
            is_h = matrix == "h"
            matrix = self.h if is_h else self.w
        # We're going to trust that we're being passed a matrix from this
        # decomposition, and just use shape to determine which it is
        elif matrix.shape == self.h.shape or matrix.shape == self.w.shape:
            is_h = matrix.shape == self.h
            matrix = self.h if is_h else self.w
        else:
            raise ValueError(
                "Matrix dimensions do not match either W or H when scaling")

        # Warn if attempting to normalise H by feature, or W by sample
        if by.lower() not in {'feature', 'sample', 'signature'}:
            raise ValueError(
                f"by must be one of 'feature', 'sample', or 'signature'; "
                f"given {by}"
            )
        if is_h and by == "feature":
            logging.warning("H matrix is sample matrix (signatures x samples), "
                            "cannot scale by feature. Scaling by sample "
                            "instead")
            by = "sample"
        if not is_h and by == "sample":
            logging.warning("W matrix is feature matrix (features x "
                            "signatures), cannot scale by feature. Scaling by "
                            "feature instead")
            by = "feature"

        # Perform scaling
        by_sig: bool = by == "signature"
        transpose: bool = (is_h and by_sig) or (not is_h and not by_sig)
        # Transpose so axis to summed is on columns
        matrix = matrix.T if transpose else matrix
        scaled: pd.DataFrame = matrix / matrix.sum()
        return scaled.T if transpose else scaled


    def plot_modelfit(self,
                      group: Optional[pd.Series] = None
                      ) -> plotnine.ggplot:
        """Plot model fit distribution.

        This provides a histogram of the model fit of samples by default. If
        a grouping is provide, this will instead produce boxplots with each
        box being the distribution within a group.

        :param group: Series giving label for group which each sample
            belongs to. Sample which are not in the group series will
            be dropped from results with warning.
        :return: Histogram or boxplots
        """

        # No grouping means histogram
        if group is None:
            return (
                    plotnine.ggplot(
                        self.model_fit.to_frame(name="model_fit"),
                        mapping=plotnine.aes(
                            x="model_fit"
                        )
                    ) +
                    plotnine.geom_histogram() +
                    plotnine.geom_vline(xintercept=self.model_fit.mean()) +
                    plotnine.xlab("Cosine Similarity")
            )

        # Join grouping to model fit
        df: pd.DataFrame = pd.concat([self.model_fit, group], axis=1)
        df.columns = ["model_fit", "group"]
        # Warn if any samples dropped
        # Warn if any sample in grouping not in model fit
        return (
                plotnine.ggplot(df, mapping=plotnine.aes(
                    x="group", y="model_fit")) +
                plotnine.geom_boxplot() +
                plotnine.ylab("Cosine Similarity")
        )

    def plot_relative_weight(self,
                             group: Optional[Union[pd.Series, Iterable]] = None,
                             group_sort: bool = True
                             ) -> plotnine.ggplot:
        """Plot relative weight of each signature in each sample."""

        rel_df: pd.DataFrame = (
            self.scaled('h')
            .T
            .stack()
            .to_frame("weight")
            .reset_index(names=["sample", "signature"])
        )
        plt: plotnine.ggplot = (
            plotnine.ggplot(
                rel_df,
                plotnine.aes(x="sample", y="weight", fill="signature")
            ) +
            plotnine.geom_col(
                position="stack",
                stat="identity"
            ) +
            plotnine.xlab("Sample") +
            plotnine.ylab("Relative Weight") +
            plotnine.scale_fill_manual(self.colors,
                                         name="Signature") +
            plotnine.theme(axis_text_x=plotnine.element_text(angle=90))
        )

        # Display categorical grouping as geom_tile - similar to the ribbon in
        # seaborn. Make as a separate figure and put together with patchworklib
        if group is not None:
            if not isinstance(group, pd.Series):
                group = pd.Series(group, index=self.h.columns)
            group_df: pd.DataFrame = (
                group
                .to_frame("group")
                .reset_index(names=["sample"])
            )
            ordered_scale: plotnine.scale_x_discrete = (
                plotnine.scale_x_discrete(limits=group.index.tolist())
            )
            plt_ribbon: plotnine.ggplot = (
                plotnine.ggplot(group_df,
                                mapping=plotnine.aes(
                                    x="sample",
                                    fill="group"
                                )) +
                plotnine.geom_col(
                    mapping=plotnine.aes(y=1.0),
                    width=1.0
                ) +
                plotnine.scale_fill_discrete(name="Group") +
                ordered_scale +
                plotnine.xlab("") +
                plotnine.ylab("") +
                plotnine.theme(axis_text_x=plotnine.element_text(angle=90),
                               axis_text_y=plotnine.element_blank(),
                               axis_ticks_minor_x=plotnine.element_blank(),
                               axis_ticks_minor_y=plotnine.element_blank()
                               )
            )
            # Remove the x-axis text from the fit plot
            plt = (
                    plt +
                    plotnine.theme(
                        axis_text_x=plotnine.element_blank(),
                        axis_ticks_major_x=plotnine.element_blank(),
                        axis_ticks_minor_x=plotnine.element_blank(),
                        axis_title_x=plotnine.element_blank()
                    ) +
                    ordered_scale
            )
            pw_fit = pw.load_ggplot(plt, figsize=(6, 4))
            pw_cat = pw.load_ggplot(plt_ribbon, figsize=(6, .2))
            plt = pw.vstack(pw_cat, pw_fit,
                            adjust_height=False,
                            direction="t",
                            margin="none")
        return plt

    def plot_pcoa(self, **kwargs):
        """Ordination of samples.

        Perform PCoA of samples based on Bray-Curtis dissimilarity. Bray-Curtis
        dissimilarities are square-root transformed, then Wisconsin double
        standardised (similar to treatment in R `cmdscale`)

        :return: Scatter plot of samples coloured by primary signature"""
        logging.info("Wisconsin double standardising data")
        std_h: pd.DataFrame = self.scaled('h')
        # Species maximum standardisation - each species max is 1
        std_h = (std_h.T / std_h.max(axis=1)).T
        # Sample total standardisation
        std_h = std_h / std_h.sum()

        logging.info("Calculating Bray-Curtis dissimilarity")
        dist: np.ndarray = squareform(pdist(std_h.T, metric="braycurtis"))
        logging.info("Square root transforming dissimilarities")
        dist = np.sqrt(dist)

        nmds = manifold.MDS(
            dissimilarity='precomputed',
            metric=True,
            **kwargs
        )
        pos: np.ndarray = nmds.fit_transform(dist)
        pos_df: pd.DataFrame = pd.DataFrame(
            pos,
            columns=[f'MDS{i}' for i in range(1, pos.shape[1]+1)],
            index=self.h.columns
        )
        pos_df['primary'] = self.primary_signature
        plt: plotnine.ggplot = (
                plotnine.ggplot(
                    pos_df,
                    plotnine.aes(x="MDS1", y="MDS2", color="primary")
                ) +
                plotnine.geom_point() +
                plotnine.scale_color_manual(values=self.colors,
                                            name="Primary Signature")
        )
        return plt

    # def to_csv(self,
    #            dir: pathlib.Path,
    #            prefix: Optional[str] = None,
    #            ):

    def __getattr__(self, item) -> Any:
        """Allow access to parameter attributes through this class as a
        convenience."""
        # Get parameter attributes
        if item in self.parameters._asdict():
            return self.parameters._asdict()[item]
        raise AttributeError()

    def __repr__(self) -> str:
        return (f'Decomposition[rank={self.rank}, '
                f'beta_divergence={self.beta_divergence:.3g}]')

    @staticmethod
    def __representative_signatures(sig: pd.Series,
                                    threshold: float) -> pd.Series:
        """Determine which signatures in a series are representative."""
        sorted_greater: pd.Series = (
            sig.sort_values(ascending=False)
            .cumsum() >= threshold)
        # Determine first index hitting threshold
        first_true: int = 0 if all(sig.isna()) else (next(
            i for i, x in enumerate(sorted_greater.items()) if x[1]) + 1)
        representatives: Set[str] = set(sorted_greater.index[0:first_true])
        return pd.Series(
            sig.index.isin(representatives),
            index=sig.index
        )

    @staticmethod
    def __default_colors(n: int) -> List[str]:
        """Set default colors for signatures.

        By default, Bang Wong's 7 color palette of colorblind distinct colors
        is used (https://www.nature.com/articles/nmeth.1618,
        https://davidmathlogic.com/colorblind). Where more than 7 signatures are
        desired, Sasha Trubetskoy's 20 distinct colours are used
        (https://sashamaps.net/docs/resources/20-colors/) and duplicate with
        warning when there are too many signatures."""
        pal_len: List[int] = [len(x) for x in Decomposition.DEFAULT_SCALES]
        if n <= max(pal_len):
            return next(x for x in Decomposition.DEFAULT_SCALES
                        if n <= len(x))[:n]
        else:
            logging.warning("Some colours are duplicated when plotting over 20 "
                            "signatures")
            return (
                    Decomposition.DEFAULT_SCALES[-1] *
                    math.ceil(n / len(Decomposition.DEFAULT_SCALES))[:n]
            )



