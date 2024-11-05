"""
Generate new models using NMF decomposition

This module provides functions to generate new models from data,
which encompasses three main steps: rank selection, regularisation
selection, and model inspection. The first of these two steps involves
running decompositions multiple times for a range of values, and can be
time-consuming. Methods are provided to run the whole process on a single
machine, but also for running individual decompositions, which are used by the
accompanying nextflow pipeline to allow spreading the computation across
multiple nodes in an HPC environment.

The main functions for each step are

* :func:`rank_selection` and :func:`plot_rank_selection`
* :func:`regu_selection` and :func:`plot_regu_selection`
* :func:`decompositions` which produces :class:`Decomposition` objects

Individual decompositions are represented by a :class:`Decomposition` object,
and visualisation and analysis are carried out using object methods (such as
:meth:`Decomposition.plot_feature_weight()`).
"""
from __future__ import annotations

import collections
import functools
import hashlib
import inspect
import itertools
import logging
import math
import os
import pathlib
import re
import shutil
import tarfile
import time
from functools import reduce
from typing import (Optional, NamedTuple, List, Iterable, Union, Tuple, Set,
                    Any, Dict, Callable, Literal, Hashable, Generator)

import click
import marsilea
import matplotlib.figure
import numpy as np
import pandas as pd
import plotnine
import yaml
from kneed import KneeLocator
import scikit_posthocs as sp
from scipy import sparse
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist, squareform
from skbio import OrdinationResults
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.decomposition._nmf import _beta_divergence
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from cvanmf.reapply import InputValidation, FeatureMatch, _reapply_model

# Initialise logger
logger: logging.Logger = logging.getLogger(__name__)
"""Logger object."""
# from cvanmf import reapply

# Type aliases
Numeric = Union[int, float]
"""Alias for python numeric types (a union of int and float)."""
PcoaMatrices = Literal['w', 'x', 'wh', 'signatures']
"""Allowed matrices which PCoA can be constructed from. Allows values w, x, 
wh, signatures (alias for w)."""

DEF_SELECTION_ORDERING: List[str] = [
    "cosine_similarity",
    "r_squared",
    "reconstruction_error",
    "l2_norm",
    "sparsity_w",
    "sparsity_h",
    "rss"
]
"""Default ordering for rank selection and regularisation selection plots."""
DEF_PCOA_POINT_AES: Dict[str, Any] = dict(
    size=2,
    alpha=0.8
)
"""Default geom_point fixed aesthetics for PCoA plots"""
DEF_RELATIVE_WEIGHT_HEIGHTS: Dict[str, float] = dict(
    bar=1.0,
    ribbon=.2,
    dot=.5,
    label=.5
)
"""Default heights for relative weight plot"""
DEF_ALPHAS: List[float] = [2 ** i for i in range(-5, 2)]
"""Default alpha values for regularisation selection"""
DEF_RANKSEL_PALETTE: plotnine.scale_fill_brewer = (
    plotnine.scale_fill_brewer(type="qual", limits=DEF_SELECTION_ORDERING,
                               palette="Set2")
)


class BicvFold(NamedTuple):
    """One fold from a shuffled matrix

    The submatrices have been joined into the structure shown below

    .. code-block:: text

        A B . B
        C D . D
        . . . .
        C D . D

    from which A will be estimated as A' using only B, C, D.
    ```
    """
    A: pd.DataFrame
    B: pd.DataFrame
    C: pd.DataFrame
    D: pd.DataFrame


class BicvSplit:
    """Shuffled matrix for bi-cross validation, split into mn matrices in a
    m x n pattern. To shuffle and split an existing matrix, use the static
    method :method:`BicvSplit.from_matrix`"""

    def __init__(
            self,
            mx: List[pd.DataFrame],
            design: Tuple[int, int],
            i: Optional[int] = None,
    ) -> None:
        """Create a shuffled matrix containing the 9 split matrices. These
        should be in the order

        .. code-block:: text

            0, 1, 2
            3, 4, 5
            6, 7, 8

        :param mx: Split matrices in a flat list. These should be all of one
            row, then all of the next row etc.
        :param design: Number of even splits on rows and columns of mx.
        :param i: Index of the split if one of many
        """

        # Some validations
        self.__mx: List[List[pd.DataFrame]] = self.__validate_mx(
            mx=mx, design=design
        )
        self.__design: Tuple[int, int] = design
        self.i = i

    @staticmethod
    def __validate_mx(
            mx: List[pd.DataFrame],
            design: Tuple[int, int]
    ) -> List[List[pd.DataFrame]]:
        m, n = design
        if len(mx) != m * n:
            raise ValueError(
                "The number of submatrices provided must match the design. "
                f"Design {design}, expected {m * n} matrices, received "
                f"{len(mx)}"
            )
        mx_d: List[Tuple] = list(itertools.batched(mx, n))
        # Ensure indices match along columns and rows
        row_match: Generator[bool] = (
            all((row[0].index == d.index).all() for d in row[1:])
            for row in mx_d
        )
        col_match: Generator[bool] = (
            all((mx_d[0][col].columns == d.columns).all() for d in
                (mx_d[r][col] for r in range(1, n))
                )
            for col in range(n)
        )
        if not all(row_match):
            raise ValueError(
                "Indices of submatrices in the same row do not match. Check "
                "that you are providing submatrices in the correct order (all "
                "submatrices of the same row should be consecutive)"
            )
        if not all(col_match):
            raise ValueError(
                "Column indices of submatrices in the same column do not "
                "match. Check that you are providing submatrices in the "
                "correct order (all submatrices of the same row should be "
                "consecutive)"
            )
        # Check that block sizes are roughly even. It is okay if they are not,
        # user may have an unusual custom design, but should warn.
        row_sizes: List[int] = [x[0].shape[0] for x in mx_d]
        col_sizes: List[int] = [x.shape[1] for x in mx_d[0]]
        # Allow to be within the remainder
        total_col, total_row = sum(col_sizes), sum(row_sizes)
        c_r, r_r = total_col % len(col_sizes), total_row % len(row_sizes)
        if max(row_sizes) - min(row_sizes) > r_r:
            logger.warning("Uneven row sizes in bicrossvalidation design: %s. "
                           "If you did not set up a custom design this is "
                           "likely an error.", row_sizes)
        if max(col_sizes) - min(col_sizes) > c_r:
            logger.warning("Uneven column sizes in bicrossvalidation design: "
                           "%s. If you did not set up a custom design this is "
                           "likely an error.", col_sizes)
        return [list(x) for x in mx_d]

    # TODO: Better slicing implementation
    @property
    def mx(self) -> List[List[pd.DataFrame]]:
        """Submatrices as a 2d list."""
        return self.__mx

    @property
    def design(self) -> Tuple[int, int]:
        """Design of holdout pattern, given as (rows, columns)."""
        return self.__design

    @property
    def x(self) -> pd.DataFrame:
        """The input matrix

        This reproduces the input matrix by concatenating the submatrices.

        :returns: Input matrix
        """
        return pd.concat([self.row(i, join=True) for i in
                          range(self.design[0])], axis=0)

    @property
    def i(self) -> int:
        """This is the i-th shuffle of the input."""
        return self.__i

    @i.setter
    def i(self, i: Optional[int]) -> None:
        self.__i: Optional[int] = int(i) if i is not None else None

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

    @property
    def num_folds(self) -> int:
        """Total number of folds in this design."""
        return self.design[0] * self.design[1]

    def __block_orderings(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Generate ordering of indices to be used in generating folds.

        For the purpose of generating folds, we want each block to take the
        top left position once. The order of the rows (or columns) other than
        what is first does not matter.
        """

        m, n = self.design
        mc: List[List[int]] = [
            [x] + [i for i in range(m) if x != i] for x in range(m)
        ]
        nc: List[List[int]] = [
            [x] + [i for i in range(n) if x != i] for x in range(n)
        ]
        return mc, nc


    def row(self,
            row: int,
            join: bool = False
            ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """Get a row of the submatrices by index. Convenience method for
        readability.
        
        :param row: Row index to get
        :param join: Join into a single matrix
        :returns: List of submatrices making up the row, or these submatrices
            joined if requested.
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
        :param join: Join into a single matrix
        :returns: List of submatrices making up the column, or these submatrices
            joined if requested.
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
        # Save submats along with their position as key
        submat_dict: Dict[str, pd.DataFrame] = {
            f'{m},{n}': self.mx[m][n] for m, n
            in itertools.product(range(self.design[0]), range(self.design[1]))
        }
        save_fn(path, **submat_dict)

    def fold(self,
             i: int) -> BicvFold:
        """Construct a fold of the data

        There are m*n possible folds of the data, this function constructs the
        i-th fold.

        :param i: Index of the fold to construct, from [0, mn)
        :returns: A, B, C, and D matrices for this fold
        """
        if i >= self.design[0] * self.design[1]:
            raise IndexError(f"Fold index out of range.")
        row, col = divmod(i, self.design[1])
        row_indices, col_indices = self.__block_orderings()
        row_idx, col_idx = (row_indices[row],
                            col_indices[col])
        # Make a rearranged mx list
        mx_i: List[List[pd.DataFrame]] = [
            [self.mx[ri][ci] for ci in col_idx] for ri in row_idx
        ]
        # Make submatrices
        a: pd.DataFrame = mx_i[0][0]
        b: pd.DataFrame = pd.concat(mx_i[0][1:], axis=1)
        c: pd.DataFrame = pd.concat([mx_i[i][0] for
                                     i in range(1, self.design[0])],
                                    axis=0)
        d: pd.DataFrame = pd.concat(
            [pd.concat(mx_i[i][1:], axis=1) for
             i in range(1, self.design[0])],
            axis=0
        )
        return BicvFold(a, b, c, d)

    @property
    def folds(self) -> List[BicvFold]:
        """List of the mn possible folds of these submatrices."""
        return [self.fold(i) for i in range(self.design[0] * self.design[1])]

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
        :param fix_i: Renumber all splits starting from 0. Does not check
            if existing numbering is unique.
        :param compress: Use compression
        :param force: Overwrite existing files
        """
        # Splits are likely to be passed as a generator, so we can't count,
        # index, or check all contents of this object. This is to prevent
        # shuffles of large matrices having to all be held in memory.
        split: BicvSplit
        num_splits: int = 0
        for i, split in enumerate(splits):
            if fix_i:
                split.i = i
            split.save_npz(path=path, compress=compress, force=force)
            num_splits = i
        logger.info("%s shuffles saved to %s", num_splits+1, path)

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
            # The name of each matrix will be the index m, n
            m, n = (max([int(y) for y in x.split(',')][0] for x in mats.keys()),
                    max([int(y) for y in x.split(',')][1] for x in mats.keys()))
            # BicvSplit expects matrices in flat form, with all matrices of
            # same row first
            sorted_mats: List[np.ndarray] = [
                mats[f'{i},{j}'] for i, j
                in itertools.product(range(m+1), range(n+1))
            ]
            return BicvSplit(
                [pd.DataFrame(x) for x in sorted_mats],
                i=use_i,
                design=(m+1, n+1)
            )

    @staticmethod
    def load_all_npz(
            path: Union[pathlib.Path, str],
            allow_pickle: bool = True,
            fix_i: bool = False
    ) -> Generator[BicvSplit]:
        """Read shuffles from files.

        Reads either all the npz files in a directory, or those specified by a
        glob. The expectation is the filenames are in format prefix_i.npz,
        where i is the number of this shuffle. If not, use fix_i to renumber
        in order loaded.

        :param path: Directory with .npz files, or glob identifying .npz files
        :param allow_pickle: Allow unpickling when loading; necessary for
            compressed files.
        :param fix_i: Renumber shuffles.
        :returns: Generator of BicvSplit objects.
        """
        npz_pth: pathlib.Path = (
            path if isinstance(path, pathlib.Path)
            else
            pathlib.Path(path)
        )
        logger.info("Loading shuffles from %s", str(npz_pth))
        # Ensure shuffles are returned in the same order as saved
        files: List[pathlib.Path] = sorted(
            list(npz_pth.glob("*.npz")),
            key=lambda x: int(x.stem.split("_")[-1])
        )

        for i, file in enumerate(files):
            shuffle: BicvSplit = BicvSplit.load_npz(path=pathlib.Path(file),
                                                    allow_pickle=allow_pickle)
            if fix_i:
                shuffle.i = fix_i
            yield shuffle

    @staticmethod
    def from_matrix(
            df: pd.DataFrame,
            n: int = 1,
            design: Tuple[int, int] = (3, 3),
            random_state: Optional[Union[int, np.random.Generator]] = None
    ) -> Generator[BicvSplit]:
        """Create random shuffles and splits of a matrix

        :param df: Matrix to shuffle and split
        :param n: Number of shuffles
        :param design: Number of blocks to divide rows and columns into.
            Default is 3x3 9-fold bicrossvalidation.
        :param random_state: Random state, either int seed or numpy Generator;
            None for default numpy random Generator.
        :returns: A generator of  splits, as BicvSplit objects
        """
        rnd: np.random.Generator = np.random.default_rng(random_state)
        logger.info(
            "Generating %s splits with state %s", str(n), str(rnd))

        shuffles: Iterable[pd.DataFrame] = (
            map(BicvSplit.__bicv_shuffle, (df for _ in range(n)))
        )
        splits: Iterable[Iterable[pd.DataFrame]] = (
            map(BicvSplit.__bicv_split, shuffles, [design] * n)
        )
        return (BicvSplit(x, design, i) for i, x in enumerate(splits))

    @staticmethod
    def __bicv_split(
            df: pd.DataFrame,
            design: Tuple[int, int]
    ) -> List[pd.DataFrame]:
        """Make an m x n split of a matrix for bi-cross validation. This is
        adapted from Clemence's work in bicv_rank.py at
        https://gitlab.inria.fr/cfrioux/enterosignature-paper, and generalised
        to non 3x3 designs..

        :param df: Matrix to split
        :param design: Number of even splits to produce of row and columns.
        :return: Matrix split into m x n sections
        """
        m, n = design
        # Cut the m x n submatrices
        chunks_feat: int = df.shape[0] // m
        chunks_sample: int = df.shape[1] // n
        thresholds_feat: List[int] = [chunks_feat * i for i in range(1, m)]
        thresholds_feat.insert(0, 0)
        thresholds_feat.append(df.shape[0] + 1)
        thresholds_sample: List[int] = [chunks_sample * i for i in range(1, n)]
        thresholds_sample.insert(0, 0)
        thresholds_sample.append(df.shape[1] + 1)

        # Return the mn matrices
        sub_mats: List[pd.DataFrame] = []
        for row in range(m):
            for col in range(n):
                sm: pd.DataFrame = (
                    df.iloc[
                        thresholds_feat[row]:
                        thresholds_feat[row + 1],
                        thresholds_sample[col]:
                        thresholds_sample[col + 1]
                    ]
                )
                sub_mats.append(sm)
        return sub_mats

    @staticmethod
    def __bicv_shuffle(
            df: pd.DataFrame,
            random_state: Optional[Union[np.random.Generator, int]] = None
    ) -> pd.DataFrame:
        """Perform a single shuffle of a matrix.

        :param df: Matrix to shuffle
        :param random_state: Randomisation state. Either a numpy Generator,
            an integer seed for a Generator, or None for the default generator.
        :returns: Shuffled matrix
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
    validation iteration, this should be the shuffled matrix split into mn 
    parts, where m is the number of parts along rows, n along columns. When 
    returning results and keep_mats is False, this will be set to None to
    avoid passing and saving large data."""
    rank: int
    """Rank of the decomposition."""
    seed: Optional[Union[int, np.random.Generator, str]] = None
    """Random seed for initialising decomposition matrices; if None no seed
    used so results will not be reproducible."""
    alpha: float = 0.0
    """Regularisation parameter applied to both :math:`H` and :math:`W` 
    matrices."""
    l1_ratio: float = 0.0
    """Regularisation mixing parameter. In range 0.0 <= l1_ratio <= 1.0."""
    max_iter: int = 3000
    """Maximum number of iterations during decomposition. Will terminate earlier
    if solution converges."""
    keep_mats: bool = False
    """Whether to return the :math:`H` and :math:`W` matrices as part of the 
    results."""
    beta_loss: str = "kullback-leibler"
    """Beta loss function for NMF decomposition."""
    init: str = "nndsvdar"
    """Initialisation method for :math:`H` and :math:`W` matrices on first step.
     Defaults to randomised non-negative SVD with small random values added to 
     0s."""

    @property
    def log_str(self) -> str:
        """Format parameters in readable way for logs/console."""
        data_str: str = (repr(self.x) if isinstance(self.x, BicvSplit) else
                         f"DataFrame[{self.x.shape}]")
        return (
            f"Bi-cross validation parameters\n"
            f"------------------------------\n"
            f"Data:             {data_str}"
            f"Seed:             {self.seed}\n"
            f"Ranks:            {self.rank}\n"
            f"L1 Ratio:         {self.l1_ratio}\n"
            f"Alpha:            {self.alpha}\n"
            f"Max Iterations:   {self.max_iter}\n"
            f"Beta Loss:        {self.beta_loss}\n"
            f"Initialisation:   {self.init}\n"
            f"Keep Matrices:    {self.keep_mats}\n"
        )

    def to_yaml(self,
                path: pathlib.Path):
        """Write parameters to a YAML file.

        Save the parameters, except the input matrix, to a YAML file.

        :param path: File to write to
        """

        # A little bit of work to coerce types. Numpy random generators
        # provide a int32/int64, where manually set seed are python int
        # whose type cannot be checked by np.issubdtype
        seed: Union[int, str] = self.seed
        if not (isinstance(seed, int) or isinstance(seed, str)):
            # This is likely a numpy int type which we will cast to int
            seed = int(seed)
        with open(path, 'w') as f:
            yaml.safe_dump(dict(
                seed=seed,
                rank=self.rank,
                l1_ratio=self.l1_ratio,
                alpha=self.alpha,
                max_iter=self.max_iter,
                beta_loss=self.beta_loss,
                init=self.init,
                keep_mats=self.keep_mats
            ), f)


class BicvResult(NamedTuple):
    """Results from a single bi-cross validation run. For each BicvSplit there
    are :math:`mn` folds (for :math:`m` splits on rows, :math:`n` splits on
    columns), for which the top left submatrix (:math:`A`) is estimated
    (:math:`A'`) using the other portions."""
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
    i: int
    """Shuffle number when there are multiple shuffles. Included to allow 
    spreading bicv across multiple processes, but without needing to return
    a copy of the full matrix."""

    def __drop_mats(self) -> BicvResult:
        """Remove all matrices from results tuple if keep_mats is False,
        otherwise just return self."""
        if self.parameters.keep_mats:
            return self
        if self.parameters.x is None and self.a is None:
            return self
        # Drop A and X matrices. For merge operator | value in last dict
        # takes precedence
        param_prime: NMFParameters = NMFParameters(
            **(self.parameters._asdict() | dict(x=None))
        )
        return BicvResult(
            **(self._asdict() | dict(parameters=param_prime, a=None))
        )

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
            i=results[0].i,
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
        ).__drop_mats()

    def to_series(self,
                  summarise: Callable[[np.ndarray], float] = np.mean
                  ) -> pd.Series:
        """Convert bi-fold cross validation results to series

        :param summarise: Function to reduce each the measures (r_squared etc)
            to a single value for each shuffle.
        :returns: Series with entry for each non-parameter measure
        """
        exclude: Set[str] = {"a", "parameters", "i"}
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
        params['shuffle_num'] = self.i
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
        :param summarise: Function to reduce each the measures (r_squared etc)
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
                   design: Optional[Tuple[int, int]] = (3, 3),
                   progress_bar: bool = True) -> Dict[int, List[BicvResult]]:
    """Bi-cross validation for rank selection.

    Run :math:`mn`-fold bicrossvalidation across a range of ranks. Briefly, the
    input matrix is shuffled `shuffles` times. Each shuffle is then split
    into :math:`m\timesn` submatrices (:math:`m` splits on rows, :math:`n`
    splits on columns). The rows and columns of submatrices are permuted, and
    the top left submatrix (:math:`A`) is estimated through NMF decompositions
    of the other matrices producing an estimate :math:`A'`. Various measures of
    how well :math:`A'` reconstructed :math:`A` are provided, see
    :class:`BicvResult` for details on the measures.

    No multiprocessing is used, as a majority of build of scikit-learn seem to
    make good use of multiple processors anyway (depending on compilation of
    underlying libraries and matrix size).

    This method returns a dictionary with each rank as a key, and a list
    containing one :class:`BicvResult` for each shuffle.

    :param x: Input matrix.
    :param ranks: Ranks of k to be searched. Iterable of unique ints.
    :param shuffles: Number of times to shuffle `x`.
    :param keep_mats: Return A' and shuffle as part of results.
    :param seed: Random value generator or seed for creation of the same.
        If not provided, will initialise with entropy from system.
    :param alpha: Regularisation coefficient
    :param l1_ratio: Ratio between L1 and L2 regularisation. L2 regularisation
        (0.0) is densifying, L1 (1.0) sparisfying.
    :param max_iter: Maximum iterations of NMF updates. Will end early if
        solution converges.
    :param beta_loss: Beta-loss function, see sklearn documentation for
        details.
    :param init: Initialisation method for H and W during decomposition.
        Used only where one of the matrices during bi-cross steps is not
        fixed. See sklearn documentation for values.
    :param design: How many blocks to split the input matrix into on rows and
        columns respectively. Defaults to 3x3 9-fold design.
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
    if design is None:
        design = (3, 3)

    # Set up random generator for this whole run
    rng: np.random.Generator = np.random.default_rng(seed)

    # Generate shuffles of data
    shuffle_gen: Generator[BicvSplit] = BicvSplit.from_matrix(
        x, random_state=rng, n=shuffles, design=design)

    rank_list: List[int] = sorted(list(ranks), reverse=True)

    # Make a generator of parameter objects to pass to bicv
    params: Iterable[NMFParameters] = itertools.chain.from_iterable(
        ((NMFParameters(x=x, rank=k, seed=rng, **args) for k in rank_list)
         for x in shuffle_gen)
    )


    # Get results
    # No real use in multiprocessing, sklearn implementation generally makes
    # good use of multiple cores anyway. Multiprocessing just makes processes
    # compete for resources and is slower.
    total_runs: int = len(rank_list) * shuffles
    res_map: Iterable = (
        map(bicv, tqdm(params, total=total_runs))
        if progress_bar else
        map(bicv, params)
    )
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
        logger.error(("Uneven number of results returned for each rank, "
                       "some rank selection iterations may have failed."))

    return grouped_results


def suggest_rank(
        rank_selection_results: Union[Dict[int, List[BicvResult]],
        pd.DataFrame],
        summarise: Callable[[np.ndarray], float] = np.mean,
        measures: List[str] = ['cosine_similarity', 'r_squared'],
        **kwargs
) -> Dict[str, int]:
    """Suggest a suitable rank.

    Attempt to identify an elbow point in the graphs of cosine similarity
    and :math:`R^2` which represent points where the rate of improvement in
    the decomposition slows.

    Please note this is only a suggestion of a suitable rank; the plots
    should still be inspected and decompositions of candidate ranks inspected to
    make a final decision.

    This is implemented using the excellent `kneed` package, and `**kwargs`
    are passed to the constructor of `KneeLocator`, you can use this if you
    wish to customise the behaviour. We use the online mode of kneed by default.

    :param rank_selection_results: Results from :func:`rank_selection`, or
        these results in DataFrame format from
        :meth:`BicvResult.results_to_table`
    :param summarise: Function to summarise results from a shuffle. Roughly
        speaking, determines which point represent the middle of the
        distribution of values for purposes of the curve.
    :param measures: The measures to consider if passed a DataFrame
    :param kwargs: Arguments passed to ``KneeLocator`` constructor
    """
    df: pd.DataFrame = (
        rank_selection_results if isinstance(rank_selection_results,
                                             pd.DataFrame)
        else BicvResult.results_to_table(
            rank_selection_results, summarise=summarise, **kwargs)
    )
    return (
        df[['rank'] + measures]
        .groupby('rank')
        .median()
        .apply(__detect_elbow, **kwargs)
        .to_dict()
    )


def suggest_rank_stability(
        rank_selection_results: Union[pd.DataFrame, Iterable[pd.Series],
                                Dict[int, List[Decomposition]]],
        measures: List[str] = ['cophenetic_correlation', 'dispersion',
                               'signature_similarity'],
        near_max: float = 0.02,
        **kwargs
) -> Dict[str, int]:
    """Suggest a suitable rank in stability based measures.

    Attempt to identify peaks in stability based rank selection criteria
    (cophenetic correlation, dispersion, signature similrity). By default the
    highest peak is selected. Where there are many similar ranks (defined by
    `near_max`), the one with the most consecutively decreasing values after it
    is selected.

    Please note this is only a suggestion of a suitable rank; the plots
    should still be inspected and decompositions of candidate ranks inspected to
    make a final decision.

    When making a plot multiple times (changing parameters etc), it may be
    preferable to calculate the measures then pass the results as a list of
    Series, as the calculation can be time consuming.

    :param rank_selection_results: Results from :func:`decompositions`,
        or a collection of series produced by :func:`dispersion`,
        :func:`cophenetic_correlation`, and :func:`signature_similarity`, or a
        DataFrame of those series joined.
    :param measures: The measures to consider if passed a DataFrame
    :param near_max: Consider peaks (:math:`p`) candidates if they are within a
        certain distance of global maximum (:math:`gm`):
        :math:`p \geq gm(1-near_max)`.
    :param kwargs: Passed to ``np.argrelmax``.
    """

    df: pd.DataFrame
    if isinstance(rank_selection_results, pd.DataFrame):
        df = rank_selection_results
    elif isinstance(rank_selection_results, dict):
        df = pd.concat([dispersion(rank_selection_results),
                        cophenetic_correlation(rank_selection_results),
                        signature_similarity(rank_selection_results)],
                       axis=1
                       ).reset_index(names=['rank'])
    else:
        df = pd.concat(list(rank_selection_results), axis=1).reset_index(
            names=['rank'])

    measures = list(set(measures).intersection(set(df.columns)))
    return (
        df[['rank'] + measures]
        .set_index('rank')
        .apply(__detect_max, **kwargs)
        .to_dict()
    )


def __detect_elbow(
        series: pd.Series,
        concave: bool = True,
        increasing: bool = True,
        **kwargs
) -> float:
    """Return estimated elbow point in a series of values using the package
    kneed. Uses the online mode by default. Pass online=False to disable."""
    direction: str = 'increasing' if increasing else 'decreasing'
    curve: str ='concave' if concave else 'convex'
    if 'online' not in kwargs:
        kwargs['online'] = True
    locator: KneeLocator = KneeLocator(
        x=series.index,
        y=series.values,
        direction=direction,
        curve=curve,
        **kwargs
    )
    return locator.knee


def __detect_max(
        series: pd.Series,
        near_max: float = 0.02,
        **kwargs
) -> float:
    """Select rank based on maxima

    :param series: Values for each rank, sorted in rank order
    :param near_max: Any maxima within 0.02 of maximum are considered
    candidates, and scored based on length of monotonically decreasing tail.
    :param kwargs: Passed to argrelmax.
    :returns: Series with index being rank, sorted by value.
    """
    from scipy.signal import argrelmax
    # Default to wrap to get peaks at 2
    # Remove rank 1 if it is present (if old results which allowed rank 1 in
    # stability results)
    # Sort rank to ensure series in ascending rank order
    series = series.sort_index(ascending=False)
    maxima_idx: np.ndarray = argrelmax(
        data=series.values,
        **(dict(mode="wrap") | kwargs))[0]
    maxima_vals: np.ndarray = series.values[maxima_idx]
    near_max: np.ndarray = maxima_vals >= ((1 - near_max) * maxima_vals.max())
    candidates: np.ndarray = maxima_idx[near_max]
    if len(candidates) == 1:
        return series.index[candidates[0]]
    if len(candidates) < 1:
        logging.error("No candidates found during stability rank suggestion.")
    logger.info("Multiple candidates (%s) found, selecting on length of "
                "monotonically decreasing values after.", maxima_idx.tolist())
    decreasing: np.ndarray = np.diff(series.values) < 0
    selected_idx, max_tail_length = 0, -1
    for candidate_idx in candidates:
        subs: np.ndarray = decreasing[candidate_idx:]
        tail_length: int
        if sum(~subs) == 0:
            tail_length = len(subs)
        else:
            tail_length = np.where(~subs)[0][0]
        logger.debug("Candidate %s, tail length %s", candidate_idx,
                     tail_length)
        if tail_length > max_tail_length:
            selected_idx, max_tail_length = candidate_idx, tail_length
    return series.index[selected_idx]


def plot_rank_selection(results: Dict[Union[int, float], List[BicvResult]],
                        exclude: Optional[Iterable[str]] = None,
                        include: Optional[Iterable[str]] = None,
                        show_all: bool = False,
                        geom: str = 'box',
                        summarise: Literal['mean', 'median'] = 'mean',
                        suggested_rank: bool = True,
                        stars_at: Optional[Dict[str, int]] = None,
                        star_size: int = 4,
                        jitter: bool = None,
                        jitter_size: float = 0.3,
                        n_col: int = None,
                        xaxis: str = "rank",
                        rotate_x_labels: Optional[float] = None,
                        geom_params: Dict[str, Any] = None,
                        **kwargs
                        ) -> plotnine.ggplot:
    """Plot rank selection results from bicrossvalidation.

    Draw either box plots or violin plots showing statistics comparing
    :math:`A` and :math:`A'` from all bicrossvalidation results across a
    range of ranks.
    The plotting library used is ``plotnine``; the returned plot object
    can be saved or drawn using ``plt_obj.save`` or ``plt_obj.draw``
    respectively.
    By default, only `cosine_similarity` and `r_squared` are plotted. You can
    define which measures to include using include, or which to exclude using
    exclude. You can also use show_all to show all the measures.

    For `cosine_similarity` and `r_squared`, an suggestion of optimal rank
    is given by identifying an elbow point in the graph using the package
    ``kneed``, indicated by a star above that rank.

    :param results: Dictionary of results, with rank as key and a list of
        :class:`BicvResult` for that rank as value
    :param exclude: Measures from :class:`BicvResult` not to plot.
    :param include: Measures from :class:`BicvResult` to plot.
    :param show_all: Show all measures, ignoring anything set in include or
        exclude.
    :param geom: Type of plot to draw. Accepts either 'box' or 'violin'
    :param summarise: How to summarise the statistics across the folds
        of a given shuffle.
    :param suggested_rank: Estimate rank using :func:`suggest_rank`.
    :param stars_at: Manually define x-axis values at which to place stars
        above the main plot. Mainly used to allow :func:`plot_regu_selection`
        to pass where to plot stars for regularisation selection.
    :param star_size: Size of star indicating suggested rank.
    :param jitter: Draw individual points for each shuffle above the main plot.
    :param jitter_size: Size of jitter points.
    :param n_col: Number of columns in the plot. If blank, attempts to guess
        a sensible value.
    :param xaxis: Value to plot along the x-axis. "rank" for rank selection,
        "alpha" for regularisation selection.
    :param rotate_x_labels: Degrees to rotate x-axis labels by. If None
        will rotate if x-axis is float.
    :param **kwargs: Passed to :func:`suggest_ranks`.
    :return: :class:`plotnine.ggplot` instance
    """
    # Intended as a user friendly interface to plot rank selection results
    # so takes string argument rather than functions etc.
    if summarise not in ['mean', 'median']:
        raise ValueError("summarise must be one of mean, median")
    if include is None and exclude is None:
        include = {'cosine_similarity', 'r_squared'}
    exclude = {} if exclude is None else exclude
    include = {} if include is None else include
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
    measures: set[str] = set(
        name for (name, value) in
        inspect.getmembers(
            BicvResult, lambda x: isinstance(x, collections._tuplegetter))
    ).difference({"a", "parameters", "i"})
    if not show_all:
        # First exclude anything explicitly requested to be removed in exclude
        measures = measures.difference(exclude)
        if len(include) > 0:
            # Subset to the intersection of all measures and those requested
            measures = measures.intersection(include)
    # Add the xaxis (could be rank or alpha)
    measures = list(measures.union({xaxis}))
    # Get results and stack so measure is a column
    full_df: pd.DataFrame = BicvResult.results_to_table(
        results, summarise=summarise_fn)
    df: pd.DataFrame = full_df[measures]
    # Make longer so we have measure as a column
    stacked: pd.DataFrame = (
        df
        .set_index(xaxis)
        .stack(dropna=False)
        .to_frame(name='value')
        .reset_index(names=[xaxis, "measure"])
    )
    # Make an ordering for the measures to be included
    measures: Set[str] = set(stacked['measure'].unique())
    facet_order: List[str] = [x for x in DEF_SELECTION_ORDERING if x in
                              measures]
    facet_order += list(measures.difference(set(facet_order)))
    stacked['measure'] = pd.Categorical(
        stacked['measure'],
        ordered=True,
        categories=facet_order
    )
    # Parameters for selected geom
    geom_params = {} if geom_params is None else geom_params

    # Plot
    plot: plotnine.ggplot = (
            plotnine.ggplot(
                data=stacked,
                mapping=plotnine.aes(
                    x=f"factor({xaxis})",
                    y="value"
                )
            )
            + plotnine.facet_wrap(facets="measure", scales="free_y",
                                  ncol=n_col)
            + requested_geom(mapping=plotnine.aes(fill="measure"),
                             **geom_params)
            + plotnine.xlab(xaxis)
            + plotnine.ylab("Value")
            + plotnine.guides(fill="none")
            + DEF_RANKSEL_PALETTE
    )

    # Add manually defined star positions
    if stars_at is not None:
        logger.debug("Adding stars from stars_at parameter")
        star_df: pd.DataFrame = (
            pd.Series(stars_at)
            .to_frame(xaxis)
            .reset_index(names=['measure'])
        )
        plot = __add_stars(
            plot,
            star_df=star_df,
            size=star_size,
            offset=0.1
        )
    # Add automated rank suggestion if requested
    if xaxis =="rank" and suggested_rank == True:
        logger.debug("Adding autoamtic rank suggestion stars")
        suggested: Dict[str, int] = suggest_rank(full_df, **kwargs)
        suggested_df: pd.DataFrame = (
            pd.Series(suggested)
            .to_frame('rank')
            .reset_index(names='measure')
        )
        suggested_df = suggested_df.loc[suggested_df['measure'].isin(measures)]
        plot = __add_stars(plot,
                          star_df=suggested_df[['rank', 'measure']],
                          size=star_size,
                          offset=0.1)

    # Determine whether the xaxis values are floats, and so axis labels should
    # be rounded for display
    xaxis_float: bool = np.issubdtype(stacked[xaxis].dtype, np.floating)
    if xaxis_float:
        rotate_x_labels = 90.0 if rotate_x_labels is None else rotate_x_labels
        plot = (
                plot
                + plotnine.scale_x_discrete(
            labels=lambda x: [f'{i:.2e}' for i in x]
        )
                + plotnine.theme(
            axis_text_x=plotnine.element_text(rotation=rotate_x_labels)
        )
        )

    # If not specifically requested, decide whether to add a jitter
    if jitter is None:
        jitter = False if len(list(results.values())[0]) > 50 else True
    if jitter:
        plot = plot + plotnine.geom_jitter(size=jitter_size)

    return plot

@staticmethod
def __add_stars(
        plt: plotnine.ggplot,
        star_df: pd.DataFrame,
        size: float,
        offset: float
) -> plotnine.ggplot:
    """Add stars indicating suggested value to a plot showing selection
    results. star_df should have two olumns: measure, defining the facet
    value to plot the star on; a second column with any name giving the
    x-axis position to plot on."""

    x_lab: str = next(x for x in star_df.columns if x != 'measure')

    star_ypos: pd.DataFrame = (
            plt.data[['measure', 'value']]
            .groupby(['measure'])
            .apply(lambda x: x.max() + (offset*(x.max()-x.min())))
        )
    # Using categorical x-axes so need to convert to index of value
    val_ordered: List = sorted(plt.data[x_lab].unique())
    star_df['idx'] = star_df[x_lab].apply(
        lambda x: (np.nan if x is None or np.isnan(x) else
                   val_ordered.index(x) + 1))
    plt_df: pd.DataFrame = star_df.merge(
        star_ypos,
        left_on='measure',
        right_on='measure',
        how='left'
    )
    plt = (
        plt
        + plotnine.geom_point(
            data=plt_df,
            mapping=plotnine.aes(x="idx", y="value", fill="measure"),
            color="black",
            shape="*",
            size=size
        )
    )
    return plt


def regu_selection(x: pd.DataFrame,
                   rank: int,
                   alphas: Optional[Iterable[float], None] = None,
                   scale_samples: Optional[bool] = None,
                   shuffles: int = 100,
                   keep_mats: Optional[bool] = None,
                   seed: Optional[Union[int, np.random.Generator]] = None,
                   l1_ratio: Optional[float] = 1.0,
                   max_iter: Optional[int] = None,
                   beta_loss: Optional[str] = None,
                   init: Optional[str] = None,
                   design: Tuple[int, int] = (3, 3),
                   progress_bar: bool = True
                   ) -> Tuple[float, Dict[float, List[BicvResult]]]:
    """Bicrossvalidation for regularisation selection.

    Run :math:`mn`-fold bicrossvalidation across a range of regularisation
    ratios, for a single rank. For a brief description of bi-cross validation
    see :func:`rank_selecton`

    No multiprocessing is used, as a majority of build of scikit-learn
    seem to make good use of multiple processors anyway.

    This method returns a tuple with

    * a float which is the tested alpha which meets the criteria in
      the ES paper
    * a dictionary with each alpha value as a key, and a list containing one
      :class:`BicvResult` for each shuffle

    :param x: Input matrix.
    :param rank: Rank of decomposition.
    :param alphas: Regularisation alpha parameters to be searched. If left
        blank a default range will be used.
    :param scale_samples: Divide alpha by number of samples. This is provided
        as the way regularisation is performed changed in newer sklearn
        versions, and alpha is multiplied by n_samples. Setting this to True
        results in the same calculation as earlier sklearn versions, such as
        the one used in the Enterosignatures paper. If this is set it is
        honoured; if left as None, when automatic alpha range is calculated
        they will be scaled by sample, when alpha range specified will not be
        scaled.
    :param shuffles: Number of times to shuffle `x`.
    :param keep_mats: Return :math:`A'` and `shuffle` as part of results.
    :param seed: Random value generator or seed for creation of the same.
        If not provided, will initialise with entropy from system.
    :param alpha: Regularisation coefficient
    :param l1_ratio: Ratio between L1 and L2 regularisation. L2 regularisation
        (0.0) is densifying, L1 (1.0) sparisfying.
    :param max_iter: Maximum iterations of NMF updates. Will end early if
        solution converges.
    :param beta_loss: Beta-loss function, see sklearn documentation for
        details.
    :param init: Initialisation method for H and W during decomposition.
        Used only where one of the matrices during bi-cross steps is not
        fixed. See sklearn documentation for values.
    :param progress_bar: Show a progress bar while running.
    :param design: Number of blocks to split input into on rows and columns
        respectively for bicrossvalidation.
    :returns: Dictionary with entry for each rank, containing a list of
        results for each shuffle (as a :class:`BicvResult` object)
    """
    # TODO: Reduce repeated code between this and rank_selection()
    # Set up a dictionary of arguments
    args: Dict[str, Any] = {}
    if keep_mats is not None:
        args['keep_mats'] = keep_mats
    if rank is not None:
        args['rank'] = rank
    if l1_ratio is not None:
        args['l1_ratio'] = l1_ratio
    if max_iter is not None:
        args['max_iter'] = max_iter
    if beta_loss is not None:
        args['beta_loss'] = beta_loss
    if init is not None:
        args['init'] = init
    if design is None:
        design = (3, 3)

    # We need result at alpha 0.0 for later decision on alpha value, so force
    # selected alphas to include 0. Also do some sanity checking such as
    # removing negative values.
    if alphas is None:
        alphas = [2 ** i for i in range(-5, 2)]
        scale_samples = scale_samples if scale_samples is not None else True
    else:
        scale_samples = scale_samples if scale_samples is not None else False
    scale_factor: float = float(x.shape[1]) if scale_samples else 1.0
    alpha_list: List[float] = [x / scale_factor for x in alphas if x >= 0.0]
    if 0.0 not in alpha_list:
        alpha_list = [0] + alpha_list

    # Set up random generator for this whole run
    rng: np.random.Generator = np.random.default_rng(seed)

    # Generate shuffles of data
    shuffles_gen: Generator[BicvSplit] = BicvSplit.from_matrix(
        x, random_state=rng, n=shuffles, design=design)

    # Make a generator of parameter objects to pass to bicv
    params: List[NMFParameters] = list(itertools.chain.from_iterable(
        [[NMFParameters(x=x, alpha=a, seed=rng, **args) for a in
          sorted(alpha_list)]
         for x in shuffles_gen]
    ))

    # Get results
    # No real use in multiprocessing, sklearn implementation generally makes
    # good use of multiple cores anyway (depending on underlying compilation
    # of BLAS etc). Multiprocessing just makes processes compete for resources
    # and is slower.
    total_runs: int = len(alpha_list) * shuffles
    res_map: Iterable = (
        map(bicv, tqdm(params, total=total_runs))
        if progress_bar else
        map(bicv, params)
    )
    results: List[BicvResult] = sorted(
        list(res_map),
        key=lambda x: x.parameters.alpha
    )
    # Collect results from the alpha values into lists, and place in a
    # dictionary with key = alpha
    grouped_results: Dict[float, List[BicvResult]] = {
        rank_i: list(list_i) for rank_i, list_i in
        itertools.groupby(results, key=lambda z: z.parameters.alpha)
    }
    # Validate that there are the same number of results for each rank.
    # Decomposition shouldn't silently fail, but best not to live in a
    # world of should. Currently deciding to warn user and still return
    # results.
    if len(set(len(y) for y in grouped_results.values())) != 1:
        logger.error(("Uneven number of results returned for each rank, "
                       "some rank selection iterations may have failed."))
    # Determine the selected regularisation parameter based on criteria from
    # the Enterosignatures paper
    best_a: float = suggest_alpha(grouped_results)

    return (best_a, grouped_results)


def suggest_alpha(regu_results: Dict[float, List[BicvResult]]) -> float:
    """Suggest a suitable value for alpha.

    Want to select the largest value of :math:`alpha` possible which does not
    detrimentally effect the quality of the decomposition. To gauge this,
    we adopt the heuristic of [REF], selecting the highest value of
    :math:`alpha` for which the mean :math:`R^2` is not lower than the (mean
    :math:`R^2` + standard deviation) at :math:`alpha=0`.

    This is called by default in :func:`regu_selection`. It is provided as
    public method as the Nextflow pipeline splits the Bicv process, and
    doesn't use :func:`regu_selection`, and so it can be called after.

    :param regu_results: Dictionary with keys being alpha values, and values
        a list of :class:`BicvResult` objects.
    """
    # Give a nicer error if 0.0 is not included
    if 0.0 not in regu_results:
        raise ValueError(
            "Results must include alpha=0.0 in order to suggest rank")
    # Calculate MEV at alpha=0.0 and SD of mean
    # Selected alpha whose MEV is greater than threshold
    mean_rsq: Dict[float, float] = {
        alpha: np.concatenate([x.r_squared for x in res]).mean()
        for alpha, res in regu_results.items()
    }
    sd_zero: float = np.std(
        np.concatenate(
            [x.r_squared for x in regu_results[0.0]]
        )
    )
    threshold: float = mean_rsq[0.0] - sd_zero
    # TODO: Make this more elegant.
    # This works for now though
    alpha_list: List[float] = list(regu_results.keys())
    best_a: float = sorted(alpha_list)[0]
    for a in sorted(alpha_list[1:]):
        if mean_rsq[a] < threshold:
            break
        best_a = a
    return best_a


def plot_regu_selection(
        regu_res: Union[Tuple[float, Dict], Dict],
        alpha_star: bool = True,
        **kwargs
) -> plotnine.ggplot:
    """Plot regularisation selection results.

    Takes a result from :func:`regu_selection` and passes to
    :func:`plot_rank_selection` to plot with alpha values along the
    x-axis. Consequently, pass any parameters for plotting as kwargs.

    :param regu_res: Results from :func:`regu_selection`.
    :param alpha_star: Suggest and plot a suitable alpha value using
        :func:`suggest_alpha`.
    """

    # regu_selection returns a tuple with best alpha value and dictionary of
    # results. Users might pass either the tuple or the dict in, so handle both.
    # Primarily, this is as I kept forgetting to only pass the dict and it
    # annoyed me.
    pass_res: Dict[float, List[BicvResult]] = {}
    suggested: Optional[float] = None
    if isinstance(regu_res, tuple):
        if not len(regu_res) == 2:
            raise ValueError("Unexpected regularisation result tuple format, "
                             "expected length 2 tuple with float, dict.")
        if not isinstance(regu_res[1], dict):
            raise ValueError("Unexpected regularisation result tuple format, "
                             "expected length 2 tuple with float, dict.")
        pass_res = regu_res[1]
        suggested = regu_res[0]
    elif isinstance(regu_res, dict):
        pass_res = regu_res
    else:
        raise ValueError("Unexpected regularisation result format, expected "
                         "either dict, or tuple of float, dict.")
    if alpha_star:
        stars_at: Dict[str, float] = dict(
            r_squared=suggest_alpha(regu_res) if suggested is None else
            suggested
        )

    return plot_rank_selection(
        **kwargs, results=pass_res, xaxis="alpha",
        stars_at=None if not alpha_star else stars_at,
        include=['r_squared']
    )


def bicv(params: Optional[NMFParameters] = None, **kwargs) -> BicvResult:
    """Perform a single run of bicrossvalidation.

    Perform one run of bicrossvalidation. Parameters can either be passed
    as a :class:`NMFParameters` tuple and are documented there, or by keyword
    arguments using the same names as :class:`NMFParameters`.

    :returns: Comparisons of the held out submatrix and estimate for each fold
    """

    if params is None:
        params = NMFParameters(**kwargs)

    fold: int = params.x.num_folds
    runs: Iterable[BicvResult] = map(__bicv_single,
                                     (params for _ in range(fold)),
                                     (params.x.fold(i) for i in range(fold))
                                     )

    # Combine results from each fold
    start: float = time.time()
    logger.info("Starting bi-cross validation; rank %s; alpha %s",
                params.rank, params.alpha)
    joined: BicvResult = BicvResult.join_folds(list(runs))
    logger.info("Completed bi-cross validation; took %s seconds",
                time.time() - start)
    return joined


def __bicv_single(params: NMFParameters, fold: BicvFold) -> BicvResult:
    """Run bicrossvalidation on a single fold. Return a results tuple with
    single entries in all the result fields, these are later joined. This
    implementation is based on scripts from
    https://gitlab.inria.fr/cfrioux/enterosignature-paper/. Quality
    measurements are only done for A and A', as intermediate steps are not
    of interest for rank or regularisation selection."""

    # Get three seeds for the decompositions
    rng: np.random.Generator = np.random.default_rng(params.seed)
    decomp_seeds: np.ndarray = rng.integers(0, 4294967295, 3)

    logger.debug("Starting fold %s; alpha=%s; rank=%s",
                id(fold), params.alpha, params.rank)

    logger.debug("Start Bicv step 1")
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
    logger.debug("Completed Bicv step 1")

    # step 2, get W_a using M_b
    # Use the function rather than object interface, as has some convenience
    # for using only one fixed initialisation
    logger.debug("Starting Bicv step 2")
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
    logger.debug("Completed Bicv step 2")

    # Step 3, get H_a using M_c
    logger.debug("Starting Bicv step 3")
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
    logger.debug("Completed Bicv step 3")

    # Step 4, calculate error for M_a
    logger.debug("Calculating quality measures for fold")
    Ma_calc = np.dot(W_a, H_a.T)

    return BicvResult(
        parameters=params,
        i=params.x.i,
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
    """Residual sum of squares.

    The square of difference between values in :math:`A` and :math:`A'`.

    :param A: Held-out matrix :math:`A`
    :param A_prime: Imputed matrix :math:`A`
    :returns: RSS
    """
    return ((A - A_prime) ** 2).sum().sum()


def _rsquared(A: np.ndarray, A_prime: np.ndarray) -> float:
    """Explained variance

    Consider the matrix as a flattened 1d array, and calculate the :math:`R^2`.
    This calculation varies from original Enterosignatures paper in that we
    subtract the mean. In the event :math:`R^2` would be inf/-inf (when total
    sum of squares is 0), this instead returns nan to make taking mean/median
    simpler later on.

    :param A: Held-out matrix :math:`A`
    :param A_prime: Imputed matrix :math:`A`
    :returns: :math:`R^2`
    """
    A_mean: float = np.mean(np.ravel(A))
    tss: float = ((A - A_mean) ** 2).sum().sum()
    rsq: float = 1 - (_rss(A, A_prime) / tss)
    return np.nan if abs(rsq) == np.inf else rsq


def _cosine_similarity(A: np.ndarray, A_prime: np.ndarray) -> float:
    """Cosine similarity between two flattened matrices

    Cosine angle between two matrices which are flattened to a 1d vector.

    :param A: Held-out matrix :math:`A`
    :param A_prime: Imputed matrix :math:`A`
    :returns: Cosine similarity
    """
    x_flat: np.array = np.ravel(A)
    wh_flat: np.array = np.ravel(A_prime)
    return wh_flat.dot(x_flat) / np.sqrt(
        x_flat.dot(x_flat) * wh_flat.dot(wh_flat)
    )


def _l2norm_calc(A: np.ndarray, A_prime: np.ndarray) -> float:
    """Calculates the L2 norm metric between two matrices

    :param A: Held-out matrix :math:`A`
    :param A_prime: Imputed matrix :math:`A`
    :returns: L2 norm
    """
    return np.sqrt(np.sum((np.array(A) - np.array(A_prime)) ** 2))


def _sparsity(x: np.ndarray) -> float:
    """Sparsity of a matrix

    :param x: Matrix
    :returns: Sparsity, 0 to 1
    """
    return 1 - (np.count_nonzero(x) / x.size)


def _stability_remove_rank_one(
        decompositions: Dict[int, List[Decomposition]]
) -> Dict[int, List[Decomposition]]:
    """Remove rank 1 from a dictionary of decompositions, as stability
    measures cannot be meaningfully generated for rank 1.
    """

    if 1 not in decompositions:
        return decompositions
    logger.warning("Rank 1 removed for stability based rank selection "
                   "(cophenetic correlation, dispersion) as coefficients "
                   "not meaningful for one signature.")
    return {
        i: l for i, l in decompositions.items() if i != 1
    }

def signature_similarity(
        decompositions: Dict[int, List[Decomposition]]
) -> pd.Series:
    """Mean cosine similarity of signatures for rank selection

    This rank selection criteria is based on the intuition that if a solution
    is good, it should be across similar multiple random initialisation of the
    data, similar to the motivation for :func:`cophenetic_correlation` and
    :func:`dispersion`.

    We pair signatures based on a cosine similarity (see
    :func:`cvanmf.stability.match_signatures`) and take the mean value between
    paired signatures at a rank, and look for clear peaks.

    Similarity is calculated between the best decomposition and all otherwise,
    not all possible pairs.

    The paired cosine similarity can also be visualised in more detail using
    :func:`cvanmf.stability.plot_signature_stability`.

    :param decompositions: Decompositions for several ranks as output by
        :func:`decompositions`.
    """

    from cvanmf.stability import signature_stability
    trimmed: Dict[int, List[Decomposition]] = _stability_remove_rank_one(
        decompositions)
    if len(trimmed) == 0:
        return pd.Series(name='k')
    s: pd.Series = signature_stability(
        trimmed
    ).groupby(['k'])['pair_cosine'].mean()
    s.name = 'signature_similarity'
    return s

def cophenetic_correlation(
        decompositions: Dict[int, List[Decomposition]],
        on: Literal['h', 'w'] = 'h'
) -> pd.Series:
    """Cophenetic correlation coefficient for rank selection

    The cophenetic correlation coefficient (ccc) is a commonly used way to
    select a suitable rank for decompositions (Brunet 2004). It is based on
    assigning each sample or feature to a single signature, and looking for
    stability in which are assigned to the same signature across multiple random
    initialisations.

    Our primary method for rank selection is bicrossvalidation, but we offer
    the ability to calculate ccc when you have performed multiple
    decompositions for a rank using :func:`decompositions`.

    :param decompositions: Results from the :func:`decompositions` function.
        A dictionary with the key being a rank, the value a list of
        decompositions for that rank.
    :param on: Look for stability in the assignment in the H matrix (samples)
        or W matrix (features).
    :returns: Series indexed by rank and with value being the ccc.
    """

    dc_use: Dict[int, List[Decomposition]] = (
        _stability_remove_rank_one(decompositions)
    )
    return pd.Series(
        index=dc_use.keys(),
        data=(
            _cophenetic_correlation(
                _cbar(x.consensus_matrix(on=on) for x in k))
            for k in dc_use.values()
        ),
        name="cophenetic_correlation"
    )


def dispersion(
        decompositions: Dict[int, List[Decomposition]],
        on: Literal['h', 'w'] = 'h'
) -> pd.Series:
    """Dispersion coefficient for rank selection

    The dispersion coefficient is a method for rank selection which
    looks for consistency in the average consensus matrix (Park 2007).
    This shares the same underlying data structure as
    :func:`cophenetic_correlation`, the average consensus matrix, looking at
    how often elements are assigned to the same signature, with elements
    assigned to the signature with maximum weight. The value for dispersion
    ranges between 0 and 1, with 1 indicating perfect stability, and 0 a highly
    scattered consensus matrix.

    Our primary method for rank selection is bicrossvalidation, but we offer
    the ability to calculate dispersion when you have performed multiple
    decompositions for a rank using :func:`decompositions`.

    :param decompositions: Results from the :func:`decompositions` function.
        A dictionary with the key being a rank, the value a list of
        decompositions for that rank.
    :param on: Look for stability in the assignment in the H matrix (samples)
        or W matrix (features).
    :returns: Series indexed by rank and with value being the dispersion
        coefficient.
    """

    dc_use: Dict[int, List[Decomposition]] = (
        _stability_remove_rank_one(decompositions)
    )
    return pd.Series(
        index=dc_use.keys(),
        data=(
            _dispersion(
                _cbar(x.consensus_matrix(on=on) for x in k))
            for k in dc_use.values()
        ),
        name="dispersion"
    )


def plot_stability_rank_selection(
        decompositions: Optional[Dict[int, List[Decomposition]]] = None,
        series: Optional[List[pd.Series]] = None,
        include: List[str] = ['cophenetic_correlation', 'dispersion',
                              'signature_similarity'],
        suggested_rank: bool = True,
        on: Literal['h', 'w'] = 'h'
) -> plotnine.ggplot:
    """Plot results for stability based rank selection methods (
    :func:`signature_stability`, :func:`cophenetic_correlation`,
    :func:`dispersion`).

    Automated rank selection uses :func:`suggest_rank_stability`.

    :param decompositions: Results from :func:`decompositions`. Not used if
        series is passed.
    :param series: Series to plot, resulting from :func:`signature_similarity`,
        :func:`cophenetic_correlation`, or :func:`dispersion`.
    :param include: Which method to include in the plot,
        a list containing values from ``{'cophenetic_correlation', 'dispersion',
        'signature_similarity'}``.
    :param suggested_rank: Make an estimate of estimate suggested rank using
        :func:`suggest_rank_stability`.
    :param on: Calculate stability of H (samples) or W (features). Not used if
        passed series.
    """
    if series is None and decompositions is None:
        raise ValueError('Must provide one of decompositions or series.')
    if series is None:
        series = []
        if 'cophenetic_correlation' in include:
            series.append(cophenetic_correlation(decompositions, on=on))
        if "dispersion" in include:
            series.append(dispersion(decompositions, on=on))
        if "signature_similarity" in include:
            series.append(signature_similarity(decompositions))
    series = [x for x in series if x.name in include]
    if len(series) == 0:
        raise ValueError(f"No valid measures in {include}")
    df: pd.DataFrame = (
        pd.concat(series, axis=1)
        .stack()
        .to_frame('value')
        .reset_index(names=['rank', 'measure'])
    )
    plt: plotnine.ggplot = (
        plotnine.ggplot(
            df,
            mapping=plotnine.aes(x='factor(rank)', y='value')
        ) +
        plotnine.geom_line(mapping=plotnine.aes(group="measure")) +
        plotnine.geom_point(mapping=plotnine.aes(color="measure")) +
        plotnine.facet_wrap(facets="measure", scales="free_y") +
        plotnine.ylab("Value") +
        plotnine.xlab("Rank") +
        plotnine.ggtitle("Stability based rank selection")
    )

    # Add auto rank suggestion
    if suggested_rank == True:
        suggested: Dict[str, int] = suggest_rank_stability(
            pd.concat(series, axis=1).reset_index(names='rank'),
            measures=include
        )
        suggested_df: pd.DataFrame = (
            pd.Series(suggested)
            .to_frame('rank')
            .reset_index(names='measure')
        )
        suggested_df = suggested_df.loc[suggested_df['measure'].isin(include)]
        # Determine the x-position of the selected rank, as the line gets
        # plotted at position n along the axis
        ranks_ordered: List[int] = sorted(df['rank'].unique())
        suggested_df['rank_pos'] = suggested_df['rank'].apply(
            lambda x: (np.nan if x is None or np.isnan(x) else
                       ranks_ordered.index(x) + 1))

        plt = (
            plt
            + plotnine.geom_vline(
                data=suggested_df,
                mapping=plotnine.aes(xintercept="rank_pos"),
                linetype="dashed",
                color="black",
                alpha=.75,
                size=1
            )
        )
    return plt


def _write_symlink(path: pathlib.Path,
                   target: Optional[pathlib.Path],
                   save_fn: Callable[[pathlib.Path], None],
                   symlink: bool) -> None:
    linked: bool = False
    if symlink and target is not None:
        if target.is_file():
            logger.debug("Attempting to symlink %s -> %s", path, target)
            try:
                path.symlink_to(target.resolve())
                linked: bool = True
            except Exception as e:
                logger.debug("Failed to create symlink %s -> %s", path, target)
    if not linked:
        logger.debug("Writing to %s", target)
        save_fn(path)


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
@click.option("-s",
              "--shuffles",
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
@click.option("--design",
              required=False,
              type=int,
              nargs=2,
              default=(3,3),
              show_default=True,
              help="Two numbers stating how to split the input matrix for "
                   "bicrossvalidation. The first is the number of even splits "
                   "to make along the rows; the second along the columns. "
                   "This will results in mn folds for each shuffle. Provide "
                   "in the format '--design 4 3'. More fold means more "
                   "iterations per shuffle, so increases execution time.")
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
        design: Tuple[int, int]
) -> None:
    """Rank selection for NMF using mn-fold bi-cross validation

    Attempt to identify a suitable rank k for decomposition of input matrix X.
    This is done by shuffling the matrix a number of times, and for each
    shuffle diving it into m x n submatrices (m splits on rows, n splits on
    columns). Each of these nine is held out and an estimate learnt from the
    remaining matrices, and the quality of the estimated matrix used to
    identify a suitable rank.

    The underlying NMF implementation is from scikit-learn, and there is more
    documentation available there for many of the NMF specific parameters there.
    """

    __configure_logger(verbosity)

    # Validate parameters
    # Rank min / max in right order
    rank_min, rank_max = min(rank_min, rank_max), max(rank_min, rank_max)
    ranks: List[int] = list(range(rank_min, rank_max, rank_step))
    if len(ranks) < 2:
        logger.fatal(("Must search 2 or more ranks; ranks provided were "
                       "%s"), str(ranks))
        return None

    # Read input data
    x: pd.DataFrame = pd.read_csv(input, delimiter=delimiter, index_col=0)
    # Check reasonable dimensions
    if x.shape[0] < 2 or x.shape[1] < 2:
        logger.fatal("Loaded matrix invalid shape: (%s)", x.shape)
        return
    # TODO: Check all columns numeric

    # Log params being used
    param_str: str = (
        f"\nData Locations\n"
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
        f"Design:           {design}"
    )
    logger.info("cvanmf rank selection")
    logger.info(param_str)

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
        progress_bar=progress,
        design=design
    )

    # Write output
    rank_tbl: pd.DataFrame = BicvResult.results_to_table(results)
    rank_plt: plotnine.ggplot = plot_rank_selection(results)
    out_path: pathlib.Path = pathlib.Path(output_dir)
    logger.info("Writing results to %s", str(out_path))
    rank_tbl.to_csv(str(out_path / "rank_selection.tsv"), sep=delimiter)
    rank_plt.save(out_path / "rank_selection.pdf")

    # Completion
    logger.info("Rank selection completed")


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
@click.option("-s",
              "--shuffles",
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
@click.option("--l1_ratio",
              required=False,
              type=float,
              default=1.0,
              show_default=True,
              help="""Regularisation mixing parameter. In range 0.0 <= l1_ratio 
              <= 1.0. This controls the mix between sparsifying and densifying
              regularisation. 1.0 will encourage sparsity, 0.0 density.""")
@click.option("--rank",
              "-k",
              required=True,
              type=int,
              help="""Number of signatures in the decomposition. 
              Regularisation is selected for a given rank, and the optimal 
              value may vary between ranks.""")
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
@click.option("--scale/--no-scale",
              required=False,
              type=bool,
              default=True,
              show_default=True,
              help="""Scale alpha parameter by number of samples. 
              Setting this to True provides the same behaviour as was applied 
              in earlier versions of scikit-learn. This is done by default,
              as the default alpha values are selected to work with this 
              regularisation calculation. The alpha values reported in the 
              output will be the scaled alpha values."""
              )
@click.option("--design",
              required=False,
              type=int,
              nargs=2,
              default=(3, 3),
              show_default=True,
              help="Two numbers stating how to split the input matrix for "
                   "bicrossvalidation. The first is the number of even splits "
                   "to make along the rows; the second along the columns. "
                   "This will results in mn folds for each shuffle. Provide "
                   "in the format '--design 4 3'. More fold means more "
                   "iterations per shuffle, so increases execution time.")
@click.argument("alpha",
                nargs=-1,
                type=float)
def cli_regu_selection(
        input: str,
        output_dir: str,
        delimiter: str,
        shuffles: int,
        progress: bool,
        verbosity: str,
        seed: int,
        rank: int,
        alpha: List[float],
        l1_ratio: float,
        max_iter: int,
        beta_loss: str,
        init: str,
        scale: bool,
        design: Tuple[int, int]
) -> None:
    """Regularisation selection for NMF on ALPHA 9 fold bi-cross validation

    Attempt to identify a suitable regularisation parameter alpha for
    decomposition of input matrix X at a given rank with a given ratio
    between L1 and L2 regularisation.
    This is done by shuffling the matrix a number of times, and for each
    shuffle diving it into 9 submatrices. Each of these nine is held out and
    an estimate learnt from the remaining matrices, and the quality of the
    estimated matrix used to identify a suitable alpha.

    The underlying NMF implementation is from scikit-learn, and there is more
    documentation available there for many of the NMF specific parameters there.

    ALPHA is a list of values to be tested. 0.0 will always be added.
    """

    __configure_logger(verbosity)

    # Validate parameters
    alphas: List[float] = __alpha_values(alpha)
    if len(alphas) < 2:
        logger.fatal(("Must search 2 or more alphas; alphas provided were "
                       "%s"), str(alphas))
        return None

    # Read input data
    x: pd.DataFrame = pd.read_csv(input, delimiter=delimiter, index_col=0)
    # Check reasonable dimensions
    if x.shape[0] < 2 or x.shape[1] < 2:
        logger.fatal("Loaded matrix invalid shape: (%s)", x.shape)
        return
    # TODO: Check all columns numeric

    logger.info("cvanmf regularisation selection")
    # Log params being used
    param_str: str = (
        f"\nData Locations\n"
        f"------------------------------\n"
        f"Input:            {input}\n"
        f"Output:           {output_dir}\n"
        f""
        f"Bi-cross validation parameters\n"
        f"------------------------------\n"
        f"Seed:             {seed}\n"
        f"Rank:             {rank}\n"
        f"Shuffles:         {shuffles}\n"
        f"L1 Ratio:         {l1_ratio}\n"
        f"Alphas:           {alphas}\n"
        f"Max Iterations:   {max_iter}\n"
        f"Beta Loss:        {beta_loss}\n"
        f"Initialisation:   {init}\n"
        f"Scale Regu:       {scale}\n"
        f"Design:           {design}\n"
    )
    logger.info(param_str)

    # Perform rank selection
    best_alpha: float
    results: Dict[float, List[BicvResult]]
    best_alpha, results = regu_selection(
        x=x,
        rank=rank,
        shuffles=shuffles,
        l1_ratio=l1_ratio,
        alphas=alphas,
        max_iter=max_iter,
        beta_loss=beta_loss,
        init=init,
        progress_bar=progress,
        scale_samples=scale,
        design=design
    )

    # Write output
    regu_tbl: pd.DataFrame = BicvResult.results_to_table(results)
    regu_plt: plotnine.ggplot = plot_regu_selection(results)
    out_path: pathlib.Path = pathlib.Path(output_dir)
    logger.info("Writing results to %s", str(out_path))
    regu_tbl.to_csv(str(out_path / "regu_selection.tsv"), sep=delimiter)
    regu_plt.save(out_path / "regu_selection.pdf")

    # Completion
    logger.info("Regularisation selection completed")

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
    :param top_n: Number of decompositions to be returned for each rank.
    :param top_criteria: Criteria to use when determining which are the top
        decompositions. Can be one of beta_divergence, rss, r_squared,
        cosine_similairty, or l2_norm.
    :param seed: Seed or random generator used
    :param alpha: Regularisation parameter applied to both H and W matrices.
    :param l1_ratio: Regularisation mixing parameter. In range 0.0 <= l1_ratio
          <= 1.0. This controls the mix between sparsifying and densifying
          regularisation. 1.0 will encourage sparsity, 0.0 density
    :param max_iter: Maximum number of iterations during decomposition. Will
        terminate earlier if solution converges
    :param beta_loss: Beta loss function for NMF decomposition
    :param init: Initialisation method for H and W matrices on first step.
        Defaults to random
    :param progress_bar: Display progress bar
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
        logger.info("Using deterministic initialisation; only a single"
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
        map(decompose, tqdm(params)) if progress_bar else
        map(decompose, params))
    results: List[Decomposition] = sorted(
        list(res_map),
        key=lambda y: y.parameters.rank
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

    * :math:`X` is the input matrix, with m features on rows, and n samples on
      columns.
    * :math:`H` is the transformed data, with k signatures on rows, and n
      samples on columns.
    * :math:`W` is the feature weight matrix, with m features on rows, and m
      features on columns.

    The scikit-learn implementation has these transposed; this package
    handles transposing back and forth internally, and expects input in the
    features x samples orientation, and provides :math:`W` and :math:`H` inline
    with the literature rather than scikit-learn.

    Decomposition objects can be sliced using the syntax::

        sliced_model = model[samples, features, signatures]
        # Only slice on one dimension
        sliced_signatures = model[:, :, ['S1', 'S2']]

    Slices must be ordered collections of strings, integer indices, or booleans.
    """

    TOP_CRITERIA: Dict[str, bool] = dict(
        cosine_similarity=True,
        r_squared=True,
        rss=False,
        l2_norm=False,
        beta_divergence=False
    )
    """Defines which criteria are available to select the best decomposition
    based on, and whether to take high values (True) or low values (False).
    
    :meta private:
    """

    DEF_SCALES: List[List[str]] = [
        # Used by preference, Bang Wong's 7 distinct colours for colour
        # blindness, https://www.nature.com/articles/nmeth.1618 via
        # https://davidmathlogic.com/colorblind
        ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2',
         '#D55E00', '#CC79A7', '#000000'],
        # Sasha Trubetskoy's 20 distinct colours
        # https://sashamaps.net/docs/resources/20-colors/
        ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
         '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff',
         '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
         '#000075', '#a9a9a9']
    ]
    """Default, color-blindness friendly colorscales for signatures in plots.
    Used by preference is Bang Wong's 7 distinct colours for colour blindness
    (https://www.nature.com/articles/nmeth.1618 via 
    https://davidmathlogic.com/colorblind). For more signatures, Sasha
    Trubetskoy's 20 distinct colours are used 
    (https://sashamaps.net/docs/resources/20-colors/)
    """

    DEF_PLOT_SAMPLE_LIMIT: int = 250
    """When there are more than this many samples, the default save method
    will not produced plots which have points for each sample."""

    DEF_PLOT_SAMPLE_LIMIT_REMOVE: Set[str] = {
        "plot_relative_weight", "plot_pcoa", "plot_modelfit_point"
    }
    """The plots to ignore when there are more than DEF_PLOT_SAMPLE_LIMIT 
    samples."""


    LOAD_FILES: List[str] = ['x.tsv', 'h.tsv', 'w.tsv', 'parameters.yaml',
                             'properties.yaml']
    """Defines the files while are loaded to recreate a decomposition object
    from disk."""

    def __init__(self,
                 parameters: NMFParameters,
                 h: pd.DataFrame,
                 w: pd.DataFrame,
                 feature_mapping: Optional['reapply.FeatureMapping'] = None
                 ) -> None:
        self.__h: pd.DataFrame = self.__string_index(h)
        self.__w: pd.DataFrame = self.__string_index(w)
        if parameters.x is not None:
            parameters.x.columns = [str(x) for x in parameters.x.columns]
            parameters.x.index = [str(x) for x in parameters.x.index]
        self.__params: NMFParameters = parameters
        self.__input_hash: Optional[int] = None
        self.__colors: List[str] = self.__default_colors(n=parameters.rank)
        self.__feature_mapping: Optional['reapply.FeatureMapping'] \
            = feature_mapping

    @staticmethod
    def __string_index(data: Union[pd.DataFrame, pd.Index]
                       ) -> Union[pd.DataFrame, pd.Index]:
        """Convert indices of pandas object to strings if they are numeric.
        Numeric indices cause problems for plotting, and for detecting what
        kind of slicing is desired, so we instead convert all to strings."""
        data.index = [str(x) for x in data.index]
        if isinstance(data, pd.DataFrame):
            data.columns = [str(x) for x in data.columns]
        return data

    @staticmethod
    def __flex_slice(df: pd.DataFrame,
                     idx_i: Union[slice, List[Union[int, str]]],
                     idx_j: Union[slice, List[Union[int, str]]]
                     ) -> pd.DataFrame:
        """Subset a dataframe by index position or value depending on list
        contents."""
        idx_i = list(df.index[idx_i]) if isinstance(idx_i, slice) else idx_i
        idx_j = list(df.columns[idx_j]) if isinstance(idx_j, slice) else idx_j
        pos_i: bool = isinstance(idx_i[0], int)
        pos_j: bool = isinstance(idx_j[0], int)
        match (pos_i, pos_j):
            case (True, True):
                return df.iloc[idx_i, idx_j]
            case (False, False):
                return df.loc[idx_i, idx_j]
            case (True, False):
                return df.iloc[idx_i, :].loc[:, idx_j]
            case (False, True):
                return df.iloc[:, idx_j].loc[idx_i, :]
        raise IndexError()

    def __getitem__(self, item):
        """Allow slicing of a model.

        Decompositions can be sliced or indexed along three axes:
        * Samples: Alters X and H.
        * Features: Alters X and W.
        * Signatures: Alters W and H.

        To slice a given axis, either a slice object, an iterable of integer
        indices, or an iterable of string indices can be provided. For this
        description, they'll be callable Slicelike.

        For positional slicing, the order is samples, features, signatures.
        This is at odds with conventional matrix addressing (rows, columns)
        with the assumption that the most common slicing operation will be
        to select a subset of samples, and so this should be the easiest to do.

        Decomposition supports slicing in the following ways:
        * Slicelike - Slice samples.
        * Tuple(Slicelike, Slicelike) - Slice samples, features
        * Tuple(Slicelike, Slicelike, Slicelike) - Slice samples, features,
            signatures
        """
        # By default, use a slice object which takes all in the current order
        Slicelike = Union[slice, Iterable]
        slc_signatures: Slicelike = slice(self.parameters.rank)
        slc_features: Slicelike = slice(self.w.shape[0])
        # Default to assuming only one slice dimension passed, will be
        # replaced if a tuple
        slc_samples: Slicelike = item

        # Validate input
        if isinstance(item, tuple):
            if len(item) == 1:
                slc_samples = item[0]
            if len(item) == 2:
                slc_samples, slc_features = item
            if len(item) == 3:
                slc_samples, slc_features, slc_signatures = item
        # If not a slice type, convert iterable to list and check indices are
        # in list
        for slc, idx, name in (x for x in
                               [(slc_samples, self.h.columns, "sample"),
                                (slc_signatures, self.h.index, "signature"),
                                (slc_features, self.w.index, "feature")]
                               if not isinstance(x[0], slice)):
            # Check all items in index - assume we've received and iterable
            slc = list(slc)
            if not isinstance(slc[0], int):
                if any(x not in idx for x in slc):
                    raise IndexError(f"Some of {name} slice indices not found "
                                     f"in the Decomposition object.")
            else:
                if max(slc) > len(idx) - 1 or min(slc) < 0:
                    raise IndexError(f"Range for {name} indices out of bounds: "
                                     f"min {min(slc)}, max {max(slc)}, "
                                     f"index length {len(slc)}")

        # Make new object with data subset
        # Make h slice now in order to determine suitable rank, can't think
        # of a better way of telling resulting dimension
        new_h: pd.DataFrame = self.__flex_slice(self.h,
                                                slc_signatures,
                                                slc_samples)
        cpy = Decomposition(
            parameters=NMFParameters(
                **(self.parameters._asdict() |
                   dict(x=(None if self.parameters.x is None else
                           self.__flex_slice(self.parameters.x,
                                             slc_features,
                                             slc_samples)),
                        rank=new_h.shape[0]))
            ),
            h=new_h,
            w=self.__flex_slice(self.w, slc_features, slc_signatures)
        )
        # Some extra work required to slice colours
        cpy.colors = (self.colors[slc_signatures]
                      if isinstance(slc_signatures, slice) else
                      [self.colors[i] for i, x in enumerate(self.names)
                       if (i if isinstance(slc_signatures[0], int) else x)
                       in slc_signatures]
                      )
        return cpy

    @property
    def h(self) -> pd.DataFrame:
        """Signature weights in each sample.

        Matrix with samples on columns, and signatures on rows, with each entry
        being a signature weight. This is not scaled, see :meth:`scaled`.
        """
        return self.__h

    @property
    def w(self) -> pd.DataFrame:
        """Feature weights in each signature.

        Matrix with signatures on columns, and features on rows, with each entry
        being a signature weight. This is not scaled, see :meth:`scaled`
        """
        return self.__w

    @property
    def parameters(self) -> NMFParameters:
        """Parameters used during decomposition."""
        return self.__params

    @property
    def cosine_similarity(self) -> float:
        """Cosine angle between flattened :math:`X` and :math:`WH`.

        A measure of how well the model reconstructs the input data. Ranges
        between 1 and 0, with 1 being perfect correlation, and 0 meaning the
        model is perpendicular to the input (no correlation). The same
        measure is available for each sample using :attr:`model_fit`.
        """
        return float(_cosine_similarity(self.parameters.x.values,
                                        self.w.dot(self.h).values))

    @property
    def r_squared(self) -> float:
        """Coefficient of determination (:math:`R^2`) between flattened
        :math:`X` and :math:`WH`.

        A measure of how well the model reconstructs the input data."""
        return _rsquared(self.parameters.x.values,
                         self.w.dot(self.h).values)

    @property
    def rss(self) -> float:
        """Residual sum of squares between flattened :math:`X`
        and :math:`WH`."""
        return _rss(self.parameters.x.values,
                    self.w.dot(self.h).values)

    @property
    def l2_norm(self) -> float:
        """L2 norm between flattened :math:`X` and :math:`WH`."""
        return _l2norm_calc(self.parameters.x.values,
                            self.w.dot(self.h).values)

    @property
    def sparsity_w(self) -> float:
        """Sparsity of :attr:`w` matrix.

        This is the proportion of entries in the :math:`W` matrix which are 0.
        """
        return _sparsity(self.w.values)

    @property
    def sparsity_h(self) -> float:
        """Sparsity of :attr:`h` matrix.

        This is the proportion of entries in the :math:`H` matrix which are 0.
        """
        return _sparsity(self.h.values)

    @property
    def beta_divergence(self) -> float:
        """The beta divergence (using the method defined in the parameters
        object) between :math:`X` and :math:`WH`."""
        return _beta_divergence(self.parameters.x,
                                self.w.values,
                                self.h.values,
                                beta=self.parameters.beta_loss)

    @property
    def wh(self) -> pd.DataFrame:
        """Product of decomposed matrices :math:`W` and :math:`H` which
        approximates input."""
        return self.w.dot(self.h)

    @property
    def model_fit(self) -> pd.Series:
        """How well each sample :math:`i` is described by the model, expressed
        by the cosine angle between :math:`X_i` and :math:`(WH)_i`. Cosine
        angle ranges between 0 and 1 in this case, with 1 being good and 0
        poor (perpendicular),"""
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
        """Names for each of the signatures.

        New names for signatures can be given as a list. This will change
        the name in the :attr:`w` and :attr:`h` matrices::

            # Set new names for a model with 4 signatures
            mdoel.names = ['A', 'B', 'X', 'Y']

        """
        return list(self.h.index)

    @names.setter
    def names(self, names: Iterable[str]) -> None:
        """Set names for each of the signatures. Renames the H and W
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
        in the :math:`H` matrix. In the unusual case where all signatures have 0
        weight for a sample, this will return NaN, and is likely a sign of
        a poor model."""
        # Replace 0s with NA, as cannot have a max when picking between 0s
        return self.h.replace(0, np.nan).idxmax()

    @property
    def quality_series(self) -> pd.Series:
        """Quality measures (r_squared, cosine similarity etc) as series.

        Each decomposition has a range of values describing it's properties and
        approximation of the input data. This property is a series which
        includes all of these properties."""
        return pd.Series(dict(
            cosine_similarity=self.cosine_similarity,
            r_squared=self.r_squared,
            rss=self.rss,
            l2_norm=self.l2_norm,
            sparsity_h=self.sparsity_h,
            sparsity_w=self.sparsity_h,
            beta_divergence=self.beta_divergence
        ))

    @property
    def colors(self) -> List[str]:
        """Colors which represents each signature in plots.

        Colors default to a colorblind distinct palette.

        Colors can be changed by setting this property. A list can be provided,
        or a dictionary mapping signature name to new color::

            # For a model with three signatures S1, S2, S3
            # Change all colors with list
            model.colors = ['red', 'blue', '#ffffff']
            # Change two colors using dictionary
            model.colors = dict(S1='green', S3='#000000')

        """
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
                    self.__colors[sig_idx] = color
                except IndexError as e:
                    logger.info("Unable to set color for %s, signature"
                                 "not found", signature)
        elif colors is None:
            self.__colors = self.__default_colors(self.parameters.rank)
        else:
            color_list: List[str] = list(colors)
            if len(color_list) < self.parameters.rank:
                logger.info("Fewer colors than signature provided. Given %s, "
                             "expected %s", len(color_list),
                             self.parameters.rank)
                self.__colors[:len(color_list)] = color_list
            elif len(color_list) > self.parameters.rank:
                logger.info("More colors than signatures provided. Given %s, "
                             "expected %s", len(color_list),
                             self.parameters.rank)
                self.__colors = color_list[:self.parameters.rank]
            else:
                self.__colors = color_list

    @property
    def feature_mapping(self) -> 'reapply.FeatureMapping':
        """Mapping of new data features to those in the model being reapplied

        When fitting new data to an existing model, the naming of feature may
        vary or some features may not exist in the model. This property holds an
        object which maps from the new data features to the model features.
        For de-novo decompositions this will be None.
        """
        return self.__feature_mapping

    @property
    def color_scale(self) -> plotnine.scale_color_discrete:
        """Plotnine scale for color aesthetic using signature colors."""
        color_dict: Dict[str, str] = dict(zip(
            self.names, self.colors))
        color_dict['modelfit'] = 'white'
        color_scale = plotnine.scale_color_manual(
            values=list(color_dict.values()),
            limits=list(color_dict.keys())
        )
        return color_scale

    @property
    def fill_scale(self) -> plotnine.scale_fill_discrete:
        """Plotnine scale for fill aesthetic using signature colors."""
        color_dict: Dict[str, str] = dict(zip(
            self.names, self.colors))
        color_dict['modelfit'] = 'white'
        fill_scale = plotnine.scale_fill_manual(
            values=list(color_dict.values()),
            limits=list(color_dict.keys())
        )
        return fill_scale

    def representative_signatures(self, threshold: float = 0.9) -> pd.DataFrame:
        """Which signatures describe a sample.

        Identify which signatures contribute to describing a samples.
        Represenative signatures are those for which the cumulative sum is
        equal to or lower than the threshold value.

        This is done by considering each sample in the sample :meth:`scaled`
        :math:`H` matrix, and taking a cumulative sum of weights in descending
        order. Any signature for which the cumulative sum is less than or equal
        to the threshold is considered representative.

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
                             threshold: float = 0.9
                             ) -> pd.DataFrame:
        """Which samples have a monodominant signature.

        A monodominant signature is one which represents at least the
        threshold amount of the weight in the :meth:`scaled` :math:`H` matrix.

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

    def reapply(self,
                y: pd.DataFrame,
                input_validation: Optional[InputValidation] = None,
                feature_match: Optional[FeatureMatch] = None,
                **kwargs
                ) -> Decomposition:
        """Get signature weights for new data.

        When the features in ``y`` exactly match those used to learn this
        decomposition, you can set the ``input_validation`` and
        ``feature_match`` parameters as None.

        In some cases, the features in new data y may not exactly match
        those used in the original decomposition, for instance if you have new
        microbiome data there may be different taxa present, or a different
        naming format may be used in the new data. The function
        ``feature_match`` can be used to handle these cases, by defining a
        function to map names between new and existing data. The
        ``input_validation`` functions is largely used for existing models, to
        valdiate that data being provided is the expected format.

        :param y: New data of the same type used to generate this decomposition
        :param input_validation: Function to validate and transform ``y``
        :param feature_match: Function to match features in ``y`` and :attr:`w`
        :param kwargs: Arguments to pass to ``validate_input`` and
            ``feature_match``
        :return: :class:`Decomposition` with signature weights for samples in
            ``y``.
        """
        # Wrapper around the _reapply_model function
        from cvanmf.reapply import match_identical
        if input_validation is None:
            input_validation = lambda x, **kwargs: x
        if feature_match is None:
            feature_match = match_identical

        return _reapply_model(
            y=y,
            w=self.w,
            colors=self.colors,
            input_validation=input_validation,
            feature_match=feature_match,
            **kwargs
        )

    def scaled(self,
               matrix: Union[pd.DataFrame, Literal['h', 'w']],
               by: Optional[str] = None
               ) -> pd.DataFrame:
        """Total sum scaled version of a matrix.

        Scale a matrix to a proportion of the feature/sample total, or
        to a proportion of the signature total.

        :param matrix: Matrix to be scaled, one of :attr:`h` or :attr:`w`, or a
            string from ``{'h', 'w'}.``
        :param by: Scale to proportion of ``sample``, ``feature``, or
            ``signature`` total. This defaults to

            * :math:`H`: ``sample``
            * :math:`W`: ``signature``

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

        if by is None:
            by = "sample" if is_h else "signature"
            logger.info("Using default scaling for matrix (by %s)",
                         by)
        # Warn if attempting to normalise H by feature, or W by sample
        if by.lower() not in {'feature', 'sample', 'signature'}:
            raise ValueError(
                f"by must be one of 'feature', 'sample', or 'signature'; "
                f"given {by}"
            )
        if is_h and by == "feature":
            logger.warning("H matrix is sample matrix (signatures x samples), "
                            "cannot scale by feature. Scaling by sample "
                            "instead")
            by = "sample"
        if not is_h and by == "sample":
            logger.warning("W matrix is feature matrix (features x "
                            "signatures), cannot scale by sample. Scaling by "
                            "signature instead")
            by = "signature"

        # Perform scaling
        by_sig: bool = by == "signature"
        transpose: bool = (is_h and by_sig) or (not is_h and not by_sig)
        # Transpose so axis to summed is on columns
        matrix = matrix.T if transpose else matrix
        scaled: pd.DataFrame = matrix / matrix.sum()
        return scaled.T if transpose else scaled

    def consensus_matrix(
            self,
            on: Union[Literal['h', 'w'], pd.DataFrame] = 'h'
    ) -> sparse.csr_array:
        r"""Consensus matrix of either :math:`H` or :math:`W`.

        Most typically, the consensus matrix is calculated on the :math:`H`
        matrix, and is a binary matrix representing whether sample :math:`i` is
        assigned to the same signature as sample :math:`j`. Samples are
        assigned to signatures based on their maximum weight. When calculated
        on :math:`W`, it is the same but for features assigned.

        The primary use of this is in generating a :math:`\bar{C}` matrix, the
        mean number of times two elements are assigned to the same signature.
        :math:`\bar{C}` is used to calculate the :meth:`cophenetic_correlation`
        and :meth:`dispersion` coefficients, a method of determining suitable
        rank.

        This is returned as a lower triangular matrix in sparse format.
        """

        mat: pd.DataFrame = (
            getattr(self, on) if isinstance(on, str)
            else
            on
        )
        # As dealing with W or H matrix, transpose so signatures are on the
        # first axis
        short_axis: int = int(mat.shape[0] > mat.shape[1])
        is_short_sigs: bool = all(x in mat.axes[short_axis] for x in self.names)
        if not is_short_sigs:
            # Possible we have as many signatures as elements
            # Check if the long axis is signatures
            is_long_sigs: bool = (
                all(x in mat.axes[abs(1-short_axis)] for x in
                    self.names)
            )
            if not is_long_sigs:
                raise ValueError("No axis in given matrix contains all "
                                 "signatures")
            short_axis = abs(1 - short_axis)
        mat = mat if short_axis == 0 else mat.T

        # Assign each column to it's max value
        bmat_csr: sparse.csr_array = sparse.csr_array(
            mat.apply(lambda x: x == x.max())
        )

        return sparse.tril(bmat_csr.T.dot(bmat_csr), format="csr")

    def discrete_signature_scale(
            self,
            axis: Literal['x', 'y'],
    ) -> Union[plotnine.scale_x_discrete, plotnine.scale_y_discrete]:
        """Make a plotnine scale which puts the signatures in order.

        By default, plotnine will alphabetically sort (S1, S11 .. S2, S21),
        this produces a scale object which can be added to a plot to put the
        signatures in their order in this object.
        """
        scale = (plotnine.scale_x_discrete if
                 axis == "x" else
                 plotnine.scale_y_discrete)
        return scale(limits=list(self.names))

    def compare_signatures(self, b: 'Comparable') -> pd.DataFrame:
        """Similarity between these signatures and one other set.

        Similarity here is defined as cosine as the angle between each
        pair of signature vectors, so 1 is identical (ignoring scale) and
        0 is perpendicular.

        This is a convenience method which calls
        :func:`stability.compare_signatures`.

        :param b: Signature matrix, or object with signature matrix
        :returns: Matrix with cosine of angles between signature vectors.
        """

        from cvanmf.stability import compare_signatures
        return compare_signatures(self, b)

    def match_signatures(self, b: 'Comparable') -> pd.DataFrame:
        """Identify optimal matches between these signatures and one other set

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
        :func:`stability.match_signatures`.

        :param b: Signature matrix, or object with signature matrix
        :returns: DataFrame with pairing and scores
        """

        from cvanmf.stability import match_signatures
        return match_signatures(self, b)

    def name_signatures_by_weight(
            self,
            cumulative_sum: float = 0.4,
            max_char_length: int = 10,
            max_num_features: int = 5,
            feature_delimiter: str = '+',
            number: bool = True,
            clean: Callable[[str], str] = lambda x: x.replace(' ', '_')
    ) -> None:
        """Give a slightly more descriptive name to each signature.

        Append features with highest relative weights to the end of
        signature names. This alters the object in place.

        :param cumulative_sum: Add features up to this cumulative sum (from
            max to min).
        :param max_char_length: Maximum length of new name (before joining
            with feature delimiter).
        :param max_num_features: Maximum number of features to use in name.
        :param feature_delimiter: When multiple features used, will join with
            this character
        :param number: Number the signatures. When true, starts each new name
            with S1, S2, etc.
        :param clean: Function to clean the string. Defaults to replacing
            spaces with underscores.
        """

        repr_features: pd.DataFrame = self.scaled('w').apply(
            Decomposition.__representative_signatures, threshold=cumulative_sum
        )
        rep_weight: pd.DataFrame = self.scaled('w') * repr_features

        def build_signature(sig_series: pd.Series) -> str:
            feat_list: List[str] = list(
                sig_series
                .sort_values(ascending=False)
                .iloc[:min(max_num_features, len(sig_series) - 1)]
                .index
            )
            feat_list = feat_list[:min(max_num_features, len(feat_list))]
            use_list: List[str] = [feat_list[0]]
            if len(feat_list) > 1:
                for new in feat_list[1:]:
                    if (sum(map(len, use_list)) + len(new)) > max_char_length:
                        break
                    else:
                        use_list.append(new)
            sig_str: str = clean(feature_delimiter.join(use_list))
            return sig_str

        signature_series = rep_weight.apply(build_signature)
        signature_series = [x[:min(len(x), max_char_length)] for x
                            in signature_series]
        if number:
            signature_series = [f'S{i + 1}_{name}' for i, name
                                in enumerate(signature_series)]
        self.names = list(signature_series)

    def plot_modelfit(self,
                      group: Optional[pd.Series] = None,
                      ) -> plotnine.ggplot:
        """Plot model fit distribution.

        This provides a histogram of the model fit of samples by default. If
        a grouping is provided, this will instead produce boxplots with each
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
        xlab: str = group.name if group.name is not None else "group"
        return (
                plotnine.ggplot(df, mapping=plotnine.aes(
                    x="group", y="model_fit")) +
                plotnine.geom_boxplot() +
                plotnine.ylab("Cosine Similarity") +
                plotnine.theme(
                    axis_text_x=plotnine.element_text(angle=90)
                ) +
                plotnine.xlab(xlab)
        )

    def plot_modelfit_point(self,
                            threshold: Optional[float] = 0.4,
                            yrange: Optional[Tuple[float, float]] = (0, 1),
                            point_size: float = 1.0
                            ) -> plotnine.ggplot:
        """Model fit for each sample as a point on a vertical scale.

        It may be of interest to look at the model fit of individual samples,
        so this plot shows the model fit of each sample as a point on a
        vertical scale. A threshold can be set below which the point will be
        coloured red to indicate low model fit, by default this is 0.4.

        :param threshold: Value below which to colour the model fit red. If
            omitted will not color any samples. The default of 0.4 is specific
            to the 5ES model (:func:`cvanmf.models.five_es`) and does not neccesarily
            represent a good threshold for other models.
        """
        mf_df: pd.DataFrame = (self.model_fit
                               .to_frame("model_fit")
                               .reset_index(names="sample"))
        mapping: plotnine.aes = plotnine.aes(x="sample", y="model_fit")
        if threshold is not None:
            mf_df['low_fit'] = mf_df['model_fit'] < (-1 if threshold is None
                                                     else threshold)
            mapping = plotnine.aes(x="sample", y="model_fit", color="low_fit")
        plt: plotnine.ggplot = (plotnine.ggplot(
            mf_df,
            mapping=mapping
        ) +
                                plotnine.geom_point(size=point_size) +
                                plotnine.theme_minimal() +
                                plotnine.theme(
                                    panel_grid_major_x=plotnine.element_blank(),
                                    panel_grid_minor_x=plotnine.element_blank(),
                                    panel_grid_minor_y=plotnine.element_blank(),
                                    axis_text_x=plotnine.element_text(angle=90)
                                ) +
                                plotnine.scale_y_continuous(
                                    limits=yrange
                                ) +
                                plotnine.ylab("Model Fit") +
                                plotnine.xlab("")
                                )
        if threshold is not None:
            plt = (plt +
                   plotnine.scale_color_manual(
                       values=["black", "red"],
                       name="Low Model Fit"
                   ) +
                   plotnine.geom_hline(
                       yintercept=threshold
                   ))
        return plt

    def plot_relative_weight(
            self,
            group: Optional[pd.Series] = None,
            group_colors: Optional[pd.Series] = None,
            model_fit: bool = True,
            heights: Union[Dict[str, float], Iterable[float]] = None,
            width: float = 6.0,
            sample_label_size: float = 5.0,
            legend_cols_sig: int = 3,
            legend_cols_grp: int = 3,
            legend_side: str = "bottom",
            **kwargs
    ):
        """Plot relative weight of each signature in each sample.

        To display the plot in a notebook environment, use ``result.render()``.
        Please note this plot uses the marsilea package rather than
        plotnine like other plots. Unfortunately, the options for combining
        multiple elements are not yet well developed in plotnine.

        Plots a stacked bar chart with a bar for each sample displaying the
        relative weight of each signature. Optionally the plot can also
        include sections at the top summarising the model fit for each
        sample, and a ribbon along displaying categorical metadata for samples.

        :param group: Categorical metadata for each sample to plot on ribbon
            at the bottom
        :param group_colors: Colour to associate with each of the metadata
            categories.
        :param model_fit: Include a top row indicating model fit per sample.
        :param heights: Height in inches for each component of the plot.
            Specify as a dictionary with keys 'dot', 'ribbon', 'bar', 'labels',
            or a list with heights for the elements included from top to bottom.
        :param width: Width of plot.
        :param sample_label_size: Size for sample labels. Set to 0 to remove
            sample labels.
        :param legend_cols_sig: Number of columns in Signature legend.
        :param legend_cols_grp: Number of columns in group legend.
        :param legend_side: Location of Signature and group legend. One of
            'top', 'right', 'left', 'bottom'
        :return: Marsilea whiteboard object. Call ``.render()`` to show plot.
        """
        # Parse height arguments
        if heights is not None:
            if isinstance(heights, dict):
                unex_keys: Set[str] = (
                    set(heights.keys())
                    .difference(DEF_RELATIVE_WEIGHT_HEIGHTS.keys())
                )
                if len(unex_keys) > 0:
                    logger.warning(
                        "Unexpected key(s) in heights: %s. Expecting: %s.",
                        heights.keys(),
                        DEF_RELATIVE_WEIGHT_HEIGHTS.keys()
                    )
            else:
                parts: List[str] = [x for x in [
                    'dot' if model_fit else None,
                    'ribbon' if group is not None else None,
                    'bar',
                    'label'
                ] if x is not None]
                vals: List[float] = list(heights)
                if len(parts) != len(vals):
                    logger.debug("Using heights %s", heights)
                    logger.warning(
                        "Passed %s heights when plot has %s parts (%s)",
                        len(vals),
                        len(parts),
                        parts
                    )
                m: int = min(map(len, (parts, vals)))
                vals, parts = vals[:m], parts[:m]
                heights = dict(zip(parts, vals))
                logger.debug("Using heights %s", heights)
        else:
            heights = {}
        heights = DEF_RELATIVE_WEIGHT_HEIGHTS | heights
        # Remove any parts not needed
        if not model_fit:
            _ = heights.pop('dot')
        if group is None:
            _ = heights.pop('ribbon')
        if sample_label_size == 0:
            _ = heights.pop('label')
        logger.debug("Heights %s", heights)

        # Get all required dataframes and match order
        rel_df: pd.DataFrame = self.scaled('h')
        mf: pd.Series = self.model_fit
        if group is not None:
            # If grouped, ensure no missing samples, and reorder weight to
            # order of grouping
            grp_missing: Set[str] = set(rel_df.columns) - set(group.index)
            grp_extra: Set[str] = set(group.index) - set(rel_df.columns)
            if len(grp_missing) > 0:
                logger.warning("%s samples missing from group, replaced with"
                                "NA.", len(grp_missing))
            if len(grp_extra) > 0:
                logger.warning("%s samples in group not in decompositions, "
                                "removed from group.", len(grp_extra))
            groupn = pd.concat(
                [group, pd.Series({x: "NA" for x in grp_missing})])
            groupn.name = group.name
            group = groupn
            group = group.drop(labels=list(grp_extra))
            rel_df = rel_df.loc[:, group.index]

        import marsilea as ma
        import marsilea.plotter as mp
        # Main stacked bar
        bar: mp.StackBar = mp.StackBar(
            rel_df,
            legend_kws=dict(title="Signature", ncols=legend_cols_sig),
            colors=dict(zip(self.names, self.colors)),
            width=1.0
        )
        wb: ma.WhiteBoard = ma.WhiteBoard(
            width=width, height=sum(heights.values()), margin=0.1
        )
        wb.add_layer(bar)

        # Add grouping
        if group is not None:
            logger.debug("Add Colors (group), height %s",
                          heights['ribbon'])
            ribbon: mp.Colors = mp.Colors(
                group,
                legend_kws=dict(ncols=legend_cols_grp,
                                title="" if group.name is None else group.name),
                **self.__group_colours(group_colors, group)
            )
            wb.add_top(ribbon, size=heights['ribbon'], pad=0.1)

        if model_fit:
            logger.debug("Add Point (model_fit), height %s",
                          heights['dot'])
            points: mp.Point = mp.Point(
                mf, label="Model Fit", linestyle='none', markersize=1.0,
                label_loc="right"
            )
            wb.add_top(points, size=heights['dot'], pad=0.1, name="model_fit")
            # wb.render()
            # point_ax = wb.get_ax("model_fit")
            # point_ax.set_ylim(0, 1.0)

        if sample_label_size > 0.0:
            logger.debug("Add Labels (labels), height %s", heights['label'])
            labels: mp.Labels = mp.Labels(
                rel_df.columns,
                fontsize=sample_label_size
            )
            wb.add_bottom(labels, size=heights['label'])

        wb.add_legends(side=legend_side)
        return wb


    def __group_colours(
            self,
            group_color: pd.Series,
            group: pd.Series
    ) -> Dict[str, Any]:
        """Make keywords arg to pass to Marsilea for colour ribbion plotting.

        If no manual colour specified, returns an empty dict so will get
        default Marsilea behaviour. If a Series linking value and colour is
        given, will remove any missing categories (to keep legend tidy) and
        add any missing ones (to avoid crashes)
        """

        if group_color is None:
            return dict()
        # Reduce to only categories which are in the decomposition
        group_dc: pd.Series = group.loc[group.index.isin(self.h.columns)]
        group_col: pd.Series = group_color.loc[group_color.index.isin(group_dc)]
        # Add any categories which are not included, and set to grey
        missing: List[str] = list(set(group.unique()) - set(
            group_col.index.unique()))
        group_col = pd.concat(
            [group_col, pd.Series(['grey'] * len(missing), missing)]
        )
        return dict(palette=group_col)

    def pcoa(self,
             on: Union[
                 pd.DataFrame, Literal["x", "h", "wh", "signatures"]] = "h",
             distance: str = 'braycurtis',
             wisconsin_standardise: bool = True,
             sqrt: bool = True
             ) -> OrdinationResults:
        """Principal Coordinates Analysis of decomposition.

        Performs PCoA on the specified matrix, and returns a scikit-bio
        OrdinationResults object. Can base distances on any matrix which has
        a column for each sample, or specify one of these via string. Defaults
        to distances based on :meth:`scaled` :math:`H` (signature weight in
        sample) matrix.

        Matrix is Wisconsin double standardised by default, as described in R
        function ``cmdscale``.

        Distance defaults to Bray-Curtis dissimilarity, and is square root
        transformed. Distance is calculated with scipy ``pdist`` function, and
        any method supported there can be specified in distance argument.

        :param on: Matrix to derive distances from
        :param distance: Distance method to use
        :param wisconsin_standardise: Apply Wisconsin double standardisation
        :param sqrt: Square root transform distances
        :return: PCoA results object from scikit-bio
        """

        from skbio import DistanceMatrix
        from skbio.stats.ordination import pcoa, pcoa_biplot

        mat: pd.DataFrame = self.__get_pcoa_matrix(on)
        # If the Decomposition has been sliced, some feature in X or WH
        # may now have all 0 values. Filter these out before standardisation
        # as creates NaNs.
        mat = mat.loc[mat.sum(axis=1) > 0]
        std_mat: pd.DataFrame = (
            _wisconsin_double_standardise(mat) if wisconsin_standardise else mat
        )
        dist: np.ndarray = squareform(pdist(std_mat.T, metric=distance))
        dist = np.sqrt(dist) if sqrt else dist
        dist_mat: DistanceMatrix = DistanceMatrix(
            dist,
            ids=std_mat.columns
        )

        pcoa_res: OrdinationResults = pcoa(dist_mat)
        pcoa_res = pcoa_biplot(pcoa_res, std_mat.T)

        return pcoa_res

    def __get_pcoa_matrix(
            self,
            mat: Union[pd.DataFrame, PcoaMatrices]
    ) -> pd.DataFrame:
        if isinstance(mat, str):
            match mat:
                case "h":
                    return self.scaled("h")
                case "wh":
                    return self.wh
                case "x":
                    return self.parameters.x
                case "signatures":
                    return self.scaled("h")
                case _:
                    raise ValueError(
                        "PCoA matrix must be one of h, x, wh, signatures.")
        elif isinstance(mat, pd.DataFrame):
            return mat
        else:
            raise ValueError("PCoA matrix must be a dataframe or string ("
                             "h, x, wh, signatures)")

    @staticmethod
    def __compare_str_series(
            str_series: Optional[Union[str, pd.Series]],
            comp_to: str
    ):
        """Compare an abject which could be a series or str to a str."""
        if isinstance(str_series, pd.Series):
            return False
        return str_series == comp_to

    @staticmethod
    def __set_guide_name(
            plt: plotnine.ggplot,
            guide: str,
            option: Optional[Union[str, pd.Series]],
            compare_to: str,
            lbl_if_match: str = None
    ) -> plotnine.ggplot:
        """Set a guide name based on whether value (or the name of value)
        matches compare_to."""
        lbl: str
        if Decomposition.__compare_str_series(option, compare_to):
            lbl = lbl_if_match
        elif isinstance(option, pd.Series):
            lbl = guide if option.name is None else option.name
        else:
            lbl = guide
        guide_dict: Dict[str, Any] = {
            guide: plotnine.guide_legend(title=lbl)
        }
        return (
                plt +
                plotnine.guides(**guide_dict)
        )

    def plot_pcoa(
            self,
            axes: Tuple[int, int] = (0, 1),
            color: Union[pd.Series, Literal['signature']] = "signature",
            shape: Optional[Union[pd.Series, Literal['signature']]] = None,
            signature_arrows: bool = False,
            point_aes: Dict[str, Any] = None,
            **kwargs
    ) -> plotnine.ggplot:
        """Ordination of samples.

        Perform PCoA of samples and plot first two axes. PCoA performed by the
        :meth:`pcoa` method, and arguments in kwargs are passed on to this
        method. Samples are coloured by primary ES.

        :param axes: Indices of PCoA axes to plot
        :param color: Metadata to use to color the points, or 'signature' to
            color based on the primary signature
        :param shape:  Metadata to used to decide shape of points,
            or 'signature' to base shape on the primary signature
        :param signature_arrows: Plot location of signatures as arrows
        :param point_aes: Dictionary of arguments to pass to geom_point
        :param kwargs: arguments to pass to :meth:`pcoa`
        :return: Scatter plot of samples
        """

        point_aes = point_aes if point_aes is not None else {}

        pcoa_res: OrdinationResults = self.pcoa(**kwargs)

        pos_df: pd.DataFrame = pcoa_res.samples
        pos_df['primary'] = self.primary_signature
        axes_str: Tuple[str, str] = tuple(f'PC{i + 1}' for i in axes)

        feat_df: pd.DataFrame = pcoa_res.features
        feat_df['signature'] = feat_df.index

        # Color and shape
        aes_dict: Dict[str, str] = dict(
            x=axes_str[0],
            y=axes_str[1],
            color="color"
        )
        aes_segment: Dict[str, str] = dict(
            xend=axes_str[0],
            yend=axes_str[1]
        )
        aes_label: Dict[str, str] = dict(
            x=axes_str[0],
            y=axes_str[1]
        )
        pos_df['color'] = (
            pos_df['primary'] if Decomposition.__compare_str_series(
                color, "signature") else
            color.loc[pos_df.index]
        )
        if shape is not None:
            pos_df['shape'] = (
                pos_df['primary'] if Decomposition.__compare_str_series(
                    shape, "signature") else shape.loc[pos_df.index]
            )
            aes_dict['shape'] = "shape"
        # If using primary signatures for colour, use object default colours
        color_scale: Optional[plotnine.scale_color_manual] = None
        if Decomposition.__compare_str_series(color, "signature"):
            color_scale = self.color_scale
        plt: plotnine.ggplot = (
                plotnine.ggplot(
                    pos_df,
                    plotnine.aes(**aes_dict)
                ) +
                plotnine.geom_point(**(DEF_PCOA_POINT_AES | point_aes)) +
                plotnine.xlab(
                    f'{axes_str[0]} ('
                    f'{pcoa_res.proportion_explained[axes[0]]:.2%})'
                ) +
                plotnine.ylab(
                    f'{axes_str[1]} ('
                    f'{pcoa_res.proportion_explained[axes[1]]:.2%})')
        )
        plt = Decomposition.__set_guide_name(
            plt, guide="color", option=color, compare_to="signature",
            lbl_if_match="Primary Signature"
        )
        if shape is not None:
            plt = Decomposition.__set_guide_name(
                plt, guide="shape", option=shape, compare_to="signature",
                lbl_if_match="Primary Signature"
            )
        if color_scale is not None:
            plt = plt + color_scale
        if signature_arrows:
            plt = (
                    plt +
                    plotnine.geom_segment(
                        feat_df,
                        plotnine.aes(**aes_segment),
                        inherit_aes=False,
                        x=0, y=0,
                        arrow=plotnine.arrow(length=0.05),
                        size=0.4,
                        linetype="solid",
                        show_legend=False,
                        color='black'
                    ) +
                    plotnine.geom_text(
                        feat_df,
                        plotnine.aes(**(aes_label | dict(label="signature"))),
                        inherit_aes=False,
                        color='black',
                        show_legend=False
                    )
            )
        return plt

    def plot_feature_weight(
            self,
            threshold: float = 0.04,
            label_fn: Callable[[str], str] = None
    ) -> plotnine.ggplot:
        """Plot features which contribute to each signature.

        Represent the relative contribution of features to signatures, showing
        any features which contribute over a threshold proportion of the weight.

        :param threshold: Show any features which contribute more than this
            proportion of the weight for this signature.
        :param label_fn: Function to map labels (use to make shortened labels
            for example)
        """

        if label_fn is None:
            label_fn = lambda x: x
        feat_weight: pd.DataFrame = (
            self.scaled('w')
            .stack()
            .reset_index()
            .set_axis(['feature', 'signature', 'rel_weight'], axis="columns")
        )
        feat_weight = feat_weight.loc[feat_weight['rel_weight'] >= threshold]
        feat_weight['label'] = feat_weight['feature'].map(label_fn)

        plt: plotnine.ggplot = (
                plotnine.ggplot(feat_weight,
                                mapping=plotnine.aes(
                                    x="signature", fill="signature", y="label",
                                    label="rel_weight"
                                )
                                ) +
                plotnine.geom_point(plotnine.aes(size="rel_weight")) +
                plotnine.geom_text(size=8, nudge_x=0.4, nudge_y=-0.05,
                                   format_string="{:.1%}") +
                plotnine.labs(x="Signature", y="Feature", fill="Signature",
                              size="Relative Weight") +
                plotnine.guides(fill=False, size=False) +
                self.fill_scale +
                self.discrete_signature_scale(axis='x')
        )
        return plt

    def __univariate_single_signature(
            self,
            signature: pd.Series,
            metadata: pd.Series,
            test: str,
            adj_method: str,
            alpha: float,
            **kwargs
    ) -> pd.Series:
        """Perform univariate tests on signature weights against metadata.

        Either Mann-Whitney or Kruskal-Wallis tests are performed. For K-W,
        post-hoc testing is performed with Dunn's test, using adjustment
        method specified.

        :param signature: Signature weights
        :param metadata: Metadata values (discrete)
        :param test: 'mw' for mann-whitney, anything else for kruskal-wallis
        :param adj_method: Multiple test adjustment method, any supported by
            statsmodels
        :param alpha: Threshold for significance
        :param kwargs: passed to test functions
        :return: Series with statistic, p, test, signature, md
        """
        # Split signature values to separate arrays
        sig_arrs: List[np.ndarray] = [
            signature.loc[metadata[metadata == x].index].rename(x) for x in
            metadata.unique()
        ]
        fields: List
        # Determine which category has the highest mean and median
        max_mean: str = functools.reduce(
            lambda x, y: x if x.mean() > y.mean() else y,
            sig_arrs
        ).name
        max_median: str = functools.reduce(
            lambda x, y: x if x.median() > y.median() else y,
            sig_arrs
        ).name
        if test == "mw":
            # Mann-Whitney U test
            if len(sig_arrs) != 2:
                raise ValueError("Mann-Whitney requires exactly 2 categories")

            from scipy.stats import mannwhitneyu
            res = mannwhitneyu(*sig_arrs, **kwargs)
            fields = [res.statistic, res.pvalue, 'mannwhitneyu', signature.name,
                      metadata.name, max_median, max_mean, None, None]
        else:
            # Default to KW test
            from scipy.stats import kruskal
            res = kruskal(*sig_arrs, **kwargs)

            # Post-hoc Dunn's test
            ph: pd.DataFrame = sp.posthoc_dunn(
                pd.concat([signature, metadata], axis=1),
                val_col=signature.name,
                group_col=metadata.name,
                p_adjust=adj_method,
            )
            # Set lower triangle to nan so no redundant pairs
            nand: np.ndarray = ph.values.copy()
            nand[np.tril_indices(nand.shape[0])] = np.nan
            ph_stack: pd.DataFrame = pd.DataFrame(
                nand,
                columns=ph.columns,
                index=ph.index
            ).stack().dropna()
            ph_sig: pd.DataFrame = ph_stack.loc[ph_stack <= alpha]
            # Format significant pairs into a string
            sig_pairs: str = ";".join(ph_sig.reset_index().apply(
                lambda x: f'{x.iloc[0]}|{x.iloc[1]}({x.iloc[2]})', axis=1
            ))

            fields = [res[0], res[1], 'kruskal', signature.name,
                      metadata.name, max_median, max_mean,
                      sig_pairs, ph.values]

        return pd.Series(data=fields,
                         index=['statistic', 'p', 'test', 'signature', 'md',
                                'max_median', 'max_mean', 'posthoc_str',
                                'posthoc_mat'])

    def __univariate_single_category(
            self,
            metadata: pd.Series,
            against: pd.DataFrame,
            drop_na: bool,
            adj_method: str,
            alpha: float
    ) -> pd.DataFrame:
        """Univariate tests for all signatures against a single set of metadata.

        :param metadata: Metadata, discrete
        :param against: Data to test against
        :param drop_na: Remove any NA values from metadata and signature
            before performing tests
        :param adj_method: Multiple test adjustment method, any supported by
            statsmodels
        :param alpha: Threshold
        :return: Dataframe with results for each signature
        """
        # Return from this should be table with signature, md_name, p, adj_p,
        # direction, test_stat, test_used
        md_clean: pd.Series = metadata.dropna() if drop_na else metadata
        categories: int = len(md_clean.unique())
        test: str = "kw" if categories > 2 else "mw"
        res = against.apply(self.__univariate_single_signature,
                            metadata=md_clean, test=test, axis=0,
                            adj_method=adj_method, alpha=alpha
                            ).T
        from statsmodels.stats.multitest import multipletests
        reject, adj_p, _, _ = multipletests(
            res['p'],
            alpha=alpha,
            method=adj_method,
            is_sorted=False,
        )
        res['alpha'] = alpha
        res['local_reject'] = reject
        res['local_adj_p'] = adj_p
        return res

    def univariate_tests(
            self,
            metadata: pd.DataFrame,
            against: Optional[Union[
                pd.DataFrame, Literal['signature', 'model_fit', 'both']
            ]] = None,
            drop_na: bool = True,
            adj_method: str = "fdr_bh",
            alpha: float = 0.05
    ) -> pd.DataFrame:
        """Test if signature relative weights vary between categories

        Test whether model weights are different between groups using
        non-parametric univariate tests. Currently uses the Mann-Whitney
        U-test on two sample cases, and Kruskall-Wallis tests on multiple
        category tests.

        For K-W tests, post-hoc tests are performed using Dunn's
        test, with the same adjustment and alpha values. Significant
        post-hoc tests are returned as a string in the results table, with the
        format A|B(0.001) for a significant result for pair A and B with
        adjusted p value of 0.001.

        :param metadata: Dataframe of metadata variables to test against. Can
            only handle discrete values.
        :param against: What to test the metadata against. This can be
            ``signature`` for relative :math:`H` weights, ``model_fit`` for per
            sample cosine similarity, or ``both`` (default). You can also
            provide any arbitrary matrix with the correct dimensions, for
            instance if you had done some custom processing of the :math:`H`
            matrix, or wanted to use absolute :math:`H` weights.
        :param drop_na: Remove any samples with ``NA`` values from metadata
            before testing. This is done on a per test basis, so one ``NA`` will
            not cause a sample to be removed for all tests.
        :param adj_method: Method to adjust for multiple tests. This is applied
            both locally (for each metadata category), and globally (considering
            all tests). Accepts any method supported by statsmodels
            ``multipletests``.
        :param alpha: Threshold value to reject :math:`H0`.
        :return: Dataframe with results for each signature and each metadata
            variable.
        """

        against = self.__get_against(against)
        res = pd.concat(
            [self.__univariate_single_category(
                metadata[x], against=against,
                drop_na=drop_na, adj_method=adj_method,
                alpha=alpha
            ) for x in metadata.columns]
        )
        from statsmodels.stats.multitest import multipletests
        reject, adj_p, _, _ = multipletests(
            res['p'],
            alpha=alpha,
            method=adj_method,
            is_sorted=False,
        )
        res['global_reject'] = reject
        res['global_adj_p'] = adj_p
        res['adj_method'] = adj_method
        return res

    def __extract_convert_metadata(
            self,
            md: pd.DataFrame,
            selector: Callable[[pd.Series], bool],
            measure_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract metadata of certain types, join to signatures, stack."""
        selected: pd.Series = md.apply(selector)
        md_subset: pd.DataFrame = md.loc[:, selected[selected].index]
        h: pd.DataFrame = measure_df
        md_subset = (
            md_subset
            .stack()
            .reset_index()
            .set_axis(['sample', 'metadata_field', 'metadata_value'], axis=1)
        ).merge(
            h.stack().reset_index().set_axis(
                ['sample', 'signature', 'signature_weight'], axis=1
            ),
            right_on="sample",
            left_on="sample"
        )
        # Order the signatures so they facet correctly
        md_subset['signature'] = pd.Categorical(
            md_subset['signature'],
            ordered=True,
            categories=h.columns
        )
        return md_subset

    def __get_against(
            self,
            against: Optional[Union[
                pd.DataFrame, Literal['signature', 'model_fit', 'both']
            ]]
    ) -> pd.DataFrame:
        """Data to plot metadata against.

        Defaults to both if not specified. 'signature' means relative
        signature weight, 'model_fit' the per sample model fit, and 'both'
        joins those two matrices.

        :param against: Either data to plot against, or a string indicating
        the desired data
        """
        against = against if against is not None else 'both'
        if isinstance(against, str):
            if against == 'signature' or against == 'signatures':
                against = self.scaled('h').T
            elif against == 'model_fit' or against == "modelfit":
                against = self.model_fit.to_frame('modelfit')
            elif against == 'both':
                against = pd.concat(
                    [self.scaled('h').T, self.model_fit.to_frame('modelfit')],
                    axis=1
                )
            else:
                raise ValueError("against must be one of model_fit or "
                                 "signatures, or a DataFrame.")
        return against

    def plot_metadata(
            self,
            metadata: pd.DataFrame,
            against: Optional[Union[pd.DataFrame, Literal['signature',
            'model_fit', 'both']]] = None,
            continuous_fn: Optional[Callable[[pd.Series], bool]] = None,
            discrete_fn: Optional[Callable[[pd.Series], bool]] = None,
            boxplot_params: Optional[Dict] = None,
            point_params: Optional[Dict] = None,
            disc_rotate_labels: Optional[float] = None,
            show_significance: bool = True,
            significance_formatter: Optional[Callable[[float, float, float],
            str]] = None,
            univariate_test_params: Dict[str, Any] = None
    ) -> Tuple[plotnine.ggplot, plotnine.ggplot]:
        """Plot relative signature weight against metadata.
        
        Produce plots of signature weight against metadata. Produces two plots,
        one with boxplots for categorical metadata, one with scatter plots for
        continuous metadata. Will infer which type each column is. To use an
        integer as categorical, convert it to Categorical type in pandas. Will
        conduct univariate tests as described in :meth:`univariate_tests` and
        indicate significance with symbols. This will be skipped if
        ``show_significance`` is False, which maybe be sensible for larger
        numbers of samples and metadata categories.

        :param metadata: Dataframe with samples on rows, and metadata on
            columns.
        :param against: DataFrame to plot the metadata against. Should
            contain an entry for each sample, with samples on rows. Defaults to
            :meth:`scaled` :math:`H` matrix (transpose of typical :math:`H`
            orientation).
        :param continuous_fn: Function to determine if a column is
            continuous. Defaults to considering any floating type or integer to
            be continuous. May want to customise if you want to use things such
            as date time formats.
        :param discrete_fn: Function to determine if a column is categorial.
            Defaults to considerings any string, or object type column with
            a number of unique values < 3/4 its length as categorical.
        :param boxplot_params: Dictionary of parameters to pass to
            ``geom_boxplot.`` These will be fixed parameters (so color="pink"
            to set all box outlines to pink).
        :param point_params: Dictionary of parameters to pass to geom_point.
            Will be fixed parameters, see above.
        :param disc_rotate_labels: Angle to rotate x axis labels by for
            boxplots.
        :param show_significance: Add significance to each subplot for discrete
            metadata.
        :param significance_formatter: Function which takes the p-value and
            adjusted p-values and returns a string to use as label. Defaults
            to :meth:`Decomposition.significance_format`.
        :param univariate_test_params: Parameters passed to
            :meth:`univariate_tests`
       :return: A tuple of plotnine ggplot objects, first is boxplots,
            second is scatter plots.
        """
        continuous_fn = (_is_series_continuous if continuous_fn is None
                         else continuous_fn)
        discrete_fn = (_is_series_discrete if discrete_fn is None else
                       discrete_fn)
        disc_rotate_labels = (
            90.0 if disc_rotate_labels is None else disc_rotate_labels)
        point_params = {} if point_params is None else point_params
        boxplot_params = {} if boxplot_params is None else boxplot_params
        against = self.__get_against(against)

        # Convert
        md: pd.DataFrame = (
            metadata.to_frame() if isinstance(metadata, pd.Series) else
            metadata
        )
        cont: pd.DataFrame = self.__extract_convert_metadata(
            md,
            selector=continuous_fn,
            measure_df=against
        )
        disc: pd.DataFrame = self.__extract_convert_metadata(
            md,
            selector=discrete_fn,
            measure_df=against
        )

        disc_plot: plotnine.ggplot = (
                plotnine.ggplot(
                    disc,
                    mapping=plotnine.aes(
                        x='metadata_value',
                        y='signature_weight',
                        fill="signature"
                    )
                ) +
                plotnine.geom_boxplot(**boxplot_params) +
                plotnine.facet_grid(rows="signature", cols="metadata_field",
                                    scales="free", space="free_x") +
                plotnine.ylab("Signature Weight") +
                plotnine.xlab("Metadata Value") +
                plotnine.guides(fill=plotnine.guide_legend(title="Signature")) +
                plotnine.theme(
                    axis_text_x=plotnine.element_text(
                        rotation=disc_rotate_labels)
                ) +
                self.fill_scale
        )
        cont_plot: plotnine.ggplot = (
                plotnine.ggplot(
                    cont,
                    mapping=plotnine.aes(
                        x='metadata_value',
                        y='signature_weight',
                        color="signature"
                    )
                ) +
                plotnine.geom_point(**point_params) +
                plotnine.facet_grid(["signature", "metadata_field"],
                                    scales="free_y") +
                plotnine.ylab("Signature Weight") +
                plotnine.xlab("Metadata Value") +
                plotnine.guides(
                    color=plotnine.guide_legend(title="Signature")) +
                self.color_scale
        )
        if show_significance:
            disc_plot = self.__attach_significance(
                disc_plot,
                against=against,
                significance_formatter=significance_formatter,
                univar_params=univariate_test_params
            )

        return disc_plot, cont_plot

    def __attach_significance(
            self,
            plt_metadata: plotnine.ggplot,
            against: pd.DataFrame,
            significance_formatter: Callable[[float, float, float], str],
            univar_params: Dict[str, Any]
    ) -> plotnine.ggplot:
        """Attach results of univariate testing to metadata plots."""

        significance_formatter = (
            Decomposition.significance_format
            if significance_formatter is None else
            significance_formatter
        )
        # Recover metadata in format we need
        df: pd.DataFrame = plt_metadata.data.pivot_table(
            columns="metadata_field", index="sample",
            values="metadata_value", aggfunc=np.min
        )

        univar_res: pd.DataFrame = self.univariate_tests(
            metadata=df,
            against=against,
            **({} if univar_params is None else univar_params)
        )

        # Need to calculate where to place the label on the x and y axes, as
        # sadly we can't just say middle and top a bit
        xpos: pd.DataFrame = (df.nunique() / 2) + 0.5
        y_max: pd.DataFrame = (
            plt_metadata.data[['signature', 'signature_weight']]
            .groupby("signature")
            .max()
        )
        y_min: pd.DataFrame = (
            plt_metadata.data[['signature', 'signature_weight']]
            .groupby("signature")
            .min()
        )
        y_range: pd.DataFrame = y_max - y_min
        y_pos: pd.DataFrame = y_max + (y_range * 0.1)
        y_pos.columns = ['y']
        ypos_const: float = y_max.max().iloc[0] * 1.1
        label_df: pd.DataFrame = (
            univar_res.merge(
                right=xpos.to_frame("x"),
                right_index=True,
                left_on="md",
                how="left"
            ).merge(
                right=y_pos,
                right_index=True,
                left_index=True,
                how="left"
            )
        )
        label_df['label_str'] = [
            significance_formatter(*tuple(x)) for _, x in
            label_df[['p', 'local_adj_p', 'global_adj_p']].iterrows()
        ]
        # Need to match columns to facet fields
        label_df = label_df.rename(columns=dict(md="metadata_field"))

        # Add a text layer
        plt_metadata = (
            plt_metadata +
            plotnine.geom_text(
                data=label_df,
                mapping=plotnine.aes(
                    x="x",
                    y="y",
                    label="label_str"
                )
            ) +
            plotnine.scale_y_continuous(expand=(0.05, 0, 0.1, 0))
        )
        return plt_metadata

    @staticmethod
    def significance_format(
            p: float,
            local_adj: float,
            global_adj: float
    ) -> str:
        """Convert p-values from unvariate tests to display strings.
        
        By default, this will use the following strategy:

        * global_adj =< 0.01 -> ***
        * global_adj =< 0.05 -> **
        * global_adj =< 0.1  -> *
        * p =< 0.01 -> ..
        * p =< 0.05 -> .

        If implementing a custom formater, ``p`` is the unadjusted p-value,
        ``local_adj`` the adjusted p-value only considering the tests for one
        metadata category, and ``global_adj`` considering all tests.
        """
        qc: List[Tuple[float, str]] = [(0.01, "***"), (0.05, "**"), (0.1, "*")]
        pc: List[Tuple[float, str]] = [(0.01, ".."), (0.05, '.')]
        for thresh, val in qc:
            if global_adj <= thresh:
                return val
        for thresh, val in pc:
            if p <= thresh:
                return val
        return ""

    def plot_weight_distribution(
            self,
            threshold: float = 0.0,
            scale_transform: Optional[str] = "log10",
            nrows: int = 1
    ) -> plotnine.ggplot:
        """Plot the distribution of feature weights in each signature.
        
        The distribution of signature weights helps described how mixed the 
        features are which describe a sample. This will sort feature weights
        for each signature independently, and plot a bar for the weight of
        each feature. So distributions which are longer indicate more features
        contribute to that signature, and the height of bars indicates
        whether this is a long tail of low weights, all even, etc.

        :param threshold: Set any weight below this to 0. Effectively, consider
        very low weights to not contribute to the signature.
        :param scale_transform: Transformation to apply to the feature weight
        axis. Can be any of the transforms in `mizani`. For no transformation,
        pass None or "identity".
        :param nrows: Number of rows in the plot. Defaults to having all plots
        on one row for comparability.
        """

        def apply_sort(x):
            x = x.sort_values(ascending=False)
            x.index = range(1, len(x) + 1)
            return x

        weight_df: pd.DataFrame = self.scaled('w').apply(
            lambda x: apply_sort(x)).stack().reset_index()
        weight_df.columns = ['order', 'signature', 'weight']
        weight_df = weight_df.loc[weight_df['weight'] > threshold]

        if scale_transform is None:
            scale_transform = "identity"
        # If log transforming, 1+relative has a more intuitive interpretation
        scale_label: str = ""
        if "log" in scale_transform:
            logger.info("Using 1+relative weight for log scale")
            weight_df['weight'] = 1 + weight_df['weight']
            scale_label = "(1+relative weight)"

        fig_weight_dist = (
            plotnine.ggplot(
                weight_df.loc[weight_df['weight'] > threshold],
                plotnine.aes(x='order', y='weight', fill='signature')
            )
            + plotnine.geom_col(width=1)
            + plotnine.facet_wrap(['signature'], nrow=nrows)
            + plotnine.scale_y_continuous(trans=scale_transform)
            + plotnine.scale_x_continuous(expand=(0, 0))
            + plotnine.coord_flip()
            + plotnine.theme(
                figure_size=(len(self.names), 2),
                axis_text_x=plotnine.element_text(angle=35, hjust=1)
            )
            + self.fill_scale
            + plotnine.ylab(
                f"Relative Weight\n({scale_transform}{scale_label})"
            )
            + plotnine.xlab("")
            + plotnine.guides(fill="none")
        )
        return fig_weight_dist

    def save(self,
             out_dir: Union[str, pathlib.Path],
             compress: bool = False,
             param_path: Optional[pathlib.Path] = None,
             x_path: Optional[pathlib.Path] = None,
             symlink: bool = True,
             delim: str = "\t",
             plots: Optional[Union[bool, Iterable[str]]] = None) -> None:
        """Write decomposition to disk.

        Export this decomposition and associated data. This is written to text
        type files (tab separated for tables, yaml for dictionaries) to allow
        simpler reading in other analysis environments such as R. Exceptions
        are raised if any tables cannot be written, but plots are allowed to
        fail though will produce log entries.

        :param out_dir: Directory to write to. Must be empty.
        :param compress: Create compressed .tar.gz rather than directory.
        :param param_path: Path to YAML file containing parameters used. If not
            given will create a copy in the directory. If given and symlink
            is True, will try to make a symlink to parameters file.
        :param x_path: Path to X matrix used. Behaves as param_path for copies/
            symlinks.
        :param symlink: Make symlinks ot param_path and x_path if possible.
        :param delim: Delimiter to used for tabular output.
        :param plots: Determine which plots to write. When left default
            (None) this will produce all plots if there are 500 or fewer samples.
            If True, all plots will produced; if False no plots will be produced.
            If a list is provided, any plots named in the list will be produced,
            i.e. if given ['pcoa', 'modelfit', 'radar'], plots from
            :meth:`plot_pcoa` and : meth:`plot_modelfit` would be produced.
            'radar' would be ignored as there is no `plot_radar` method.
        """

        out_dir = pathlib.Path(out_dir)
        logger.debug("Create decomposition output dir: %s", out_dir)
        out_dir.mkdir(parents=True, exist_ok=False)

        # Output tables
        for tbl, fname in (
                [(x, f'{x}.tsv') for x in
                 ['h', 'w', 'model_fit', 'primary_signature',
                  'quality_series']] +
                [(self.representative_signatures(),
                  'representative_signatures.tsv'),
                 (self.monodominant_samples(), 'monodominant_samples.tsv'),
                 (self.scaled('h'), "h_scaled.tsv"),
                 (self.scaled('w'), "w_scaled.tsv")]
        ):
            logger.debug("Write decomposition table: %s", out_dir / fname)
            df: Union[pd.Series, pd.DataFrame] = (
                getattr(self, tbl) if isinstance(tbl, str) else tbl
            )
            df.to_csv(out_dir / fname, sep=delim)

        # Output some YAML properties
        with open(out_dir / "properties.yaml", "w") as f:
            yaml.safe_dump(dict(
                colors=self.colors,
                names=self.names
            ), f)

        # Output feature mapping
        if self.feature_mapping is not None:
            self.feature_mapping.to_df().to_csv(
                out_dir / "feature_mapping.tsv",
                sep=delim
            )

        # Symlink or write data / parameters
        _write_symlink(path=out_dir / "x.tsv",
                       target=x_path,
                       save_fn=lambda x: self.parameters.x.to_csv(
                           x,
                           sep=delim
                       ),
                       symlink=symlink)
        _write_symlink(path=out_dir / "parameters.yaml",
                       target=param_path,
                       save_fn=self.parameters.to_yaml,
                       symlink=symlink)

        # Determine which plots to produce

        # Attempt to output default plots
        for plot_fn in self.__plots_to_save(plot=plots):
            plt_path: pathlib.Path = out_dir / plot_fn
            logger.debug("Write decomposition plot: %s", plt_path)
            if plot_fn == "plot_metadata":
                # Requires a metadata object, so skip
                continue
            try:
                plt_obj: Union[
                    plotnine.ggplot, matplotlib.figure.Figure,
                    marsilea.WhiteBoard] = (
                    getattr(self, plot_fn)()
                )

                if isinstance(plt_obj, plotnine.ggplot):
                    plt_obj.save(plt_path.with_suffix(".pdf"))
                elif isinstance(plt_obj, marsilea.WhiteBoard):
                    plt_obj.save(plt_path.with_suffix(".pdf"))
                else:
                    plt_obj.savefig(plt_path.with_suffix(".pdf"))
            except Exception as e:
                # TODO: This is not a great pattern, refine exception catching
                logger.warning("Failed to save plot %s (%s)",
                                plt_path, str(e))

        # Compress if requested
        if compress:
            with tarfile.open(
                    out_dir.with_suffix(".tar.gz"),
                    'w:gz') as tar:
                for f in out_dir.iterdir():
                    if f.is_file():
                        tar.add(f, arcname=f.name)
            shutil.rmtree(out_dir)

    def __plots_to_save(
            self, plot: Optional[Union[bool, Iterable[str]]]) -> Set[str]:
        """Determine which plotting functions to attempt when saving a
        decomposition.

        :param plot: bool (all/nothing), list of plots, or None for default.
        :return: List of plot function names to call
        """

        all_plots: Set[str] = set(x for x in dir(self) if "plot_" in x)
        if isinstance(plot, bool):
            return all_plots if plot else {}
        if plot is not None:
            return all_plots.intersection(set(f'plot_{x}' for x in plot))
        # Default behaviour is to not make plots with all samples in the
        # case that there are lot of samples
        if self.h.shape[1] > self.DEF_PLOT_SAMPLE_LIMIT:
            logger.warning(
                ("More than %s samples; will not save plots with points for "
                 "each sample by default (%s). To force these plots to be "
                 "saved, pass plot=True to save()."),
                self.DEF_PLOT_SAMPLE_LIMIT,
                self.DEF_PLOT_SAMPLE_LIMIT_REMOVE
            )
            return all_plots.difference(self.DEF_PLOT_SAMPLE_LIMIT_REMOVE)
        return all_plots


    @staticmethod
    def save_decompositions(decompositions: Dict[int, List[Decomposition]],
                            output_dir: pathlib.Path,
                            symlink: bool = True,
                            delim: str = "\t",
                            compress: bool = False,
                            **kwargs) -> None:
        """Save multiple decompositions to disk.

        Write multiple decompositions to disk. The structure is that a
        directory is created for each rank, then within that a directory for
        each decomposition. By default the input data and parameters will be
        saved at the top level, and symlinked to by each individual
        decomposition.

        The files output are tables for W and H matrices, scaled W and H,
        tables basic analyses (primary es etc), and all default plots where
        possible.

        :param decompositions: Decompositions in form output by
            :func:`decompositions`.
        :param output_dir: Directory to write to which is either empty or
            does not exist.
        :param symlink: Symlink the parameters and input X files.
        :param delim: Delimiter for tabular output.
        :param compress: Compress each decomposition folder to .tar.gz
        :param **kwargs: Passed to :meth:`Decomposition.save`
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        # Ensure this is empty
        if len([x for x in output_dir.iterdir()
                if x.name != "parameters.yaml"]) > 0:
            raise FileExistsError(
                f"Output directory {output_dir} is not empty; directory "
                "decompositions are bing saved to should be empty.")
        # Write data to top level
        x: Decomposition = list(decompositions.values())[0][0]
        x_path: pathlib.Path = output_dir / "x.tsv"
        x.parameters.x.to_csv(x_path, sep=delim)
        for rank, ds in decompositions.items():
            rank_dir: pathlib.Path = output_dir / str(rank)
            rank_dir.mkdir()
            d: Decomposition
            for i, d in enumerate(ds):
                decomp_dir: pathlib = rank_dir / str(i)
                d.save(decomp_dir,
                       compress=compress,
                       param_path=None,
                       x_path=x_path,
                       symlink=symlink,
                       delim=delim,
                       **kwargs)

    @staticmethod
    def __load_stream(name: str,
                      stream,
                      delim: str = "\t"
                      ) -> Optional[Union[pd.DataFrame, pd.Series, Dict]]:
        """Read a file as appropriate type based on name. Intended to handle
        files read from directory or tarfile."""

        xtn: str = name.split(".")[-1]
        res_obj = None
        if xtn == "tsv":
            df: pd.DataFrame = pd.read_csv(stream,
                                           index_col=0,
                                           sep=delim)
            # Collapse to series if only a single column
            res_obj = df if df.shape[1] > 1 else df.iloc[:, 0]
        elif xtn == "yaml":
            res_obj = yaml.safe_load(stream)
        else:
            logger.debug("Ignored file %s due to extension", name)
        stream.close()
        return res_obj

    @staticmethod
    def __load_gzip(gzip: os.PathLike,
                    delim: str = "\t"
                    ) -> Dict[str, Union[pd.DataFrame, pd.Series, Dict]]:
        """Load decomposition data from .tar.gz."""

        # Loop through tar members and load as appropriate type based on
        # extension
        f: tarfile.TarFile
        with tarfile.open(gzip, 'r:gz') as f:
            return {
                pathlib.Path(n).name: Decomposition.__load_stream(
                    n, f.extractfile(n), delim)
                for n in f.getnames()
                if pathlib.Path(n).name in Decomposition.LOAD_FILES
            }

    @staticmethod
    def __load_dir(dir: os.PathLike,
                   delim: str = "\t"
                   ) -> Dict[str, Union[pd.DataFrame, pd.Series, Dict]]:
        """Load decomposition data from directory."""

        # Loop through dir contents and load
        dir_path: pathlib.Path = pathlib.Path(dir)
        return {n.name: Decomposition.__load_stream(n.name, n.open(), delim)
                for n in dir_path.iterdir()
                if n.name in Decomposition.LOAD_FILES}

    @staticmethod
    def load(in_dir: os.PathLike,
             x: Optional[Union[pd.DataFrame, str, os.PathLike]] = None,
             delim: str = "\t"):
        """Load a decomposition from disk.

        Loads a decomposition previously saved using :meth:`save`. Will
        automatically determine whether this is a directory or .tar.gz.
        Can provide a DataFrame of the :math:`X` input matrix, primarily this is
        so when loading multiple decompositions they can all reference the
        same object. Can also provide an explicit path; if not provided will
        attempt to load from ``x.tsv``.

        :param in_dir: Directory or .tar.gz containing decomposition.
        :param x: Either the X input matrix as a DataFrame, or a path to
            a delimiter-separated copy of the X matrix. If None, will attempt
            to load from x.tsv.
        :param delim: Delimiter for tabular data
        """

        # Get data from either directory or tar.gz.
        in_path: pathlib.Path = pathlib.Path(in_dir)
        data: Dict[str, Union[pd.DataFrame, pd.Series, Dict]] = (
            Decomposition.__load_gzip(in_path, delim)
            if in_path.suffix == ".gz" else
            Decomposition.__load_dir(in_path, delim)
        )

        # Get X if required
        x_mat: pd.DataFrame = None
        if isinstance(x, pd.DataFrame):
            x_mat = x
        elif x is not None:
            # Should be a pathlike
            x_path: pathlib.Path = pathlib.Path(x)
            x_mat = pd.read_csv(x_path,
                                index_col=0,
                                sep=delim)
        if x_mat is None:
            # Attempt to recover from directory dict
            x_mat = data['x.tsv']

        # Validate that we have all the expected files
        if not set(Decomposition.LOAD_FILES[1:]).issubset(set(data.keys())):
            missing: Set[str] = (set(Decomposition.LOAD_FILES[:2])
                                 .difference(set(data.keys())))
            raise IndexError(
                f"Required file(s) {missing} not found when loading from "
                f"{str(in_path)}")

        # Make a parameters object without x as we are handling that
        # separately
        param_dict: Dict[str, Any] = data['parameters.yaml']
        xless: Dict[str, Any] = {n: v for n, v in param_dict.items()
                                 if n != 'x'}
        params: NMFParameters = NMFParameters(
            x=x_mat,
            **xless
        )

        # Feature mapping will not exist for de-novo decompositions
        if "feature_mapping.tsv" in data:
            logger.warning(
                "Feature mappings are present, but are not currently read when loading "
                "a decomposition from disk")
            # TODO: Make GenusMapping object from table

        # Make Decomposition object
        decomp: Decomposition = Decomposition(
            parameters=params,
            h=data['h.tsv'],
            w=data["w.tsv"]
        )
        decomp.names = data['properties.yaml']['names']
        decomp.colors = data['properties.yaml']['colors']

        return decomp

    @staticmethod
    def load_decompositions(in_dir: os.PathLike,
                            delim: str = "\t"
                            ) -> Dict[int, List[Decomposition]]:
        """Load multiple decompositions.

        Load a set of decompositions previously saved using
        :meth:`save_decompositions`. Will attempt to share a reference to
        the same :math:`X` matrix for memory reasons. The output is a dictionary
        with keys being ranks, and values being lists of decompositions
        for that rank.

        :param in_dir: Directory to read from
        :param delim: Delimiter for tabular data files
        """
        # Load x
        in_path: pathlib.Path = in_dir
        x_mat: pd.DataFrame = pd.read_csv(
            in_path / "x.tsv",
            index_col=0,
            sep=delim
        )

        # Get list of numeric subdirectories
        num_subdirs: List[pathlib.Path] = [
            d for d in in_path.iterdir()
            if d.is_dir() and re.match(r'^\d*$', d.name) is not None
        ]
        num_subdirs = sorted(num_subdirs,
                             key=lambda x: int(x.name))
        decomps: Dict[int, List[Decomposition]] = {}
        for subdir in num_subdirs:
            decomps[int(subdir.name)] = [
                Decomposition.load(d, x=x_mat, delim=delim)
                for d in sorted(
                    (d for d in subdir.iterdir() if
                     re.match(r'^\d*(?:\.tar\.gz)?$', d.name) is not None and
                     (d.is_dir() or d.suffix == ".gz")),
                    key=lambda x: int(x.name.split('.')[0])
                )
            ]
        return decomps

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
        pal_len: List[int] = [len(x) for x in Decomposition.DEF_SCALES]
        if n <= max(pal_len):
            return next(x for x in Decomposition.DEF_SCALES
                        if n <= len(x))[:n]
        else:
            logger.warning("Some colours are duplicated when plotting over 20 "
                            "signatures")
            return (
                (Decomposition.DEF_SCALES[-1] *
                 math.ceil(n / len(Decomposition.DEF_SCALES)))[:n]
            )


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
              default="random",
              show_default=True,
              help="""Method to use when intialising H and W for 
              decomposition.""")
@click.option("--n_runs",
              required=False,
              type=int,
              default=20,
              show_default=True,
              help=("Number of times to run decomposition for each rank. "
                    "Ignored when init is a deterministic method "
                    "(nndsvd/nndsvda)."))
@click.option("--top_n",
              required=False,
              type=int,
              default=5,
              show_default=True,
              help=("Keep and report only the best top_n decompositions "
                    "of the n_runs decompositions produced. Which are the best "
                    "decompositions is determined by top_criteria. Ignored "
                    "when init is a deterministic method (nndsvd/nndsvda)."))
@click.option("--top_criteria",
              required=False,
              type=click.Choice(list(Decomposition.TOP_CRITERIA.keys())),
              default="beta_divergence",
              show_default=True,
              help=("Criteria used to determine which of the n_runs "
                    "decompositions to keep and report.")
              )
@click.option("--compress/--no_compress",
              type=bool,
              default=False,
              show_default=True,
              help="Compress output folders to .tar.gz. Default is to output "
                   "each decomposition to a separate folder.")
@click.option("--symlink/--no-symlink",
              type=bool,
              default=False,
              show_default=True,
              help="Create a symlink for files which do not vary between runs ("
                   "input, parameters, etc). If disabled, will make redundant "
                   "copies.")
@click.argument("ranks",
                nargs=-1,
                type=int)
def cli_decompose(
        input: str,
        output_dir: str,
        delimiter: str,
        progress: bool,
        verbosity: str,
        seed: int,
        l1_ratio: float,
        alpha: float,
        max_iter: int,
        beta_loss: str,
        init: str,
        n_runs: int,
        top_n: int,
        top_criteria: str,
        compress: bool,
        ranks: List[int],
        symlink: bool
) -> None:
    """Decompositions for RANKS.

    RANKS is a list of ranks for which to generate decompositions.

    Generate a number of decompositions for each the specified ranks. NMF
    solutions are non-unique and depend on initialisation, so when using an
    initialisation with randomness multiple solutions can be produced.
    From these solutions, the best can be retained based on criteria such
    as reconstruction error or cosine similarity.

    Some initialisation methods are deterministic, and as such only a single
    decomposition will be produced.

    The output is H and W matrices for each decomposition, tables of quality
    scores, and some analyses with default parameters. For further analysis,
    decompositions can be loaded using Decomposition.from_dir, or tables used
    directly for custom analyses. By default, a symlink to the input data
    """
    __configure_logger(verbosity)

    # Validate arguments
    # Make ranks unique, sort from high to low
    ranks: List[int] = sorted(set(ranks), reverse=True)
    # Require output directory to be empty or not exist, so not
    # overwriting results
    output_path: pathlib.Path = pathlib.Path(output_dir)
    if output_path.is_dir():
        if len(list(output_path.iterdir())) > 0:
            logger.fatal("Output directory %s must be empty",
                          output_path)
        return
    if not output_path.exists():
        # Attempt to create directory and ensure it is writable
        output_path.mkdir(parents=True, exist_ok=True)

    # Read input data
    x: pd.DataFrame = pd.read_csv(input, sep=delimiter, index_col=0)
    __validate_input_matrix(x)

    # Log params being used
    param_str: str = (
        f"\n"
        f"Data Locations\n"
        f"------------------------------\n"
        f"Input:            {input}\n"
        f"Output:           {output_dir}\n"
        f"\n"
        f"Decomposition Parameters\n"
        f"------------------------------\n"
        f"Random Starts:    {n_runs}\n"
        f"Keep Top N:       {top_n}\n"
        f"Top Criteria:     {top_criteria}\n"
        f"Seed:             {seed}\n"
        f"Ranks:            {ranks}\n"
        f"L1 Ratio:         {l1_ratio}\n"
        f"Alpha:            {alpha}\n"
        f"Max Iterations:   {max_iter}\n"
        f"Beta Loss:        {beta_loss}\n"
        f"Initialisation:   {init}\n"
    )
    logger.info(param_str)

    # Write parameters to output directory. Also ensures we have write access
    # before engaging in expensive computation.
    try:
        with (output_path / "parameters.yaml").open(mode="w") as f:
            yaml.safe_dump(dict(
                input=input,
                output=output_dir,
                ranks=ranks,
                n_runs=n_runs,
                top_n=top_n,
                top_criteria=top_criteria,
                seed=seed,
                l1_ratio=l1_ratio,
                alpha=alpha,
                max_iter=max_iter,
                beta_loss=beta_loss,
                init=init
            ), f)
    except Exception as e:
        logger.fatal("Unable to write to output directory")
        return

    # Make decompositions
    logger.info("Beginning decompositions")
    decomps: Dict[int, List[Decomposition]] = decompositions(
        x=x,
        ranks=ranks,
        random_starts=n_runs,
        top_n=top_n,
        top_criteria=top_criteria,
        seed=seed,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        beta_loss=beta_loss,
        init=init,
        progress_bar=progress
    )
    logger.info("Decomposition complete")

    # Write decompositions
    logger.info("Write decompositions to %s", output_path)
    Decomposition.save_decompositions(decomps,
                                      output_dir=output_path,
                                      symlink=symlink,
                                      delim=delimiter,
                                      compress=compress)
    logger.info("Decomposition completed")


def __configure_logger(verbosity: str) -> None:
    """Set logging format and level for CLI. Verbosity should be warning,
    info, or debug."""

    log_level: int = logging.WARNING
    match verbosity:
        case "debug":
            log_level = logging.DEBUG
        case "info":
            log_level = logging.INFO
    logger.setLevel(log_level)


def __validate_input_matrix(x: pd.DataFrame) -> None:
    """Check input matrix has sensible format and values."""
    # Check shape
    if x.shape[0] < 2 or x.shape[1] < 2:
        logger.fatal("Loaded matrix invalid shape: (%s)", x.shape)
        raise ValueError("Matrix has invalid shape")
    # All columns should be numeric (either float or int)
    if not all(np.issubdtype(x, np.integer) or np.issubdtype(x, np.floating) for
               x in x.dtypes):
        logger.fatal("Loaded matrix has non-numeric columns")
        raise ValueError("Loaded matrix has non-numeric columns")
    # Non-negativity constraint
    if (x < 0).any().any():
        logger.fatal("Loaded matrix contains negative values")
        raise ValueError("Loaded matrix contains negative values")
    # Don't accept NaN
    if x.isna().any().any():
        logger.fatal("Loaded matrix contains NaN/NA values")
        raise ValueError("Loaded matrix contains NaN/NA values")


def __alpha_values(alphas: Optional[List[float]]) -> List[float]:
    """Return default alpha value to search if alphas is None."""
    alpha_list: List[float] = [] if alphas is None else list(sorted(alphas))
    if alphas is None or len(alpha_list) == 0:
        logger.info("Using default range of alphas values")
        return DEF_ALPHAS
    return alpha_list


def _wisconsin_double_standardise(h: pd.DataFrame) -> pd.DataFrame:
    """Wisconsin double standardise matrix as per R function cmdscale"""
    logger.info("Wisconsin double standardising data")
    # Species maximum standardisation - each species max is 1
    h = (h.T / h.max(axis=1)).T
    # Sample total standardisation
    std_h: pd.DataFrame = h / h.sum()
    return std_h


def _is_series_continuous(series: pd.Series) -> bool:
    """True if a column is float or int."""
    try:
        return (
                np.issubdtype(series.dtype, np.floating) or
                np.issubdtype(series.dtype, np.integer)
        )
    except TypeError as e:
        return False


def _is_series_discrete(series: pd.Series) -> bool:
    """True if a column is object type, and type is string, or there are
    less than 3n/4 distinct values, or is pd.Categorical."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if series.dtype == object:
        if isinstance(series[0], str):
            return True
        unique_vals: Set = set(series)
        return len(unique_vals) > (0.75 * len(series[~series.isna()]))
    return False


def _set_intersect_and_difference(
        l: Iterable[Hashable],
        r: Iterable[Hashable]
) -> Tuple[Set, Set, Set]:
    """Return l-r, lnr, r-l."""
    lset: Set = set(l)
    rset: Set = set(r)
    return (
        lset.difference(rset),
        lset.intersection(rset),
        rset.difference(lset)
    )

def _cbar(
        consensus_matrices: Iterable[sparse.sparray]
) -> np.ndarray:
    """Produce a mean consensus matrix from consensus matrices from multiple
    decompositions.

    For large studies (n=10k samples), holding all these nxn matrices in
    memory is inefficient, so we permit passing a generator to this
    function.

    :param consensus_matrices: Consensus matrices in lower triangular sparse
        format.
    :returns: Lower triangular consensus matrix in numpy format
    """

    summed: Tuple[int, sparse.sparray] = reduce(
        __c_add, enumerate(consensus_matrices))
    c_bar: np.ndarray = summed[1].toarray() / (summed[0] + 1)
    return c_bar

def __c_add(
        a: Tuple[int, sparse.sparray],
        b: Tuple[int, sparse.sparray],
) -> Tuple[int, sparse.sparray]:
    """Add enumerated c matrices."""
    if a is None:
        return b[0], b[1].astype('int')
    return b[0], a[1] + b[1].astype('int')

def _cophenetic_correlation(
        c_bar: np.ndarray
) -> float:
    """Calculate Cophenetic Correlation Coefficient (Brunet 2004).

    Takes a mean consensus matrix, and calculate cophenetic correlation.
    Performs clustering using scipys linkage method.
    """
    # The np.where() symmetrises the matrix, for only positive values.
    dist: np.array = squareform(
        1 - np.where(np.tri(*c_bar.shape, dtype=bool), c_bar, c_bar.T)
    )
    link = linkage(dist, method="average")
    ccc, _ = cophenet(link, dist)
    return ccc

def _dispersion(
        c_bar: np.ndarray
) -> float:
    """Calculate dispersion coefficient (Park 2007)."""
    sym: np.ndarray = np.where(np.tri(*c_bar.shape, dtype=bool), c_bar, c_bar.T)
    return np.sum(4 * ((c_bar - 0.5) * (c_bar - 0.5))) / (sym.shape[0] ** 2)