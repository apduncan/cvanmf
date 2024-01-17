"""
Generate new Enterosignature models using NMF decomposition

It is likely that additional Enterosignatures exist in data with different
populations, or new Enterosignatures might become evident with ever increasing
taxonomic resolution. This module provides functions to generate new models
from data, which encompasses three main steps: rank selection, regularisation
selection, and model inspection. The first of these two steps involves
running decompositions multiple times for a range of values, and can be time
consuming. Methods are provided to run the whole process on a single machine,
but also for running individual decompositions, which are used by the
accompanying nextflow pipeline to allow spreading the computation across
multiple nodes in a HPC environment.
"""
import logging
import pathlib
from typing import Optional, NamedTuple, List, Iterable, Union

import numpy as np
import pandas as pd

# TODO: Refactor this to a class so the load / save methods can be attached
BicvSplit = List[pd.DataFrame]
"""Alias for a 3x3 split of a matrix."""


class BicvParameters(NamedTuple):
    """Parameters for a single bifold cross-validation run. See sklearn NMF
    documentation for more detail on parameters."""
    mx: List[pd.DataFrame]
    """Shuffled matrix split into 9 parts for bi-cross validation."""
    seed: Optional[int]
    """Random seed for initialising decomposition matrices; if None no seed
    used so results will not be reproducible."""
    rank: int
    """Rank of the decomposition."""
    alpha: float = 0.0
    """Regularisation parameter applied to both H and W matrices."""
    l1_ratio: float = 0.0
    """Regularisation mixing parameter. In range 0.0 <= l1_ratio <= 1.0."""
    max_iter: int = 3000
    """Maximum number of iterations during decomposition. Will terminate earlier
    if solution converges."""
    keep_mats: bool = False
    """Whether to return the H and W matrices as part of the results."""


def __bicv_split(df: pd.DataFrame) -> BicvSplit:
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
    chunks_samp: int = df.shape[1] // 3
    thresholds_feat: List[int] = [chunks_feat * i for i in range(1, 3)]
    thresholds_feat.insert(0, 0)
    thresholds_feat.append(df.shape[0] + 1)
    thresholds_samp: List[int] = [chunks_samp * i for i in range(1, 3)]
    thresholds_samp.insert(0, 0)
    thresholds_samp.append(df.shape[1] + 1)

    # Return the 9 matrices
    # TODO: Tidy this a little
    matrix_nb = [i for i in range(1, 3 * 3 + 1)]
    all_sub_matrices: List[pd.DataFrame] = [None] * 9
    done = 0
    row = 0
    col = 0
    while row < 3:
        while col < 3:
            done += 1
            all_sub_matrices[done - 1] = df.iloc[
                                         thresholds_feat[row]:thresholds_feat[row + 1],
                                         thresholds_samp[col]: thresholds_samp[col + 1]
                                         ]
            col += 1
        row += 1
        col = 0
    assert (len(all_sub_matrices) == len(matrix_nb))
    return all_sub_matrices


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


def bicv_shuffles(df: pd.DataFrame, n: int = 200, seed: int = None
                  ) -> List[BicvSplit]:
    """Shuffle and make 3x3 splits of matrix for bi-cross validation.
    
    :param df: Abundance matrix, features on row, samples on columns
    :type df: pd.DataFrame
    :param n: Number of shuffles
    :type n: int
    :param seed: Random seed for shuffles
    :type seed: int
    :returns: List of shuffled and split matrices, each of which itself is
        a length 9 list containing a DataFrame
    :rtype: List[List[pd.DataFrame]]
    """

    rnd: np.random.Generator = np.random.default_rng(seed)
    logging.info("Generating %s splits with seed %s", str(n), str(seed))
    shuffles: Iterable[pd.DataFrame] = map(__bicv_shuffle,
                                           (df for _ in range(n))
                                           )
    splits: List[BicvSplit] = list(map(__bicv_split, shuffles))
    return splits

def save_bicv_shuffles(shuffles: Iterable[BicvSplit],
                       path: pathlib.Path,
                       prefix: str = "",
                       compress: bool = True) -> None:
    """Write shuffled matrices to disk.

    Each shuffle will be saved as an npz file containing the 9 matrices,
    optionally with compression. File name is "{prefix}shuffle_{n}.npz.
    Compression is on by default, as often sparse microbiome data tends to
    create very large npz format files compared to their text representation
    counterparts, but can compress well. Primarily intended for pipeline
    use where it is convenient to pass a shuffle as an input file to a process.

    :param shuffles: Iterable of shuffled and split matrices
    :type shuffles: List[BicvSplit]
    :param path: Directory to save to
    :type path: pathlib.Path
    :param prefix: String to prefix each file with
    :type prefix: str
    :param compress: Save in compressed format
    """
    save_fn = np.savez_compressed if compress else np.savez
    logging.info("Saving shuffles to %s", str(path))
    for i, shuffle in shuffles:
        out_path: pathlib.Path = path / f"shuffle_{i}.npz"
        save_fn(out_path, shuffle)

def load_bicv_shuffles(path: Union[pathlib.Path, str],
                       allow_pickle: bool = True) -> List[BicvSplit]:
    """Load shuffled and split matrices from disk.

    Read shuffled matrices from npz files, probably produced by
    :function:`save_bicv_shuffles`

    :param path: Either a directory in which the only .npz files are the
        shuffles to load, or a glob which returns all the shuffles to load.
    :type path: Union[pathlib.Path, str]
    :param allow_pickle: Allow unpickling when loading. Necessary for compressed
        matrices. However, unpickling is insecure, so you can disable it if not
        using compression.
    :returns: List of shuffled split matrices
    :rtype: List[BicvSplit]
    """
    if isinstance(path, pathlib.Path):
        pass
    raise NotImplementedError()

