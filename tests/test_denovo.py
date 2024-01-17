"""Tests for denovo ES generation."""
from typing import List

import numpy as np
import pandas as pd

from enterosig import models
from enterosig.denovo import BicvSplit, bicv_shuffles


def test_bicv_shuffles():
    """Shuffling and splitting working as intended?"""
    df: pd.DataFrame = models.example_abundance()
    shuffles: List[BicvSplit] = bicv_shuffles(
        df, n=2, seed=4298
    )

    # Overall number of entries in matrix is consistent
    shuf_len: int = sum(x.size for x in shuffles[0])
    assert shuf_len == df.size, \
        "Shuffled and split matrix different size to input"

    # Number of columns and rows consistent
    shuf_cols: int = sum(x.shape[1] for x in shuffles[0][0:3])
    shuf_rows: int = sum(x.shape[0] for x in (
        shuffles[0][0], shuffles[0][3], shuffles[0][6]))
    assert shuf_rows == df.shape[0], "Split and input shapes inconsistent"
    assert shuf_cols == df.shape[1], "Split and input shapes inconsistent"

    # Sum of entries consistent
    entry_sum: float = sum(x.sum().sum() for x in shuffles[0])
    assert np.isclose(entry_sum, df.sum().sum()), \
        "Sum of values in split and input are inconsistent"

    # Sorted list of entries as consistent
    input_vals: np.ndarray = np.sort(df.values.ravel())
    shuffled_vals: np.ndarray = np.sort(np.concatenate(
        list(x.values.ravel() for x in shuffles[0])
    ))
    assert all(np.isclose(input_vals, shuffled_vals))
