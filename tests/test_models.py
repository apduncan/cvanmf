"""Tests for model and example data loading."""
import numpy as np
import pandas as pd

from enterosig import models


def test_five_es() -> None:
    """Load the 5 ES model and check it has the right properties."""

    df: pd.DataFrame = models.five_es()
    assert df.shape == (592, 5), "5ES model has wrong dimensions"
    assert set(df.columns) == {"ES_Bact", "ES_Bifi", "ES_Prev", "ES_Firm",
                               "ES_Esch"}, "5ES model has wrong columns"

def test_example_abundance() -> None:
    """Load the example data and check it has the right properties."""

    df: pd.DataFrame = models.example_abundance()
    assert df.shape == (586, 1152), "Sample data has wrong dimensions"
    # Check all columns are numeric
    all_num: bool = all(np.vectorize(
        lambda x: np.issubdtype(x, np.number)
    )(df.dtypes))
    assert all_num == True, "Sample data not numeric"