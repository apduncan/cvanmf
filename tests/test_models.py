"""Tests for model and example data loading."""
import numpy as np
import pandas as pd

from cvanmf.denovo import Decomposition
from cvanmf.models import five_es, Signatures
from cvanmf.data.utils import example_abundance


def test_five_es() -> None:
    """Load the 5 ES model and check it has the right properties."""

    model: Signatures = five_es()
    assert model.w.shape == (592, 5), "5ES model has wrong dimensions"
    assert set(model.w.columns) == {"ES_Bact", "ES_Bifi", "ES_Prev", "ES_Firm",
                                    "ES_Esch"}, "5ES model has wrong columns"

    # Test that this can be applied to new data without error
    y: pd.DataFrame = example_abundance().iloc[:, :20]
    res: Decomposition = model.reapply(y)
    assert res.h.shape[0] == model.w.shape[1], \
        "Incorrect number of signatures in reapplied results."
    assert set(res.h.columns) == set(y.columns), \
        "Incorrect number of columns in reapplied results."
    assert not res.h.isna().any().any(), \
        "NA values in reapplied results"


def test_reapply() -> None:
    """Can a model be applied to new data and retain the right properties."""
    esm: Signatures = five_es()
    # Make some dummy data to fit to the 5 ES model
    h: np.ndarray = np.random.uniform(low=0, high=1, size=[5, 15])
    wh: pd.DataFrame = esm.w.dot(h)
    wh.columns = [f"SMPL_{i}" for i in wh.columns]
    new_decomp: Decomposition = esm.reapply(wh)
    assert isinstance(new_decomp, Decomposition), "Reapply returned wrong type."
    assert dict(
        zip(new_decomp.names, new_decomp.colors)
    ) == esm.colors, "Colors not retained when reapplied."


def test_example_abundance() -> None:
    """Load the example data and check it has the right properties."""

    df: pd.DataFrame = example_abundance()
    assert df.shape == (586, 1152), "Sample data has wrong dimensions"
    # Check all columns are numeric
    all_num: bool = all(np.vectorize(
        lambda x: np.issubdtype(x, np.number)
    )(df.dtypes))
    assert all_num == True, "Sample data not numeric"
