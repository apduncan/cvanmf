from typing import Tuple, List

import pandas as pd
import pytest

from cvanmf.data import leukemia, swimmer, synthetic_dense
from cvanmf.data.utils import ExampleData


def is_example_data_valid(
        d: ExampleData,
        name: str,
        short_name: str,
        shape: Tuple[int, int],
        has_row_metadata: bool,
        has_col_metadata: bool,
        expected_other_metadata: List[str],
        rank: int
) -> bool:
    """Test whether an example data is in the expected format"""
    __tracebackhide__ = True
    if not isinstance(d.data, pd.DataFrame):
        pytest.fail("Data missing or not a DataFrame.")
    if d.data.shape != shape:
        pytest.fail("Data not expected shape.")
    if d.name != name:
        pytest.fail("Incorrect name.")
    if d.short_name != short_name:
        pytest.fail("Short name incorrect.")
    if has_row_metadata and d.row_metadata is None:
        pytest.fail("Missing expected row metadata.")
    if has_col_metadata and d.col_metadata is None:
        pytest.fail("Missing expected col metadata")
    expected_other_metadata = (
        [] if expected_other_metadata is None else expected_other_metadata
    )
    missing: List[str] = [x for x in expected_other_metadata
                          if x not in d.other_metadata]
    if len(missing) > 0:
        pytest.fail(f"Missing expected metadata: {missing}")
    if rank != d.rank:
        pytest.fail("Unexpected rank.")
    # Test non-negativity
    if d.data.min().min() < 0.0:
        pytest.fail("Negative values in data.")
    return True


def test_leukemia():
    a = leukemia()
    is_example_data_valid(
        a,
        name="Leukemia (ALL/AML) Gene Expression",
        short_name="ALL_AML",
        shape=(5000, 38),
        has_col_metadata=True,
        has_row_metadata=True,
        rank=4,
        expected_other_metadata=[]
    )

def test_swimmer():
    a = swimmer()
    is_example_data_valid(
        a,
        name="Swimmer Dataset",
        short_name="swimmer",
        shape=(1024, 256),
        has_col_metadata=False,
        has_row_metadata=False,
        rank=17,
        expected_other_metadata=[]
    )

def test_dense():
    a = synthetic_dense(
        h_sparsity=0.5, k=4, shared_features=0.5, keep_mats=True,
        m=100, n=150
    )
    is_example_data_valid(
        a,
        name="Synthetic Dense Data",
        short_name="SYNTH_DENSE",
        shape=(100, 150),
        has_row_metadata=True,
        has_col_metadata=False,
        expected_other_metadata=['h', 'w', 'shared_features', 'h_sparsity',
                                 'scale_lognormal_params',
                                 'normal_noise_params'],
        rank=4
    )
    # Check sparsity
    sparsity: float = ((a.other_metadata['h'] == 0.0).sum().sum() /
                       (a.other_metadata['h'].shape[0] *
                        a.other_metadata['h'].shape[1]))
    assert sparsity == 0.5, ("Sparsity in H matrix doesn't match requested "
                             "value.")
