"""Tests for denovo ES generation."""
import itertools
import math
import pathlib
import random
from typing import List, Dict

import matplotlib.pyplot
import numpy as np
import pandas as pd
import plotnine
import pytest

from enterosig import models
from enterosig.denovo import BicvSplit, BicvFold, bicv, _cosine_similarity, rank_selection, BicvResult, \
    plot_rank_selection, decompose, NMFParameters, decompositions, Decomposition

@pytest.fixture
def small_overlap_blocks(scope="session") -> pd.DataFrame:
    """Small overlapping block diagonal matrix with k=4, for use in testing
    de-novo methods."""

    # Matrix dimensions
    i, j = 100, 100
    # Rank of matrix (number of blocks)
    k: int = 3

    # Width of blocks without overlap
    base_h, tail_h = divmod(i, k)
    base_w, tail_w = divmod(j, k)
    # Overlap proportion - proportion of block's base dimension to extend 
    # block by
    overlap_proportion: float = 0.1
    overlap_h: int = math.ceil(base_h * overlap_proportion)
    overlap_w: int = math.ceil(base_w * overlap_proportion)
    # Make a randomly filled matrix, multiply by mask matrix which has 0 
    # or 1 then apply noise (so 0s also have some noise)
    mask: np.ndarray = np.zeros(shape=(i, j))
    for ki in range(k):
        h: int = base_h + tail_h if k == ki else base_h
        w: int = base_w + tail_w if k == ki else base_w
        h_start: int = max(0, ki * base_h)
        h_end: int = min(i, h_start + base_h + overlap_h)
        w_start = max(0, ki * base_w)
        w_end: int = min(j, w_start + base_w + overlap_w)
        mask[h_start:h_end, w_start:w_end] = 1.0
    block_mat: np.ndarray = np.random.uniform(
        low=0.0, high=1.0, size=(i, j)) * mask
    # Apply noise from normal distribution
    block_mat = block_mat + np.absolute(
        np.random.normal(loc=0.0, scale=0.1, size=(i, j)))
    # Convert to DataFrame and add some proper naming for rows/cols
    return pd.DataFrame(
        block_mat, 
        index=[f'feat_{i_lab}' for i_lab in range(i)],
        columns=[f'samp_{j_lab}' for j_lab in range(j)])


@pytest.fixture
def small_decomposition(
    small_overlap_blocks,
    scope="session"
    ) -> Decomposition:
    """A single decomposition of the small_overlap_blocks dataset."""
    return decompose(NMFParameters(
        x=small_overlap_blocks,
        rank=4,
        seed=4298
    ))


@pytest.fixture
def small_rank_selection(
    small_overlap_blocks,
    scope="session"
    ) -> Dict[int, List[BicvResult]]:
    """Rank selection carried out on a small matrix with overlapping
    block diagonal structure."""
    return rank_selection(
        x=small_overlap_blocks,
        ranks=list(range(2, 7)),
        shuffles=5,
        seed=4298,
        progress_bar=False
    ) 


@pytest.fixture
def small_bicv_result(
    small_overlap_blocks,
    scope="session"
    ) -> BicvResult:
    """BiCV run on a single shuffled matrix."""
    df: pd.DataFrame = small_overlap_blocks
    shuffles: List[BicvSplit] = BicvSplit.from_matrix(
        df=df, n=1, random_state=4298)

    return bicv(x=shuffles[0], rank=4, seed=99, keep_mats=False)


@pytest.fixture
def small_decompositions_random(
    small_overlap_blocks,
    scope="session"
    ) -> Dict[int, List[Decomposition]]:
    """Get 'best' decompositions for a small dataset from random 
    initialisations."""
    res = decompositions(
        x=small_overlap_blocks,
        random_starts=5,
        ranks=[3, 4, 5],
        top_n=3,
        top_criteria="cosine_similarity"
    )
    return res


@pytest.fixture
def small_decompositions_deterministic(
    small_overlap_blocks,
    scope="session"
    ) -> Dict[int, List[Decomposition]]:
    """Get single decomposittion for a small dataset from a deterministic
    initialisation."""
    res = decompositions(
        x=small_overlap_blocks,
        random_starts=5,
        ranks=[3, 4, 5],
        top_n=3,
        init="nndsvd",
        top_criteria="cosine_similarity"
    )
    return res


def test_bicv_split(small_overlap_blocks):
    """Shuffling and splitting working as intended?"""
    df: pd.DataFrame = small_overlap_blocks
    shuffles: List[BicvSplit] = BicvSplit.from_matrix(
        df=df, n=2, random_state=4298
    )

    # Overall number of entries in matrix is consistent
    shuf_len: int = shuffles[0].size
    assert shuf_len == df.size, \
        "Shuffled and split matrix different size to input"

    # Number of columns and rows consistent
    shuf_cols: int = shuffles[0].shape[1]
    shuf_rows: int = shuffles[0].shape[0]
    assert shuf_rows == df.shape[0], "Split and input shapes inconsistent"
    assert shuf_cols == df.shape[1], "Split and input shapes inconsistent"

    # Sum of entries consistent
    entry_sum: float = shuffles[0].x.sum().sum()
    assert np.isclose(entry_sum, df.sum().sum()), \
        "Sum of values in split and input are inconsistent"

    # Sorted list of entries as consistent
    input_vals: np.ndarray = np.sort(df.values.ravel())
    shuffled_vals: np.ndarray = np.sort(shuffles[0].x.values.ravel())
    assert all(np.isclose(input_vals, shuffled_vals))

    # Check we did actually shuffle the indices
    assert not all(shuffles[0].x.index == df.index), \
        "Index not shuffled"
    assert not all(shuffles[0].x.columns == df.columns), \
        "Columns not shuffled"

    # Get folds
    shuffles[0].fold(0)


def test_bicv_folds(small_overlap_blocks):
    """Are folds be made from submatrices correctly?"""
    df: pd.DataFrame = small_overlap_blocks
    shuffle: BicvSplit = BicvSplit.from_matrix(
        df=df, n=1, random_state=4298
    )[0]
    folds: List[BicvFold] = [shuffle.fold(i) for i in range(9)]

    def fold_sum(fold) -> float:
        return sum(x.sum().sum() for x in fold)

    # Check each fold has expected naming etc.
    for fold in folds:
        # No values should be NA
        for key, val in fold._asdict().items():
            assert not val.isna().any().any(), \
                f"NAs introduced in matrix {key} during fold generation"
        # B and D should share column names and order
        assert all(fold.B.columns == fold.D.columns), \
            "Columns not matched between B and D"
        # A and B should share index names and order
        assert all(fold.A.index == fold.B.index), \
            "Index not matched between A and B"
        # A and C should share column names and order
        assert all(fold.A.columns == fold.C.columns), \
            "Columns not matched between A and C"
        # C and D should share index names and order
        assert all(fold.C.index == fold.D.index), \
            "Columns not matched between C and D"

        # Check sum equal to input
        assert np.isclose(fold_sum(fold), df.sum().sum()), \
            "Sum of values in fold not equal to sum of input matrix"

    # Check no folds are identical
    for a, b in itertools.combinations(folds, 2):
        assert not np.array_equal(a, b), \
            "Two folds were equal"


def test_bicv_split_io(
    small_overlap_blocks,
    tmp_path: pathlib.Path):
    """Test BicvSplit I/O"""

    df: pd.DataFrame = small_overlap_blocks
    shuffles: List[BicvSplit] = BicvSplit.from_matrix(
        df=df, n=2, random_state=4298
    )

    # Basic operations
    # Save single shuffle
    shuffles[0].save_npz(tmp_path)
    assert tmp_path.with_name("shuffle_0.npz"), \
        "Saving a single shuffle failed"
    # Error on unforced overwrite
    with pytest.raises(FileExistsError):
        shuffles[0].save_npz(tmp_path)
    # Output all with force
    BicvSplit.save_all_npz(shuffles, tmp_path, force=True)
    # Count number of output files
    num_written: int = len(list(tmp_path.glob("shuffle_*.npz")))
    assert num_written == len(shuffles), \
        "Incorrect number of files written by save_all_npz"

    # Load single
    shuffle_0 = BicvSplit.load_npz(tmp_path / "shuffle_0.npz")
    # Check these are equivalent
    assert all(np.isclose(shuffle_0.x.values, shuffles[0].x.values).ravel()), \
        "Loaded object is not equivalent to source object"
    # Load multiple
    shuffles_loaded: List[BicvSplit] = BicvSplit.load_all_npz(tmp_path)
    # Should be equal to shuffles
    assert all(
        np.array_equal(a.x.values, b.x.values)
        for a, b in zip(shuffles, shuffles_loaded)
    ), \
        "Loaded matrices not equivalent to those saved"


def test_bicv(small_bicv_result: BicvResult):
    """A single iterations of BiCV for a single rank."""

    res: BicvResult = small_bicv_result
    # TODO: Complete tests, only checks for execution without error currently
    assert res.a is None, \
        "A' matrix returned when should be None due to keep_mats."
    # Check not getting any invalid values in result series
    for name, val in res._asdict().items():
        if name in {'parameters', 'a'}:
            continue
        assert all(~np.isnan(val)), \
            f"NAs in {name}"


def test__cosine_similarity(small_decomposition: Decomposition):
    a: np.ndarray = np.array([1.0, 0.0, 2.0, 0.5])
    assert _cosine_similarity(a, a) == 1.0, \
        "Similarity between identical vectors must be 1.0"
    # Should ignore scale
    assert _cosine_similarity(a, a * 2) == 1.0, \
        "Not scale invariant"
    # Should be -1 for inverse vector
    assert _cosine_similarity(a, -a) == -1.0, \
        "Should be -1 for inverse"
    # Test in expected range on example decomposition object
    assert np.max(small_decomposition.cosine_similarity) <= 1.0, \
        "Value >1.0 in example decomposition"
    assert np.min(small_decomposition.cosine_similarity) >= -1.0, \
        "Value <1.0 in example descomposition"


def test_rank_selection(small_rank_selection: Dict[int, List[BicvResult]]):
    """Test output a small rank selection run. Rank selection is actually
    run in a fixture, so result can be cached and reused between tests."""
    # Same number of results for each rank
    res_num: List[int] = [len(x) for x in small_rank_selection.values()]
    assert all(x == res_num[0] for x in res_num[1:]), \
        ("Different numbers of results for some ranks."
        "Some runs probably failed.")
    # Check getting correct result types
    for res in itertools.chain.from_iterable(small_rank_selection.values()):
        assert isinstance(res, BicvResult)
    # No more checks here, test properties of BicvResult elsewhere


def test_to_series(small_bicv_result: BicvResult):
    res: BicvResult = small_bicv_result
    series: pd.Series = res.to_series()
    # Test getting the right type and no missing values
    assert isinstance(series, pd.Series), \
        "to_series does not return as Series"
    assert not any(series.isnull()), \
        "NaN values in series"


def test_results_to_table(small_rank_selection):
    # Join from a dict across ranks / alpha values
    res = small_rank_selection
    res_df: pd.DataFrame = BicvResult.results_to_table(res)
    assert isinstance(res_df, pd.DataFrame), \
        "results_to_table did not return DataFrame"
    # TODO: Move these parameters out somewhere rather than magic numbers
    assert res_df.shape[0] == (
        len(set(res_df['rank'])) * len(set(res_df['shuffle_num']))), \
            "Incorrect number of rows in results table"
    assert not res_df.isnull().any().any(), \
        "Null values in rank selection table"


def test_plot_rank_selection(
    tmp_path: pathlib.Path,
    small_rank_selection
    ):
    res: Dict[int, List[BicvResult]] = small_rank_selection
    res_df: pd.DataFrame = BicvResult.results_to_table(res)
    plt = plot_rank_selection(res, exclude=None, geom="box")
    pth = (tmp_path / "test_rank_sel.png")
    plt.save(pth)
    # Test that the file exists and isn't empty
    assert pth.exists(), "Plot file not created"
    assert pth.stat().st_size > 0, "Plot file is empty"


def test_decompose(small_decomposition):
    """Test that decompose runs, and that Decomposition properties are in 
    expected format."""
    d: Decomposition = small_decomposition
    assert isinstance(d, Decomposition), \
        "decompose returns incorrect type"
    # Test properties

    # Floats
    for n in {'cosine_similarity', 'rss', 'r_squared', 'beta_divergence',
        'l2_norm', 'sparsity_h', 'sparsity_w'}:
        assert isinstance(getattr(d, n), float), \
            f"{n} is not a float"

    # Float series
    for n in {'model_fit'}:
        assert isinstance(getattr(d, n), pd.Series), \
            f"{n} is not a Series"
        assert np.issubdtype(getattr(d, n).dtype, np.floating), \
            f'{n} is not float data type'
        assert not getattr(d, n).isnull().any(), \
            f"{n} contains null values"

    # Dataframe float types
    for n in {'w', 'h', 'wh'}:
        assert isinstance(getattr(d, n), pd.DataFrame), \
            f"{n} is not a DataFrame"
        assert not getattr(d, n).isnull().any().any(), \
            f"{n} contains null values"
        assert all(np.issubdtype(x, np.floating)
                    for x in getattr(d, n).dtypes), \
            f"{n} has non-float columns"
        
    # Dataframe and Series dimensions, including derived properties
    assert all([
        d.w.shape == tuple(reversed(d.h.shape)),
        d.wh.shape == (d.w.shape[0], d.h.shape[1]),
        d.model_fit.size == d.w.shape[0],
        len(d.colors) == d.w.shape[1],
        d.monodominant_samples().shape[0] == d.w.shape[0],
        d.primary_signature.shape[0] == d.w.shape[0],
        d.representative_signatures().shape[0] == d.w.shape[0]
        ]), \
        "Series, matrices and colors do not have matching shapes"

    # String lists
    for n in {'names', 'colors'}:
        assert all(isinstance(x, str) and len(x) > 0 for x in d.colors), \
            f"{n} not all non-empty strings"
    
    # Basic check on parameters
        assert isinstance(d.parameters, NMFParameters), \
            "Parameters incorrect type"


def test_decompositions(
    small_decompositions_random,
    small_decompositions_deterministic
    ):
    # Check we get the same number of decompositions for each rank, incase some
    # are silenty failing
    cnt_decomps: Iterable[int] = list(
        len(x) for x in small_decompositions_random.values())
    assert all(x == cnt_decomps[0] for x in cnt_decomps), \
        ("Different numbers of decompositions for each rank, some "
         "decompositions may be silently failing.")
    cnt_decomps: Iterable[int] = list(
        len(x) for x in small_decompositions_deterministic.values())
    assert all(x == cnt_decomps[0] for x in cnt_decomps), \
        ("Different numbers of decompositions for each rank, some "
         "decompositions may be silently failing.")
    # Only a single decomposition for deterministic initialisation
    assert len(next(iter(small_decompositions_deterministic.values()))) == 1, \
        "Deterministic initialisation returned more than one decomposition"

    # The calls to generate decompositions should put them in order of
    # descending cosine_similarity.
    cos_sim: List[float] = [
        x.cosine_similarity for x in
        next(iter(small_decompositions_random.values()))]
    assert np.array_equal(np.array(cos_sim),
                          np.array(sorted(cos_sim, reverse=True))
                          ), \
        "Results not sorted by criteria (descending cosine similarity)."
    
    # Check all items are Decomposition type
    assert all(
        isinstance(x, Decomposition) for x in
        itertools.chain.from_iterable(small_decompositions_random.values())
        ), \
        "Random initialisation returned non-Decomposition objects"
    assert all(
        isinstance(x, Decomposition) for x in
        itertools.chain.from_iterable(
            small_decompositions_deterministic.values()
            )
        ), \
        "Random initialisation returned non-Decomposition objects"


def test_scaled(
    small_decomposition
    ):
    # TSS scaling
    h: pd.DataFrame = small_decomposition.scaled('h', by='sample')
    assert np.allclose(h.sum(), 1.0), \
        "Sample-scaled H does not sum to 1."
    # Should sample scale by default
    h = small_decomposition.scaled('h')
    assert np.allclose(h.sum(), 1.0), \
        "Default H scaling does not have sample sum of 1."
    
    



def test_primary_signature():
    matplotlib.pyplot.switch_backend("Agg")
    x: pd.DataFrame = pd.read_csv("/Users/pez23lof/Documents/cellgen/gut_cell_atlas/gca_cell_tables/count.tsv", sep="\t", index_col=0)
    res = decompose(NMFParameters(
        x=x,
        rank=5
    ))
    # _ = res.primary_signature
    # _ = res.representative_signatures()
    # _ = res.monodominant_samples()
    grp = np.random.choice(['a', 'b'], size=x.shape[1])
    plot = res.plot_relative_weight(group=grp)
    # plot = res.plot_pcoa()
    plot.save("/Users/pez23lof/test.png", height=8, width=8)