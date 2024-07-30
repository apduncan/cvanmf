"""Tests for denovo ES generation."""
import itertools
import logging
import pathlib
import re
from typing import List, Dict, Iterable, Tuple, Set, Optional

import matplotlib.pyplot
import numpy as np
import pandas as pd
import plotnine
import pytest
from click.testing import CliRunner

import cvanmf.data.utils
from cvanmf import models
from cvanmf.denovo import BicvSplit, BicvFold, bicv, _cosine_similarity, \
    rank_selection, BicvResult, plot_rank_selection, decompose, NMFParameters, \
    decompositions, Decomposition, cli_rank_selection, regu_selection, \
    plot_regu_selection, cli_regu_selection, suggest_rank, _cbar, \
    _cophenetic_correlation, cophenetic_correlation, _dispersion, dispersion, \
    plot_stability_rank_selection
from cvanmf.reapply import match_identical


# Deal with matplotlib backend
@pytest.fixture(scope="module", autouse=True)
def pyplot_backend():
    matplotlib.pyplot.switch_backend("Agg")


@pytest.fixture
def small_overlap_blocks(scope="session") -> pd.DataFrame:
    """Small overlapping block diagonal matrix with k=4, for use in testing
    de-novo methods."""
    return cvanmf.data.utils.synthetic_blocks(100, 100, 0.25, 3,
                                              scale_lognormal_params=True).data


@pytest.fixture
def small_decomposition(
        small_overlap_blocks,
        scope="session"
) -> Decomposition:
    """A single decomposition of the small_overlap_blocks dataset."""
    return decompose(NMFParameters(
        x=small_overlap_blocks,
        rank=3,
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
def small_regu_selection(
        small_overlap_blocks,
        scope="sesson"
) -> Tuple[float, Dict[float, List[BicvResult]]]:
    return regu_selection(
        x=small_overlap_blocks,
        rank=3,
        shuffles=5,
        seed=4928,
        progress_bar=False
    )


@pytest.fixture
def small_bicv_result(
        small_overlap_blocks,
        scope="session"
) -> BicvResult:
    """BiCV run on a single shuffled matrix."""
    df: pd.DataFrame = small_overlap_blocks
    shuffles: Iterable[BicvSplit] = BicvSplit.from_matrix(
        df=df, n=1, random_state=4298)

    return bicv(x=next(shuffles), rank=4, seed=99, keep_mats=False)


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
        top_criteria="cosine_similarity",
        progress_bar=False
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
        init="nndsvda",
        top_criteria="cosine_similarity",
        progress_bar=False
    )
    return res


@pytest.fixture()
def small_decomposition_metadata_cd(
        small_decomposition
) -> pd.DataFrame:
    """Random metadata with continuous and discrete columns"""
    rnd_cat: pd.DataFrame = pd.Series(
        np.random.choice(["A", "B"], size=small_decomposition.h.shape[1]),
        index=small_decomposition.h.columns
    ).to_frame(name="rand_cat")
    rnd_cat['rand_cat_too'] = (
        small_decomposition
        .scaled("h")
        .idxmax().astype("str")
    )
    rnd_cat['rand_cat_na'] = rnd_cat['rand_cat_too'].replace("S3", np.nan)
    rnd_cat['rand_cat_categorical'] = pd.Categorical(
        rnd_cat['rand_cat_too'],
        categories=rnd_cat['rand_cat_too'].unique()
    )
    rnd_cat['rand_cont'] = np.random.uniform(size=rnd_cat.shape[0])
    return rnd_cat


def is_decomposition_close(a: Decomposition, b: Decomposition) -> bool:
    """Test whether two decomposition are equal (within tolerance). Used
    for checking equivalence between source and loaded versions."""
    # Check equivalence of w, h, and x matrices, as all other properties are
    # calculated from these
    __tracebackhide__ = True
    for prop in ['w', 'h', 'x']:
        if not np.allclose(getattr(a, prop), getattr(b, prop)):
            pytest.fail(f'{prop} not equivalent between decompositions')
    # Parameters object
    a_param: Dict = a.parameters._asdict()
    b_param: Dict = b.parameters._asdict()
    a_param.pop("x")
    b_param.pop("x")
    if a_param != b_param:
        pytest.fail("Parameters tuple not equivalent")
    # Color / name settings
    if a.names != b.names:
        pytest.fail("Names not equivalent")
    if a.colors != b.colors:
        pytest.fail("Colors not equivalent")


def are_decompositions_close(a: Dict[int, List[Decomposition]],
                             b: Dict[int, List[Decomposition]]):
    """Are multiple decompositions equivalent?"""

    # Flatten each and compare elementwise
    paired = list(zip(
        itertools.chain.from_iterable(a.values()),
        itertools.chain.from_iterable(b.values())
    ))
    for a_i, b_i in paired:
        is_decomposition_close(a_i, b_i)


def test_bicv_split(small_overlap_blocks):
    """Shuffling and splitting working as intended?"""
    df: pd.DataFrame = small_overlap_blocks
    shuffles: List[BicvSplit] = list(BicvSplit.from_matrix(
        df=df, n=2, random_state=4298
    ))

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

    # Attempting to initialise with incorrect number of submatrices should
    # raise an exception
    with pytest.raises(ValueError):
        mx8 = list(itertools.chain.from_iterable(shuffles[0].mx))[:-1]
        _ = BicvSplit(mx8)

    # Getting a row without joining
    row_list: List[pd.DataFrame] = shuffles[0].row(0, join=False)
    col_list: List[pd.DataFrame] = shuffles[1].col(0, join=False)
    assert isinstance(row_list, list), \
        "Row not returned as list when join=False"
    assert isinstance(col_list, list), \
        "Column not returned as list when join=False"

    # Out of range fold
    with pytest.raises(IndexError):
        shuffles[0].fold(9)

    folds: List[BicvFold] = shuffles[0].folds
    assert len(folds) == 9, "Should have 9 folds"
    assert all([isinstance(x, BicvFold) for x in folds]), \
        "folds property returns incorrect type"


def test_bicv_folds(small_overlap_blocks):
    """Are folds be made from submatrices correctly?"""
    df: pd.DataFrame = small_overlap_blocks
    shuffle: BicvSplit = list(BicvSplit.from_matrix(
        df=df, n=1, random_state=4298
    ))[0]
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
    shuffles: List[BicvSplit] = list(BicvSplit.from_matrix(
        df=df, n=2, random_state=4298
    ))

    # Basic operations
    # Save single shuffle
    shuffles[0].save_npz(tmp_path)
    assert tmp_path.with_name("shuffle_0.npz"), \
        "Saving a single shuffle failed"
    # Error on unforced overwrite
    with pytest.raises(FileExistsError):
        shuffles[0].save_npz(tmp_path)
    with pytest.raises(ValueError):
        shuffles[0].i, i = None, shuffles[0].i
        shuffles[0].save_npz(tmp_path, force=True)
    shuffles[0].i = i
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
    shuffles_loaded: List[BicvSplit] = list(
        BicvSplit.load_all_npz(tmp_path, fix_i=True)
    )
    # Should be equal to shuffles
    assert all(
        np.array_equal(a.x.values, b.x.values)
        for a, b in zip(shuffles, shuffles_loaded)
    ), \
        "Loaded matrices not equivalent to those saved"

    # Non-unique values of i
    shuffles[0].i = shuffles[1].i
    # fix_i should renumber
    BicvSplit.save_all_npz(shuffles, tmp_path, force=True, fix_i=True)
    reloaded: List[BicvSplit] = list(BicvSplit.load_all_npz(tmp_path))
    unique_i: Set[Optional[int]] = set(x.i for x in reloaded)
    assert len(unique_i) == len(shuffles)


def test_bicv(small_bicv_result: BicvResult):
    """A single iterations of BiCV for a single rank."""

    res: BicvResult = small_bicv_result
    # TODO: Complete tests, only checks for execution without error currently
    assert res.a is None, \
        "A' matrix returned when should be None due to keep_mats."
    assert res.parameters.x is None, \
        "X matrix should not be returned when keep_mats=False"
    # Check not getting any invalid values in result series
    for name, val in res._asdict().items():
        if name in {'parameters', 'a', 'i'}:
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
    # There should be results for ranks 2 to 6
    rank_set: Set[int] = set(small_rank_selection.keys())
    assert rank_set == set(range(2, 7)), \
        "Missing results for some ranks which should be tested."
    # Same number of results for each rank
    res_num: List[int] = [len(x) for x in small_rank_selection.values()]
    assert all(x == res_num[0] for x in res_num[1:]), \
        ("Different numbers of results for some ranks."
         "Some runs probably failed.")
    # Check getting correct result types
    for res in itertools.chain.from_iterable(small_rank_selection.values()):
        assert isinstance(res, BicvResult)
    # No more checks here, test properties of BicvResult elsewhere


def test_regu_selection(
        small_regu_selection: Tuple[float, Dict[float, List[BicvResult]]]):
    """Test output of a small regularisation selection run. Regularisation
    selection is actually run in a fixture, so result can be shared between
    tests."""
    est, res = small_regu_selection

    # There should be results for the standard alpha values
    alpha_set: Set[int] = set(res.keys())
    alpha_expected: Set[float] = set(
        [0] + [2 ** i for i in range(-5, 2)]
    )
    assert len(alpha_set) == len(alpha_expected), \
        "Missing results for some alphas which should be tested."

    res_num: List[int] = [len(x) for x in res.values()]
    assert all(x == res_num[0] for x in res_num[1:]), \
        ("Different numbers of results for some ranks."
         "Some runs probably failed.")
    # Check getting correct result types
    for res in itertools.chain.from_iterable(res.values()):
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

    # Ensure elbow detection fails gracefully
    # Remove anything above correct rank to eliminate elbow
    res_no_elbow: Dict[int, List[BicvResult]] = {
        rank: res for rank, res in small_rank_selection.items()
        if rank < 4
    }
    plt = plot_rank_selection(res_no_elbow, exclude=None, geom="box")
    pth = (tmp_path / "test_rank_sel_no_elbow.png")
    plt.save(pth)
    # Test that the file exists and isn't empty
    assert pth.exists(), "Plot file not created"
    assert pth.stat().st_size > 0, "Plot file is empty"


def test_plot_regu_selection(
        tmp_path: pathlib.Path,
        small_regu_selection
):
    best_alpha, res = small_regu_selection
    plt = plot_regu_selection(small_regu_selection,
                              exclude=None, geom="box")
    pth = (tmp_path / "test_regu_sel.png")
    plt.save(pth)
    # Test that the file exists and isn't empty
    assert pth.exists(), "Plot file not created"
    assert pth.stat().st_size > 0, "Plot file is empty"


def test_plot_model_fit_point(
        tmp_path: pathlib.Path,
        small_decomposition
):
    matplotlib.pyplot.switch_backend("Agg")
    plt = small_decomposition.plot_modelfit_point(threshold=0.9, yrange=None)
    pth = (tmp_path / "test_rank_sel.png")
    plt.save(pth)
    # Test that the file exists and isn't empty
    assert pth.exists(), "Plot file not created"
    assert pth.stat().st_size > 0, "Plot file is empty"


def test_plot_relative_weight(
        tmp_path: pathlib.Path,
        small_decomposition
):
    matplotlib.pyplot.switch_backend("Agg")
    # Want to plot a category with it
    rnd_cat: pd.Series = pd.Series(
        np.random.choice(["A", "B"], size=small_decomposition.h.shape[1]),
        index=small_decomposition.h.columns
    )
    # Add a sample that is not in the decomposition
    rnd_cat.loc['dn3259nfn'] = 'C'
    # Remove one sample that should be there
    rnd_cat = rnd_cat.drop(index=[rnd_cat.index[0]])
    plt = small_decomposition.plot_relative_weight(
        group=rnd_cat,
        model_fit=True,
        heights=dict(ribbon=2, bar=2, nonsense=2),
        sample_label_size=3.0,
        legend_cols_h=2,
        legend_cols_v=2,
    )
    pth = (tmp_path / "test_rank_sel.png")
    plt.savefig(pth)
    # Test that the file exists and isn't empty
    assert pth.exists(), "Plot file not created"
    assert pth.stat().st_size > 0, "Plot file is empty"


def test_plot_feature_weight(
        tmp_path: pathlib.Path,
        small_decomposition
):
    matplotlib.pyplot.switch_backend("Agg")
    plt = small_decomposition.plot_feature_weight(threshold=0.02)
    pth = (tmp_path / "test_feature_weight.png")
    plt.save(pth)
    # Test that the file exists and isn't empty
    assert pth.exists(), "Plot file not created"
    assert pth.stat().st_size > 0, "Plot file is empty"


def test_plot_pcoa(
        tmp_path: pathlib.Path,
        small_decomposition
):
    matplotlib.pyplot.switch_backend("Agg")
    rnd_cat: pd.Series = pd.Series(
        np.random.choice(["A", "B"], size=small_decomposition.h.shape[1]),
        index=small_decomposition.h.columns
    )
    rnd_cat.name = "Random Category"
    plt = small_decomposition.plot_pcoa(
        color=rnd_cat,
        shape="signature",
        point_aes=dict(size=5, alpha=0.3),
        signature_arrows=True
    )
    pth = (tmp_path / "test_pcoa.png")
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
        d.model_fit.size == d.h.shape[1],
        len(d.colors) == d.h.shape[0],
        d.monodominant_samples().shape[0] == d.h.shape[1],
        d.primary_signature.shape[0] == d.h.shape[1],
        d.representative_signatures().shape == d.h.shape
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
    # Feature scaling H should default to sample scaling
    h_feat = small_decomposition.scaled('h', by='feature')
    assert np.allclose(h_feat, h), \
        "Feature scaling H did not default to sample scaling."
    # Signature scaling
    h_sig = small_decomposition.scaled('h', by='signature')
    assert np.allclose(h_sig.sum(axis=1), 1.0), \
        "Signature scaling H did not have signature sum of 1"

    w: pd.DataFrame = small_decomposition.scaled('w', by='feature')
    assert np.allclose(w.sum(axis=1), 1.0), \
        "Feature-scaled W does not sum to 1."
    # Should feature scale by default
    w = small_decomposition.scaled('w')
    assert np.allclose(w.sum(), 1.0), \
        "Default W scaling does not have signature sum of 1."
    # Sample scaling W should default to feature scaling
    w_feat = small_decomposition.scaled('w', by='sample')
    assert np.allclose(w_feat, w), \
        "Sample scaling W did not default to sample scaling."
    # Signature scaling
    w_sig = small_decomposition.scaled('w', by='signature')
    assert np.allclose(w_sig.sum(), 1.0), \
        "Signature scaling H did not have signature sum of 1"

    # Invalid values for by should error
    with pytest.raises(ValueError):
        # This is how I end up typing signature about half the time
        h_null = small_decomposition.scaled('h', by='sginatntuer')
    # Non H or W matrix should error
    with pytest.raises(ValueError):
        h_null = small_decomposition.scaled(
            pd.DataFrame(np.random.uniform(size=(20, 7)))
        )


def test_cli_rank_selection(
        small_overlap_blocks: pd.DataFrame,
        tmp_path: pathlib.Path
):
    """Run the CLI command for rank selection."""
    # Write our block data to the temp path
    small_overlap_blocks.to_csv(tmp_path / "input.tsv", sep="\t")
    runner: CliRunner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        result = runner.invoke(cli_rank_selection,
                               [
                                   "--input", tmp_path / "input.tsv",
                                   "-o", td,
                                   "-d", "\t",
                                   "--shuffles", "5",
                                   "--no-progress",
                                   "--seed", "4928",
                                   "-l", "3",
                                   "-u", "5",
                                   "--log_warning"
                               ]
                               )
    assert result.exit_code == 0, \
        "CLI rank selection did had non-zero exit code"
    td_path: pathlib.Path = pathlib.Path(td)
    for expected_file in ['rank_selection.tsv', 'rank_selection.pdf']:
        assert (td_path / expected_file).is_file(), \
            f"{expected_file} not created"


def test_cli_regu_selection(
        small_overlap_blocks: pd.DataFrame,
        tmp_path: pathlib.Path
):
    """Run the CLI command for rank selection."""
    # Write our block data to the temp path
    small_overlap_blocks.to_csv(tmp_path / "input.tsv", sep="\t")
    runner: CliRunner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        result = runner.invoke(cli_regu_selection,
                               [
                                   "--input", tmp_path / "input.tsv",
                                   "-o", td,
                                   "-d", "\t",
                                   "--shuffles", "5",
                                   "--no-progress",
                                   "--seed", "4928",
                                   "--rank", "3",
                                   "--log_warning"
                               ]
                               )
    assert result.exit_code == 0, \
        "CLI regu selection had non-zero exit code"
    td_path: pathlib.Path = pathlib.Path(td)
    for expected_file in ['regu_selection.tsv', 'regu_selection.pdf']:
        assert (td_path / expected_file).is_file(), \
            f"{expected_file} not created"


# TODO: Tests for plotting functions.
# Leaving for now as might rewrite to use a different plotting method, 
# patchworklib seems not to play so nicely with 3.12
def test_save(small_decomposition: Decomposition,
              tmp_path: pathlib.Path):
    """Ensure that decompositions and plots can be written to disk."""
    matplotlib.pyplot.switch_backend("Agg")
    odir: pathlib.Path = tmp_path / "test_save"
    small_decomposition.save(
        out_dir=odir
    )
    for expected_file in [
        "h.tsv", "h_scaled.tsv", "model_fit.tsv", "monodominant_samples.tsv",
        "plot_modelfit.pdf", "plot_pcoa.pdf", "plot_relative_weight.pdf",
        "primary_signature.tsv", "quality_series.tsv",
        "representative_signatures.tsv", "w.tsv", "w_scaled.tsv", "x.tsv"]:
        assert (odir / expected_file).is_file(), \
            f"Expected file {expected_file} not created"
        assert (odir / expected_file).stat().st_size > 0, \
            f"File {expected_file} is empty (st_size <= 0)"
    # TODO: Tests for symlinking properly etc.

    # Test suppressing plots
    odir_noplot: pathlib.Path = tmp_path / "test_save_noplot"
    small_decomposition.save(
        out_dir=odir_noplot,
        plots=False
    )
    for expected_file in [
        "h.tsv", "h_scaled.tsv", "model_fit.tsv", "monodominant_samples.tsv",
        "primary_signature.tsv", "quality_series.tsv",
        "representative_signatures.tsv", "w.tsv", "w_scaled.tsv", "x.tsv"]:
        assert (odir / expected_file).is_file(), \
            f"Expected file {expected_file} not created"
        assert (odir / expected_file).stat().st_size > 0, \
            f"File {expected_file} is empty (st_size <= 0)"

    # Test only selecting specific plots
    # Test suppressing plots
    odir_specific: pathlib.Path = tmp_path / "test_save_specific"
    small_decomposition.save(
        out_dir=odir_specific,
        plots=["feature_weight", "pcoa"]
    )
    for expected_file in ["plot_feature_weight.pdf", "plot_pcoa.pdf"]:
        assert (odir / expected_file).is_file(), \
            f"Expected file {expected_file} not created"
        assert (odir / expected_file).stat().st_size > 0, \
            f"File {expected_file} is empty (st_size <= 0)"
    # Determine that there are no undesired pdfs
    for f in odir_specific.glob("*.pdf"):
        assert str(f.name) in [
            "plot_feature_weight.pdf", "plot_pcoa.pdf"], \
            f"Unexpected plot {f} produced"


def test_quality_series(small_decomposition):
    """Are the properties of a decomposition being produced as a series
    correctly?"""
    s: pd.Series = small_decomposition.quality_series
    assert s.size == 7, "Some properties not in series"
    assert not any(s.isna()), "Some properties are NaN/NA"
    assert not any(s.isnull()), "Some properties are null"


def test_save_load_decompositions(small_decompositions_random,
                                  tmp_path: pathlib.Path):
    # Trim small decompositions so we're using a little less time
    # Keep two decompositions for two ranks
    smaller_decomps: Dict[int, List[Decomposition]] = {
        i: [x[:25, :, :] for x in small_decompositions_random[i][0:2]] for i in
        small_decompositions_random.keys()
    }
    Decomposition.save_decompositions(
        smaller_decomps,
        output_dir=tmp_path / "test_save_decompositions"
    )
    loaded: Dict[int, List[Decomposition]] = Decomposition.load_decompositions(
        tmp_path / "test_save_decompositions"
    )
    are_decompositions_close(smaller_decomps, loaded)

    # Test that load is sharing a reference to input data X
    keys: List[int] = list(loaded.keys())
    assert loaded[keys[0]][0].x is loaded[keys[1]][0].x, \
        "Not reusing X matrix when loading multiple decompositions"

    # Test compressed output
    Decomposition.save_decompositions(
        smaller_decomps,
        output_dir=tmp_path / "test_save_decompositions_compressed",
        compress=True
    )
    loaded: Dict[int, List[Decomposition]] = Decomposition.load_decompositions(
        tmp_path / "test_save_decompositions_compressed"
    )
    are_decompositions_close(smaller_decomps, loaded)


def test_load(small_decomposition: Decomposition,
              tmp_path: pathlib.Path):
    """Can a single decomposition be loaded successfully?"""
    small_decomposition.save(tmp_path / "saved_decomp")
    loaded: Decomposition = Decomposition.load(tmp_path / "saved_decomp")
    is_decomposition_close(small_decomposition, loaded)

    # We should also be able to provide a string path for saving, and should be able
    # to load from compressed form
    path_str: pathlib.Path = tmp_path / "save_string"
    path_targz: pathlib.Path = path_str.with_suffix(".tar.gz")
    small_decomposition.save(str(path_str), compress=True)
    assert path_targz.exists(), "Compressed decomposition output not created."
    loaded_comp: Decomposition = Decomposition.load(str(path_targz))
    is_decomposition_close(small_decomposition, loaded_comp)


def test_colors(small_overlap_blocks):
    """Check default colours get set correctly."""

    # Was an issue with colours for rank 7 decomposition being garbled.
    matplotlib.pyplot.switch_backend("Agg")
    k7: Decomposition = decompositions(
        small_overlap_blocks, random_starts=1, ranks=[7])[7][0]
    hex_reg: re.Pattern = re.compile(r'^#[A-Fa-f0-9]{6}$')
    assert len(k7.colors) == 7, "Incorrect number of colors set"
    # Each colour should be a hex code
    assert all(hex_reg.match(x) is not None for x in k7.colors), \
        "Expected all default colors to be hex codes (i.e. #000fff)"


def test_slicing(small_decomposition):
    """Do all the different slicing methods work?"""

    matplotlib.pyplot.switch_backend("Agg")
    slcd: Decomposition = small_decomposition[:, :, ['S1', 'S2']]
    slcd: Decomposition = small_decomposition[[0, 4, 6, 7]]
    slcd = small_decomposition[:4, 6:11, [0, 2]]
    assert (slcd.h.shape == (2, 4) and slcd.w.shape == (5, 2)
            and slcd.x.shape == (5, 4)), \
        "Sliced Decomposition has incorrect dimensions"
    with pytest.raises(IndexError):
        slcd = small_decomposition[[0, 4], ["A non existent sample"]]


def test_reapply(small_decomposition):
    """Can new data be transformed using the model?"""
    # TODO: Improve this test, only checks it does not crash currently
    new_decomp: Decomposition = small_decomposition.reapply(
        y=small_decomposition.x,
        input_validation=lambda x, **kwargs: x,
        feature_match=match_identical,
        family_rollup=True
    )
    # This should generate similar h matrices
    # Should they be more similar than this?
    assert np.allclose(small_decomposition.h, new_decomp.h,
                       atol=0.025), \
        "Signature weights not equal for identical data."

    # Test with defaults
    new_decomp: Decomposition = small_decomposition.reapply(
        y=small_decomposition.x
    )
    # This should generate similar h matrices
    # Should they be more similar than this?
    assert np.allclose(small_decomposition.h, new_decomp.h,
                       atol=0.025), \
        "Signature weights not equal for identical data."


def test_nmf_parameters(small_overlap_blocks):
    nmf_params: NMFParameters = NMFParameters(small_overlap_blocks, 2)
    log_str: str = nmf_params.log_str
    assert len(log_str) > 0, "Parameter log string should not be empty"


def test_univariate_tests(small_decomposition):
    rnd_cat: pd.DataFrame = pd.Series(
        np.random.choice(["A", "B"], size=small_decomposition.h.shape[1]),
        index=small_decomposition.h.columns
    ).to_frame(name="rand_cat")
    rnd_cat['rand_cat_too'] = (
        small_decomposition
        .scaled("h")
        .idxmax().astype("str")
    )
    rnd_cat['rand_cat_na'] = rnd_cat['rand_cat_too'].replace("S3", np.nan)
    small_decomposition.univariate_tests(metadata=rnd_cat)


def test_plot_metadata(
        small_decomposition,
        small_decomposition_metadata_cd,
        tmp_path: pathlib.Path
):
    disc_plt, cont_plt = small_decomposition.plot_metadata(
        metadata=small_decomposition_metadata_cd
    )
    disc_pth = (tmp_path / "test_disc.png")
    disc_plt.save(disc_pth)
    # Test that the file exists and isn't empty
    assert disc_pth.exists(), "Plot file not created"
    assert disc_pth.stat().st_size > 0, "Plot file is empty"
    cont_pth = (tmp_path / "test_cont.png")
    cont_plt.save(cont_pth)
    # Test that the file exists and isn't empty
    assert cont_pth.exists(), "Plot file not created"
    assert cont_pth.stat().st_size > 0, "Plot file is empty"


def test_name_signatures_by_weight(
        small_decomposition
):
    small_decomposition.name_signatures_by_weight(
        max_char_length=20, max_num_features=1
    )


def test_suggest_rank(
        small_rank_selection
):
    res: Dict[str, float] = suggest_rank(small_rank_selection)
    foo = 'ar'


def test_consensus_matrix(
        small_decomposition
):
    # Slice to there is an uneven number of features and samples
    sliced: Decomposition = small_decomposition[:75, :, :]
    c = sliced.consensus_matrix()
    assert c.shape == (sliced.h.shape[1],
                       sliced.h.shape[1]), "Incorrect shape on H (default)"
    cw = sliced.consensus_matrix('w')
    assert cw.shape == (sliced.w.shape[0],
                        sliced.w.shape[0]), "Incorrect shape on W"
    # Sum must be > 0, must be some elements in the same cluster
    assert cw.sum() > 0, "No True values in consensus matrix"


def test__cbar(
        small_decompositions_random
):
    one_rank: List[Decomposition] = (
        list(small_decompositions_random.values())[0])
    cbar = _cbar(x.consensus_matrix() for x in one_rank)
    assert cbar.shape == (one_rank[0].h.shape[1],
                          one_rank[0].h.shape[1]), \
        "Incorrect shape for cbar on H"


def test__cophenetic_correlation(
        small_decompositions_random
):
    one_rank: List[Decomposition] = (
        list(small_decompositions_random.values())[0])
    cbar = _cbar(x.consensus_matrix() for x in one_rank)


def test_cophenetic_correlation(small_decompositions_random):
    res = cophenetic_correlation(small_decompositions_random)


def test__dispersion(small_decompositions_random):
    # All ones or zeros should give 1 (very consistent)
    res_ones = _dispersion(np.ones(shape=(10, 10)))
    assert res_ones == 1.0, "Dispersion for all 1s should be 1."
    res_zeroes = _dispersion(np.zeros(shape=(10, 10)))
    assert res_zeroes == 1.0, "Dispersion for all 0s should 1"
    res_rand = _dispersion(np.random.uniform(size=(10, 10)))
    assert res_rand < 0.9, "Dispersion for random should be low."
    one_rank: List[Decomposition] = (
        list(small_decompositions_random.values())[0])
    cbar = _cbar(x.consensus_matrix() for x in one_rank)
    res = _dispersion(cbar)


def test_dispersion(small_decompositions_random):
    res = dispersion(small_decompositions_random)
    ff = 66


def test_plot_stability_rank_selection(
        small_decompositions_random,
        tmp_path: pathlib.Path,
):
    plt: plotnine.ggplot = plot_stability_rank_selection(
        small_decompositions_random)
    plt.save(tmp_path / "plt_stability.png")
    ff = 66
