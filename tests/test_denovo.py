"""Tests for denovo ES generation."""
import itertools
import math
import os
import pathlib
import random
import re
from typing import List, Dict

from click.testing import CliRunner
import matplotlib.pyplot
import numpy as np
import pandas as pd
import plotnine
import pytest

from cvanmf import models
from cvanmf.denovo import BicvSplit, BicvFold, bicv, _cosine_similarity, \
    rank_selection, BicvResult, plot_rank_selection, decompose, NMFParameters, \
    decompositions, Decomposition, cli_rank_selection


# Deal with matplotlib backend
@pytest.fixture(scope="module", autouse=True)
def pyplot_backend():
    matplotlib.pyplot.switch_backend("Agg")


@pytest.fixture
def small_overlap_blocks(scope="session") -> pd.DataFrame:
    """Small overlapping block diagonal matrix with k=4, for use in testing
    de-novo methods."""
    return models.synthetic_data(100, 100, 0.25, 3)


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
    plt = small_decomposition.plot_relative_weight(group=rnd_cat, model_fit=True)
    pth = (tmp_path / "test_rank_sel.png")
    plt.savefig(pth)
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
                                   "--log_debug"
                               ]
                               )
    assert result.exit_code == 0, \
        "CLI rank selection did had non-zero exit code"
    td_path: pathlib.Path = pathlib.Path(td)
    for expected_file in ['rank_selection.tsv', 'rank_selection.pdf']:
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
        i: small_decompositions_random[i][0:2] for i in
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