import pathlib
from importlib.resources import files
from pathlib import Path
from typing import Set, List, Dict, Iterable

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from cvanmf.denovo import Decomposition, NMFParameters
from cvanmf.reapply import cli, to_relative
from cvanmf import models
from cvanmf.reapply import (GenusMapping, EnteroException, validate_table,
                            match_genera, nmf_transform, transform_table,
                            ReapplyResult, model_fit)


def test_genus_mapping() -> None:
    """Test GenusMapping class which handles match genera in abundance table and
    model matrices."""

    source_taxa: Set[str] = {"A", "B", "C", "D"}
    target_taxa: Set[str] = {"B", "C", "D", "E", "X"}
    mapping: GenusMapping = GenusMapping(target_taxa=target_taxa,
                                         source_taxa=source_taxa)

    # Add a mapping
    mapping.add(genus_from="B", genus_to="B")
    assert mapping.mapping["B"] == ["B"], "Mapping not added succesfully"

    with pytest.raises(EnteroException):
        # Add a mapping with a non-existent target
        mapping.add("B", "Missing")
        # Add a mapping with a non-existent source
        mapping.add("Missing", "B")

    # Add a second mapping for a taxon
    mapping.add(genus_from="B", genus_to="C")
    assert mapping.mapping["B"] == ["B", "C"], "Second mapping not added"
    assert len(mapping.conflicts) == 1, "Conflict not detected"
    assert mapping.conflicts[0] == ("B", ["B", "C"]), "Conflict not reported"

    # Check taxa with no mapping are identified correctly
    assert set(mapping.missing()) == {"A", "C", "D"}

    # Get mapping as a DataFrame
    df: pd.DataFrame = mapping.to_df()
    assert df.shape == (5, 2), "DataFrame of mapping has wrong dimensions"
    assert set(df['input_genus']) == source_taxa, ("Some source taxa missing in"
                                                   "dataframe of mapping")
    assert set(df['es_genus']) == {"", "B", "C"}, "Unexpected target taxa"

    # Transform an abundance matrix
    mapping.add("C", "B")
    mat: pd.DataFrame = pd.DataFrame.from_dict(
        dict(A=[1, 0], B=[1, 1], C=[0, 1], D=[1, 1], N=[2, 2]),
        columns=["smpl1", "smpl2"],
        orient="index"
    )
    trans_mat: pd.DataFrame = mapping.transform_abundance(mat)
    assert isinstance(trans_mat, pd.DataFrame), ("transform_abundance did not "
                                                 "return a DataFrame")
    # B maps to B, C maps to B also. So B should be B + C.
    pd.testing.assert_series_equal(
        trans_mat.loc["B"],
        mat.loc[["B", "C"]].sum(),
        check_names=False
    ), "transform_abundance sums are incorrect"

    # Transform a W matrix
    w: pd.DataFrame = pd.DataFrame.from_dict(
        dict(A=[1, 0], B=[1, 1], C=[0, 1], D=[1, 1]),
        columns=["es1", "es2"],
        orient="index"
    )
    trans_w: pd.DataFrame = mapping.transform_w(w, mat)
    assert isinstance(trans_mat, pd.DataFrame), (
        "transform_w did not return a DataFrame")
    # Check that the taxa N which is in the abundance but not W matrix has
    # been added and sums to 0
    assert "N" in trans_w.index, "Missing taxon N not added to transformed W"
    assert trans_w.loc["N"].sum() == 0, "Missing taxon does not have 0 weight"


def test_validate_table() -> None:
    """Test abunance table validations"""

    # Test for only one sample - maybe we should allow this though?
    malformed: pd.DataFrame = pd.DataFrame(
        [[0], [2]], index=["a", "b"]
    )
    with pytest.raises(EnteroException):
        validate_table(malformed, logger=lambda x: None)

    # Test for no sample names
    no_colnames: pd.DataFrame = pd.DataFrame(
        [[0, 1, 2], [3, 4, 5]], index=["a", "b"]
    )
    with pytest.raises(EnteroException):
        validate_table(no_colnames, logger=lambda x: None)

    # Run on some example data
    # We'll modify this dataset to trigger as many validation steps as possible
    # - transposed
    transposed: pd.DataFrame = models.example_abundance().T
    # - rank identifiers (d__;p__ etc)
    transposed.columns = [f'd__{x}' for x in transposed.columns]
    # - numeric feature
    transposed["-1"] = 0
    # - completely unknown taxa
    transposed["?;?"] = 0
    # - sample with 0 taxa observed
    transposed.loc["empty", :] = 0
    # - legitimate(ish) taxon with 0 observations
    transposed["Eukaryota;Ilex_aquifolium"] = 0
    log: List[str] = []
    tidied: pd.DataFrame = validate_table(transposed,
                                          logger=lambda x: log.append(x))
    # Has this been transposed? Should have more samples than taxa for the
    # example data
    assert tidied.shape[1] > tidied.shape[0], "Not succesfully transposed"
    assert tidied.shape[0] == 586, "Tidied data should have 5 columns (1 per ES)"
    # Have ranked identifiers been removed?
    assert sum('d__' in x for x in tidied.index) == 0, \
        "Rank identifiers not removed"
    # Have numeric features been removed?
    assert sum(str(x).isnumeric() for x in tidied.index) == 0, \
        "Numeric taxon IDs not removed"
    # Has the unknown taxon been removed?
    assert "?;?" not in tidied.index, "Unknown taxon not removed"
    # Have taxa with no observations been removed?
    assert not any(tidied.sum(axis=1) == 0), \
        "Taxa with 0 observations not removed"

    # Check error thrown on duplicates
    dup_cols: List[str] = list(transposed.columns)
    dup_cols[0] = dup_cols[1]
    transposed.columns = dup_cols
    with pytest.raises(EnteroException):
        validate_table(transposed, logger=lambda x: log.append(x))


def test_match_genera() -> None:
    """Test function for matching genera in an abundance table and model. This
    is the bulk of work in reapplying Eneterosignatures."""

    # Make test data to work with
    abd: pd.DataFrame = models.example_abundance()
    w: pd.DataFrame = models.five_es()
    # Make an arbitrary hard mapping
    mapping: Dict[str, str] = {
        abd.index[0]: w.index[-1]
    }
    log: List[str] = []

    t_abd, t_w, map = match_genera(es_w=w, abd_tbl=abd, hard_mapping=mapping,
                                   family_rollup=True, logger=log.append)
    # Basic sanity check of results
    assert t_w.shape[1] == w.shape[1], "W matrix column counts do not match"
    assert t_abd.shape[1] == abd.shape[1], \
        "Abundance column counts do not match"
    # Only columns which exist in the original W matrix should have any weight
    has_weight: Iterable[str] = t_w[t_w.sum(axis=1) > 0].index
    assert all(x in w.index for x in has_weight), \
        "Taxa not in original model have been assigned weight"
    # Check hard mapping respected
    assert abd.index[0] not in t_abd.index, \
        "User specified hard mapping not respected"
    assert w.index[-1] in t_abd.index, \
        "User specified hard mapping not respected"

    # Test just that no errors with different parameters
    # Without mapping
    t_w, t_abd, map = match_genera(es_w=w, abd_tbl=abd, hard_mapping=None,
                                   family_rollup=True, logger=log.append)
    # No rollup
    t_w, t_abd, map = match_genera(es_w=w, abd_tbl=abd, hard_mapping=mapping,
                                   family_rollup=False, logger=log.append)


def test_model_fit() -> None:
    """Test functions for calculating model fit."""

    # Check simplest case - two identical matrices should produce all 1s
    w: pd.DataFrame = models.five_es()
    abd: pd.DataFrame = models.example_abundance()
    res: Decomposition = transform_table(abundance=abd, rollup=True,
                                         model_w=w)
    mf: pd.Series = res.model_fit
    # Check for stable mean model fit between versions
    assert round(mf.mean(), 3) == round(0.636286, 3), \
        "Change in mean model fit"

    # Model fit between identical matrices should be 1.0
    clone_decomp: Decomposition = Decomposition(
        parameters=NMFParameters(
            **(res.parameters._asdict() | dict(x=res.wh))
        ),
        h=res.h,
        w=res.w
    )
    # Sometimes, 1.0 != 1.0, thanks computers
    assert all(np.isclose(clone_decomp.model_fit, 1.0)), \
        "Model fit between identical matrices not all 1.0"



def test_cli(tmp_path) -> None:
    """Test the click command line interface for reapplying."""

    runner: CliRunner = CliRunner()
    out_dir: pathlib.Path = tmp_path / "output_to"
    # Subset the example data as it's too large for tests
    pd.read_csv(
        files("cvanmf.data").joinpath("NW_ABUNDANCE.tsv"),
        sep="\t",
        index_col=0
    ).iloc[:, :20
    ].to_csv(
        tmp_path / "small_nw.tsv",
        sep="\t"
    )

    result = runner.invoke(
        cli,
        ("--abundance " +
         str(tmp_path / "small_nw.tsv") +
         " -o " +
         str(out_dir))
    )

    # Check output files are created
    for f in ['w.tsv', 'h.tsv', 'x.tsv', 'feature_mapping.tsv']:
        assert (out_dir / f).exists(), f"Output {f} not created"
