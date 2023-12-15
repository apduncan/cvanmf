from typing import Set

import pandas as pd
import pytest

from enterosig import models
from enterosig.transform import GenusMapping, EnteroException


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