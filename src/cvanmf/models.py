"""Load existing Enterosignature models."""
import logging
import math
from importlib.resources import files
from typing import NamedTuple, List, Optional, Callable, Dict, Any

import numpy as np
import pandas as pd


class Signatures(NamedTuple):
    """Definition of an existing signature model.

    This provides the definition of existing signatures required to reapply
    the signature model to new data. Where Decomposition stores the input and
    H matrix, these are not necessary for transforming new data. Rather, we
    only need the W matrix, the colors associated with each signature (for
    consistency of representation), and the preprocessing steps (to match
    features in the new data with those in the W matrix)."""
    w: pd.DataFrame
    """Feature weights (W matrix) for this model."""
    colors: List[str]
    """Color for each signature in the model."""
    feature_match: 'FeatureMatch'
    """Function to map features in new data to those in the model W matrix."""
    input_validation: 'InputValidation' = lambda x: x
    """Function to validate and potentially transform input table. Defaults
    to identity function"""
    citation: Optional[str] = None
    """Citation when using this model."""

    def reapply(self,
                y: pd.DataFrame,
                **kwargs) -> 'Decomposition':
        """Transform new data using this signature model.

        :param y: New data of same type as the existing model.
        """
        from cvanmf import reapply
        return reapply._reapply_model(
            y=y,
            **(self._asdict() | kwargs)
        )


def five_es() -> Signatures:
    """The 5 Enterosignature model of Frioux et al.
    (2023, https://doi.org/10.1016/j.chom.2023.05.024). A summary of this model
    can also be found on the website https://enterosignatures.quadram.ac.uk

    :return: 5 Enterosignature model
    :type: Signatures
    """

    w: pd.DataFrame = pd.read_csv(
        str(files("cvanmf.data").joinpath("ES5_W.tsv")),
        sep="\t",
        index_col=0
    )
    from cvanmf import reapply
    return Signatures(w=w,
                      colors=None,
                      feature_match=reapply.match_genera,
                      input_validation=reapply.validate_genus_table,
                      citation=(
                          "Frioux, C. et al. Enterosignatures define common "
                          "bacterial guilds in the human gut microbiome. "
                          "Cell Host & Microbe 31, 1111-1125.e6 (2023)."
                          " https://doi.org/10.1016/j.chom.2023.05.024")
                      )


def five_es_x() -> pd.DataFrame:
    """The genus level relative abundance data used to train five ES model in
    Frioux et al. (2023, https://doi.org/10.1016/j.chom.2023.05.024).

    :return: Genus level relative abundance table using GTDB r207 taxonomy
    :rtype: pd.DataFrame
    """
    return pd.read_csv(
        str(files("cvanmf.data").joinpath("ES5_X.tsv")),
        sep="\t",
        index_col=0
    )


def example_abundance() -> pd.DataFrame:
    """The genus level relative abundance data for Non-Western cohort from
    Frioux et al. (2023, https://doi.org/10.1016/j.chom.2023.05.024).

    :return: Genus level relative abundance table using GTDB r207 taxonomy
    :rtype: pd.DataFrame
    """
    return pd.read_csv(
        str(files("cvanmf.data").joinpath("NW_ABUNDANCE.tsv")),
        sep="\t",
        index_col=0
    )


def synthetic_data(m: int = 100,
                   n: int = 100,
                   overlap: float = 0.25,
                   k: int = 3) -> pd.DataFrame:
    """Create some simple synthetic data.

    Create an m x n matrix with blocks along the diagonal which overlap to an extent
    defined by overlap.

    :param m: Number of rows in matrix
    :param n: Number of columns in matrix
    :param overlap: Proportion of block length to participate in overlap
    :param k: Number of signatures
    """

    # Matrix dimensions
    i, j = m, n

    # Width of blocks without overlap
    base_h, tail_h = divmod(i, k)
    base_w, tail_w = divmod(j, k)
    # Overlap proportion - proportion of block's base dimension to extend
    # block by
    overlap_proportion: float = overlap
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
