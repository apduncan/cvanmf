"""Load existing Enterosignature models."""
import logging
from importlib.resources import files

import pandas as pd


def five_es() -> pd.DataFrame:
    """The 5 Enterosignature model of Frioux et al.
    (2023, https://doi.org/10.1016/j.chom.2023.05.024). A summary of this model
    can also be found on the website https://enterosignatures.quadram.ac.uk

    :return: W matrix of 5 Enterosignature model
    :type: pd.DataFrame
    """
    logging.info("If you use this model please cite Frioux et al. "
                 "(2023, https://doi.org/10.1016/j.chom.2023.05.024)")

    # with resources.path("enterosig.data", "ES5_W.tsv") as f:
    return pd.read_csv(
        str(files("enterosig.data").joinpath("ES5_W.tsv")),
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
        str(files("enterosig.data").joinpath("NW_ABUNDANCE.tsv")),
        sep="\t",
        index_col=0
    )