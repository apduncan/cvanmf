"""Load existing Enterosignature models."""
import logging
from importlib.resources import files
from typing import NamedTuple, List, Optional, Dict, Union

import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)

FIVE_ES_COLORS = {
    "ES_Bact": "#E69F00",
    "ES_Firm": "#023e8a",
    "ES_Prev": "#D55E00",
    "ES_Bifi": "#009E73",
    "ES_Esch": "#483838"
}


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
    colors: Union[List[str], Dict[str, str]]
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
    can also be found on the website https://enterosignatures.quadram.ac.uk.
    The `reapply` method for this model will normalise (total-sum-scale) input
    data after applying filters to match model format, so data provided does
    not need to be normalised.

    :return: 5 Enterosignature model
    :type: Signatures
    """

    w: pd.DataFrame = pd.read_csv(
        str(files("cvanmf.data").joinpath("ES5_W.tsv")),
        sep="\t",
        index_col=0
    )
    citation: str = (
        "Frioux, C. et al. Enterosignatures define common bacterial guilds in "
        "the human gut microbiome. Cell Host & Microbe 31, 1111-1125.e6 ("
        "2023). https://doi.org/10.1016/j.chom.2023.05.024")
    logger.warning("If you use the 5ES model please cite %s",
                   citation)
    from cvanmf import reapply
    return Signatures(w=w,
                      colors=FIVE_ES_COLORS,
                      feature_match=reapply.match_genera,
                      input_validation=reapply.validate_genus_table,
                      citation=(
                          "Frioux, C. et al. Enterosignatures define common "
                          "bacterial guilds in the human gut microbiome. "
                          "Cell Host & Microbe 31, 1111-1125.e6 (2023)."
                          " https://doi.org/10.1016/j.chom.2023.05.024")
                      )
