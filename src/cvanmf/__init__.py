"""CrossVAlidated NMF (cvanmf)

A package implementing Gabriel holdout bicrossvalidation for non-negative matrix
factorisation (NMF) parameter selection, as well as methods for analysis and
visualisation of NMF decompositions, and for transforming new data using an
existing models.

The underlying implementation of NMF is from scikit-learn, this package does
not provide a novel implementation of NMF.
"""
import logging
# read version from installed package
from importlib.metadata import version
__version__ = version("cvanmf")

# Configure logging
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
ch: logging.Handler = logging.StreamHandler()
fm: logging.Formatter = logging.Formatter(
    fmt='%(levelname)s [%(name)s] [%(asctime)s]: %(message)s',
    datefmt='%d/%m/%Y %I:%M:%S'
)
ch.setFormatter(fm)
logger.addHandler(ch)

# from cvanmf.reapply import reapply
# from cvanmf.denovo import (rank_selection, plot_rank_selection,
#                            regu_selection, plot_regu_selection, decompositions)
