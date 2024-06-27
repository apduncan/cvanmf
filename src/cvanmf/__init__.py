"""Crossvalidated NMF (cvanmf)

A package implementing nine fold bi-crossvalidation for non-negative matrix
factorisation (NMF) parameter selection, as well as methods for analysis and
visualisation of NMF decompositions, and for transforming new data using an
existing model.

The underlying implementation of NMF is from scikit-learn, this package does
not provide a novel implementation of NMF.
"""

# read version from installed package
from importlib.metadata import version
__version__ = version("cvanmf")

# from cvanmf.reapply import reapply
# from cvanmf.denovo import (rank_selection, plot_rank_selection,
#                            regu_selection, plot_regu_selection, decompositions)
