"""Example data for decomposition.

This modules provides both real world datasets, and functions to synthetic data
with a known number of signatures. Each is returned as an
:class:`ExampleData` object, which includes metadata and
citations for the data where appropriate. Each can be loaded by calling the
relevant function (i.e. `data.leukemia()`)

Real Data
---------
:func:`leukemia` is gene expression data from ALL/AML-type leukemia
patients. :func:`lung_cancer_cells` is the count of types of cells in
non-small cell lung cancer tissues. :func:`swimmer` is an image dataset
with stick figure representations of a swimmer from above.

Synthetic Data
--------------
:func:`synthetic_blocks` makes data with an overlapping block pattern along
the diagonal. :func:`synthetic_dense` makes data with a dense structure
(each sample can contain any number of signatures).
"""

from cvanmf.data.utils import (leukemia, swimmer,
                               lung_cancer_cells, example_abundance,
                               synthetic_blocks, synthetic_dense, ExampleData)
