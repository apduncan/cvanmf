API Reference
=============

This page contains auto-generated API reference documentation [#f1]_.

For generating new models, see :mod:`cvanmf.denovo`. To reapply a model, use
:mod:`cvanmf.reapply`. Some existing models (currently just the five
Enterosignature model) as provided by :mod:`cvanmf.models`. Example datasets, both synthetic
and read data, are provided in :mod:`data`.

.. toctree::
    :titlesonly:

    cvanmf/denovo/index
    cvanmf/reapply/index
    cvanmf/stability/index
    cvanmf/models/index
    cvanmf/data/index

Experimental Modules
--------------------
:py:mod:`cvanmf.combine` is an experimental module for combining models learnt
from multiple cohorts to a single non-redundant pool of signatures. It is less
well documented that other modules. Some of the elements will eventually be
moved into :mod:`cvanmf.stability` to characterise how stable signatures are
between multiple decompositions with the same parameters, but this module is
currently incomplete, and so not linked above as part of the main documentation

.. [#f1] Created with `sphinx-autoapi <https://github.com/readthedocs/sphinx-autoapi>`_
