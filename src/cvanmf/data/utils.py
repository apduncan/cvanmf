"""Functions for dealing with example case study and synthetic data."""
import itertools
import logging
import math
import random
import re
from importlib.resources import files
from pathlib import Path
from typing import NamedTuple, Optional, Dict, Any, List, Union, Set, Callable

import numpy as np
import numpy.random
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)

class ExampleData(NamedTuple):
    """Example data, including citations, description, and other metadata."""
    data: pd.DataFrame
    """Table containing the data."""
    rank: Optional[Union[int, List[int]]]
    """Correct rank for this data, or ranks if more than one can be 
    considered correct."""
    row_metadata: Optional[pd.DataFrame]
    """Metadata associated to each row."""
    col_metadata: Optional[pd.DataFrame]
    """Metadata associated to each column."""
    other_metadata: Optional[Dict[str, Any]]
    """Any other metadata related to this data (data dictionaries etc.)"""
    description: str
    """Longform description of the data."""
    name: str
    """Descriptive name."""
    short_name: str
    """Short name."""
    doi: Optional[str]
    """DOI for data or original paper."""
    citation: Optional[str]
    """Preferred citation if you use this data."""

    def _repr_html_(self) -> str:
        """HTML output summarising dataset for display in notebooks etc."""
        html: str = (
            "<table style='font-family: monospace'>"
            "<tr><td colspan='2' style='font-weight: bold'>"
            f"{self.name}</td></tr>"
            f"<tr><td style='font-size: smaller' colspan='2'>cvanmf example "
            f"data</td></tr>\n"
            f"<tr><td>shape</td><td>{str(self.data.shape)}</td></tr>\n"
            f"<tr><td>rank</td><td>{str(self.rank)}</td></tr>\n"
            f"<tr><td>description</td><td>"
            f"{_shorten_docstring(self.description)}</td></tr>\n"
        )
        if self.doi is not None:
            html += f"<tr><td>doi</td><td>{self.doi}</td></tr>\n"
        if self.citation is not None:
            html += f"<tr><td>citation</td><td>{self.citation}</td></tr>\n"
        html += "</table>"
        return html

    def __repr__(self) -> str:
        repr: str = (
            f"ExampleData[{self.short_name}, shape={self.data.shape}, "
            f"rank={self.rank}]"
        )
        return repr


def leukemia() -> ExampleData:
    """Gene expression data for ALL and AML B- and T-cell type leukemia.

    This data was analysed in Brunet et al (2004), and often used as a
    standard dataset for biological applications of NMF since. It has two
    broad categories (ALL/AML), but AML can be refined into two subtypes (B/T).
    B-cell AML appears to contain a further stable sub-grouping, so we have
    indicated the true rank of this data as 3 or 4.
    """

    data: pd.DataFrame = pd.read_csv(
            str(files("cvanmf.data").joinpath(
                Path("ALL_AML", "ALL_AML_counts.tsv.gz")
            )),
            sep="\t",
            index_col=0
        )
    col_md: pd.DataFrame = pd.DataFrame(
        data.columns.map(__label_all_leuk),
        index=data.columns
    )
    return ExampleData(
        data=data,
        row_metadata=pd.read_csv(
            str(files("cvanmf.data").joinpath(
                Path("ALL_AML", "ALL_AML_feature_md.tsv.gz")
            )),
            sep="\t",
            index_col=0
        ),
        col_metadata=col_md,
        other_metadata=None,
        rank=4,
        description=_shorten_docstring(leukemia.__doc__),
        name="Leukemia (ALL/AML) Gene Expression",
        doi="doi:10.1073/pnas.030853110",
        citation=("Brunet, J-P., Tamayo, P., Golub, T. R. & Mesirov, "
                  "J. P. Metagenes and molecular pattern discovery using "
                  "matrix factorization. Proc. Natl. Acad. Sci. U.S.A. 101, "
                  "4164–4169 (2004)."),
        short_name="ALL_AML"
    )


def swimmer() -> ExampleData:
    """Stick figure images of a swimmer with 4 limbs in 4 positions.

    Designed by Donoho et al. to be partially representable by NMF, each image
    has a line torso, accompanied by four straight limbs which can be in one
    of four positions (0, 45, 90, 135 and 180 degrees from torso). With the
    exception of the torso, each limb position should be representable by
    NMF decomposition. As such, the true rank of this data is 17 (4*4 limbs,
    plus torso), but with conventional NMF as implemented here the torso cannot
    be learnt, only the 16 limbs.
    """

    return ExampleData(
        data=pd.read_csv(
            str(files("cvanmf.data").joinpath(
                Path("swimmer", "swimmer.tsv.gz")
            )),
            sep="\t",
            index_col=0
        ),
        rank=17,
        row_metadata=None,
        col_metadata=None,
        other_metadata=None,
        description=_shorten_docstring(swimmer.__doc__),
        name="Swimmer Dataset",
        doi="",
        citation=("Donoho, D. & Stodden, V. When Does Non-Negative Matrix "
                  "Factorization Give a Correct Decomposition into Parts? in "
                  "Advances in Neural Information Processing Systems (eds. "
                  "Thrun, S., Saul, L. & Schölkopf, B.) vol. 16 (MIT Press, "
                  "2003)."),
        short_name="swimmer"
    )


def lung_cancer_cells() -> ExampleData:
    """Relative cell-compositions from non-small cell lung cancer studies.

    Gives the number of cells of different types in lung tissue samples from a
    non-small cell lung cancer atlas, which was compiled from 29 studies and
    includes 556 samples, from 318 individuals (86 of which are healthy
    controls). The data was uploaded to cellxgene using their standard
    ontologies, which is the source we have taken the data from. Metadata
    provided here is a mixture of metadata from cellxgene, and some from the
    original paper. We have selected out only the tissues samples labelled as
    "lung". In total, this gives 224 samples, and 33 cell types. The data
    here is total-sum-scaled, i.e. each sample sums to 1.
    """

    return ExampleData(
        data=pd.read_csv(
            str(files("cvanmf.data").joinpath(
                Path("NSCLC", "nsclc.lung.composition.tsv.gz")
            )),
            sep="\t",
            index_col=0
        ),
        rank=[],
        row_metadata=None,
        col_metadata=pd.read_csv(
            str(files("cvanmf.data").joinpath(
                Path("NSCLC", "nsclc.lung.metadata.tsv.gz")
            )),
            sep="\t",
            index_col=0
        ),
        other_metadata=None,
        description=_shorten_docstring(lung_cancer_cells.__doc__),
        name="Non-small cell lung cancer",
        doi="https://doi.org/10.1016/j.ccell.2022.10.008",
        citation=("Salcher, S. et al. High-resolution single-cell atlas "
                  "reveals diversity and plasticity of tissue-resident "
                  "neutrophils in non-small cell lung cancer. Cancer Cell 40, "
                  "1503-1520.e8 (2022). \n"
                  "Program, C. S.-C. B. et al. CZ CELL×GENE Discover: A "
                  "single-cell data platform for scalable exploration, "
                  "analysis and modeling of aggregated data. "
                  "2023.10.30.563174 Preprint at "
                  "https://doi.org/10.1101/2023.10.30.563174 (2023)."),
        short_name="NSCLC"
    )


def synthetic_blocks(m: int = 100,
                     n: int = 100,
                     overlap: float = 0.25,
                     k: int = 3,
                     normal_noise_params: Optional[Dict] = None,
                     scale_lognormal_params: Optional[Dict] = None
                     ) -> ExampleData:
    """Generate simple synthetic data.

    Create an m x n matrix with blocks along the diagonal which overlap to an
    extent defined by overlap.

    :param m: Number of rows in matrix
    :param n: Number of columns in matrix
    :param overlap: Proportion of block length to participate in overlap
    :param k: Number of signatures
    :param normal_noise_params: Parameters to pass to `numpy.random.normal`
        to apply noise to entries. Leave as none to use default parameters.
    :param scale_lognormal_params: Parameters to pass to
        `numpy.random.lognormal` to scale each feature (give some features
        higher values than others). If set to true, will use default parameters
        for distribution. Leave as None to skip feature scaling.
    """

    # Matrix dimensions
    i, j = m, n

    # Noise parameters
    do_scale: bool = scale_lognormal_params is not None
    scale_lognormal_params: Dict = (
        dict() if scale_lognormal_params is None or
        not isinstance(scale_lognormal_params, dict)
        else scale_lognormal_params
    )
    normal_noise_params = (
        dict() if normal_noise_params is None
             else normal_noise_params
    )

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
    normal_noise_params = dict(loc=0.0, scale=0.1) | normal_noise_params
    block_mat = block_mat + np.absolute(
        np.random.normal(size=(i, j), **normal_noise_params))
    # Scale features
    if do_scale:
        fscale: np.ndarray = np.random.lognormal(
            **scale_lognormal_params,
            size=m
        )
        block_mat = fscale.reshape(-1, 1) * block_mat
    # Convert to DataFrame and add some proper naming for rows/cols
    df: pd.DataFrame = pd.DataFrame(
        block_mat,
        index=[f'feat_{i_lab}' for i_lab in range(i)],
        columns=[f'samp_{j_lab}' for j_lab in range(j)])
    # TSS scale
    df = df / df.sum()
    return ExampleData(
        data=df,
        rank=k,
        row_metadata=None,
        col_metadata=None,
        other_metadata=dict(
            overlap=overlap,
            scale_lognormal_params=scale_lognormal_params,
            normal_noise_params=normal_noise_params
        ),
        doi="",
        citation="",
        description="Synthetic data with blocks along the diagonal which "
                    "overlap to a defined extent.",
        name="Synthetic Overlapping Blocks",
        short_name="SYNTH_BLOCKS",
    )


def synthetic_dense(m: int = 100,
                    n: int = 100,
                    h_sparsity: float = 0.0,
                    shared_features: float = 0.25,
                    k: int = 3,
                    normal_noise_params: Optional[Dict] = None,
                    scale_lognormal_params: Optional[Dict] = None,
                    keep_mats: bool = False
                    ) -> ExampleData:
    """Generate dense synthetic data.

    Dense data is generated by making a :math:`W` matrix with :math:`k`
    signatures, and multiplying this with a randomly filled :math:`H` matrix.
    Optionally, a proportion of the :math:`H` matrices can be randomly set to 0
    using `h_sparsity`. The extent to which features are shared between
    signatures is defined via `shared_features`. Each signature is initially
    assigned an even proportion of the :math:`m` features (remainder spread as
    evenly as possible between them), so there are no shared features. Then if
    :math:`|k|` is the number of features assigned to a signature, each
    signature is assigned :math:`|k|*shared\_features` randomly selected from
    the remaining features. This means the overlapping structure is potentially
    quite different from that of :func:`synthetic_blocks`.

    :param m: Number of features.
    :param n: Number of samples.
    :param h_sparsity: Proportion of :math:`H` matrix to randomly set to 0
    :param shared_features: Amount of shared features to add to a signature, as
        a proportion of it's base size.
    :param k: Number of signatures.
    :param normal_noise_params: Parameters passed to
        :func:`numpy.random.normal` when adding noise to data.
    :param scale_lognormal_params: Parameters passed to
        :func:`numpy.random.lognormal` when selecting weights for features in a
        signature. If this is None, a uniform distribution between 0 and 1 is
        used instead.
    :param keep_mats: Return the :math:`H` and :math:`W` matrices used to
        generate the data in the :attr:`ExampleData.other_metadata`.
    """

    # Create W
    k_scale: np.array = np.array([1] * k)
    k_feats: np.array = np.floor((k_scale / k_scale.sum()) * m)
    rem: int = m - int(k_feats.sum())
    # Got to be a nicer way to do this
    for i in range(rem):
        k_feats[i] = k_feats[i] + 1
    k_feat_names: List[Set[str]] = [
        set(f'sig{i}_feat{j}' for j in range(int(k_feats[i]))) for i in range(k)
    ]
    # Add shared features
    all_feats: Set[str] = set(itertools.chain.from_iterable(k_feat_names))
    for k_names in k_feat_names:
        other: List[str] = random.choices(
            list(all_feats.difference(k_names)),
            k=math.floor(len(k_names) * shared_features)
        )
        k_names.update(other)
    w: np.ndarray
    if scale_lognormal_params:
        w = np.random.lognormal(size=(m, k), **scale_lognormal_params)
    else:
        w = np.random.uniform(size=(m, k), low=0.0, high=1.0)
    mask: pd.DataFrame = pd.DataFrame(
        np.zeros(shape=(m, k)),
        index=list(all_feats),
        columns=[i for i in range(k)]
    ).sort_index()
    mask = mask.apply(lambda x: x.index.isin(k_feat_names[x.name]))
    w = w * mask.values

    # Create H
    h: np.ndarray = np.random.uniform(low=0.0, high=1.0, size=n*k)
    h[numpy.random.choice(
        list(range(n*k)),
        size=math.floor(n*k*h_sparsity),
        replace=False
    )] = 0.0
    h = h.reshape(k, n)

    wh: pd.DataFrame = pd.DataFrame(
        w.dot(h),
        index=mask.index,
        columns=[f's{i}' for i in range(n)]
    )
    normal_noise_params = (dict() if normal_noise_params is None else
                           normal_noise_params)
    normal_noise_params = dict(loc=0.0, scale=0.1) | normal_noise_params
    wh = (wh + numpy.random.normal(size=wh.shape,
                                   **normal_noise_params)).clip(lower=0)

    other_metadata: Dict[str, Any] = dict(
        h_sparsity=h_sparsity,
        shared_features=shared_features,
        scale_lognormal_params=scale_lognormal_params,
        normal_noise_params=normal_noise_params
    )
    if keep_mats:
        other_metadata['h'] = h
        other_metadata['w'] = w

    return ExampleData(
        data=wh,
        col_metadata=None,
        row_metadata=mask,
        rank=k,
        other_metadata=other_metadata,
        name="Synthetic Dense Data",
        short_name="SYNTH_DENSE",
        doi="",
        citation="",
        description=""
    )


def example_abundance() -> pd.DataFrame:
    """Genus level Non-Western cohort bacterial microbiome abundance.

    From Frioux et al. (2023, https://doi.org/10.1016/j.chom.2023.05.024).

    :return: Genus level relative abundance table using GTDB r207 taxonomy.
    :rtype: pd.DataFrame
    """
    return pd.read_csv(
        str(files("cvanmf.data").joinpath("NW_ABUNDANCE.tsv")),
        sep="\t",
        index_col=0
    )

def _shorten_docstring(s: str) -> str:
    """Strip newlines and multiple spaces from a docstring to make it a more
    readable format when returned."""
    # Want to keep any double newlines, but remove any singles
    # This is a very messy way of doing it but didn't have time to work out a
    # regex
    s = (
        s.replace("\n\n", ":lb:")
        .replace("\n", "")
        .replace(":lb:", "\n\n")
    )
    s = re.sub(r"[\t ]+", " ", s)
    return s

def __label_all_leuk(smpl: str) -> str:
    if smpl[:3] == "AML":
        return "AML"
    if "B-cell" in smpl:
        return "ALL B-Cell"
    if "T-cell" in smpl:
        return "ALL T-Cell"
    raise ValueError("Leukemia sample with incorrect label encountered")



