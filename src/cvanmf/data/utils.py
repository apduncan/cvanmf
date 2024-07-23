"""Functions for dealing with example case study and synthetic data."""
import math
import re
from importlib.resources import files
from pathlib import Path
from typing import NamedTuple, Optional, Dict, Any, List, Union

import numpy as np
import pandas as pd


class ExampleData(NamedTuple):
    """Example data with associated metadata, including citations and
    description."""
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
    """Stick figure images of a swimmer with 4 limbs in 4 positions

    Designed by Donoho et al. to be fully representable by NMF, each image
    has a line torso, accompanied by four straight limbs which can be in one
    of four positions (0, 45, 90, 135 and 180 degrees from torso). With the
    exception of the torso, each limb position should be representable by
    NMF decomposition. As such, the true rank of this data is 17 (4*4 limbs,
    plus torso).
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


def synthetic_blocks(m: int = 100,
                     n: int = 100,
                     overlap: float = 0.25,
                     k: int = 3,
                     normal_noise_params: Optional[Dict] = None,
                     scale_lognormal_params: Optional[Dict] = None
                     ) -> ExampleData:
    """Create some simple synthetic data.

    Create an m x n matrix with blocks along the diagonal which overlap to an
    extent defined by overlap.

    :param m: Number of rows in matrix
    :param n: Number of columns in matrix
    :param overlap: Proportion of block length to participate in overlap
    :param k: Number of signatures
    :param normal_noise_params: Parameters to pass to numpy.random.normal
        to apply noise to entries. Leave as none to use default parameters.
    :param scale_lognormal_params: Parameters to pass to numpy.random.lognormal
        to scale each feature (give some features higher values than others).
        If set to true, will use default parameters for distribution. Leave as
        None to skip feature scaling.
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



