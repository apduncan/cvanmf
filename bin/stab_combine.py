#!/usr/bin/env python

import itertools
import logging
import pathlib
from typing import Dict, List, Union
import pickle

from cvanmf import denovo
import click
import pandas as pd


@click.command()
@click.argument("files", nargs=-1,
                type=click.Path(exists=True, file_okay=False, dir_okay=True,
                                readable=True))
def cli(files: List[str]) -> None:
    """Join the stability rank selection criteria from multiple non-regularised models."""

    res: List[pd.Series] = [
        pd.read_csv(pathlib.Path(x) / "stability_ranksel_values.tsv",
                    sep="\t")
        for x in files]
    
    df: pd.DataFrame = pd.concat(res)
    cols = list(df.columns)
    cols[0] = 'rank'
    df.columns = cols
    df.to_csv("stability_rank_analysis.tsv", sep="\t")


if __name__ == "__main__":
    cli()
