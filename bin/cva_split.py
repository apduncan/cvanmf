#!/usr/bin/env python
import pathlib
from typing import List
from cvanmf import denovo
import pandas as pd
import click

@click.command()
@click.option("--matrix", "-m", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              readable=True),
              help="Input matrix in tab-separated format.")
@click.option("--seed", "-s", type=int, default=7297108,
              help="Random state seed.")
@click.option("--num_shuffles", "-n", type=int, required=True,
              help="Number of random shuffles of the matrix to produce.")
def cli(matrix: str, seed: int, num_shuffles: int) -> None:
    """Shuffle and split a matrix for NMF 9-fold bi-fold-cross validation. 
    The output is shuffled matrices in npz format suitable for loading by
    cvanmf.

    :param matrix: Input matrix location
    :param seed: Random seed
    :param num_shuffles: This is the i-th split
    """
    x: pd.DataFrame = pd.read_csv(matrix, sep="\t", index_col=0)
    shuffs: List[denovo.BicvSplit] = (
        denovo.BicvSplit
        .from_matrix(x, n=num_shuffles, random_state=seed)
    )
    denovo.BicvSplit.save_all_npz(shuffs, pathlib.Path(""),
                                   compress=True)


if __name__ == "__main__":
    cli()