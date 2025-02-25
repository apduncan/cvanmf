#!/usr/bin/env python

import itertools
import logging
from typing import Dict, List, Union
import pickle

from cvanmf import denovo
import click


@click.command()
@click.option("--group", "-g", 
              type=click.Choice(['alpha', 'rank'], case_sensitive=False),
              required=True, default="rank")
@click.argument("files", nargs=-1,
                type=click.Path(exists=True, file_okay=True, dir_okay=False,
                                readable=True))
def cli(files: List[str], group: str) -> None:
    """Join the individual BicvResult objects into the dictionary format
    expected by cvanmf for plotting."""

    res: List[denovo.BicvResult] = [pickle.load(open(f, 'rb')) for f in files]

    results: List[denovo.BicvResult] = sorted(
        res,
        key=lambda x: getattr(x.parameters, group)
    )

    # Collect results from the ranks/alphas into lists, and place in a 
    # dictionary with key = rank
    grouped_results: Dict[Union[int, float], List[denovo.BicvResult]] = {
        rank_i: list(list_i) for rank_i, list_i in
        itertools.groupby(results, key=lambda z: getattr(z.parameters, group))
    }

    # Validate that there are the same number of results for each value.
    # Decomposition shouldn't silently fail, but best not to live in a
    # world of should. Currently deciding to warn user and still return
    # results.
    prefix: str = "rank" if group == "rank" else "regu"
    if len(set(len(y) for y in grouped_results.values())) != 1:
        logging.error(("Uneven number of results returned for each rank, "
                       "some rank selection iterations may have failed."))

    with open(f"{prefix}_combined.pickle", "wb") as f:
        pickle.dump(grouped_results, f)
    
    (denovo.BicvResult.results_to_table(grouped_results)
     .to_csv(f'{prefix}_selection.tsv', sep="\t"))

if __name__ == "__main__":
    cli()
