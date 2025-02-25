#!/usr/bin/env python
import logging
import pathlib
import pickle
from typing import Dict, List, Optional
from cvanmf import denovo
from cvanmf.stability import signature_stability, plot_signature_stability
import click
import pandas as pd
import plotnine

@click.command()
@click.option("--matrix", "-i", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              readable=True),
              help="Input full matrix")
@click.option("--output", "-o", required=True,
              type=click.Path(exists=False, file_okay=False, dir_okay=True),
              help="Directory to write output to.")
@click.option("--regu_res", "-r", required=False,
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              readable=True),
              help="Results from regularisation selection, to pick alpha.")
@click.option("--random_starts", '-r', type=int, default=100,
              help="Number of random initialisations for random init methods")
@click.option("--seed", "-s", type=int, default=7297108,
              help="Random state seed.")
@click.option("--rank", "-k", type=int, required=True,
              help="Rank for this decomposition.")
@click.option("--l1_ratio", "-l", type=float, default=1.0,
              help="Ratio of L1 to L2 regularisation.")
@click.option("--max_iter", "-m", type=int, default=3000,
              help="Maximum iterations during each run of NMF.")
@click.option("--init", type=str, required=False,
              help="Initialisation method")
@click.option("--beta_loss", "-b",
              type=click.Choice(['kullback-leibler', 'frobenius',
                                 'itakura-saito'], case_sensitive=False),
              default="kullback-leibler",
              help="Beta-loss function to use during decomposition.")
@click.option("--stability", type=bool, default=False,
              help="Calculate stability based rank selection coefficients. This can significantly increase memory requirements an execution time for large matrices.")
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True,
              help="Show verbose logs.")
def cli(matrix: str,
        output: str,
        regu_res: Optional[str],
        random_starts: int,
        seed: int,
        rank: int,
        l1_ratio: float,
        max_iter: int,
        init: str,
        beta_loss: str,
        stability: bool,
        verbose: bool) -> None:
    """Produce a decomposition with heuristically selected alpha when provided.

    :param matrix: Full matrix
    :param output: Directory to write to
    :param regu_res: Regularisation selection results
    :param seed: Random state seed
    :param rank: Rank of decomposition to run
    :param l1_ratio: Ratio of L1 to L2 regularisation
    :param max_iter: Maximum iterations to allow during NMF
    :param init: Initialisation method for W and H
    :param beta_loss: Beta-loss function for decomposition
    :param verbose: Activate verbose logging
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s', 
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )

    mat: pd.DataFrame = pd.read_csv(matrix, index_col=0, delimiter="\t")

    best_alpha: float = 0.0
    if regu_res is not None:
        # Load regularisation results, and matrix
        with open(regu_res, 'rb') as f:
            regu_dict: Dict[float, list[denovo.BicvResult]] = pickle.load(f)
        # Get heuristically determined best alpha
        best_alpha = denovo.suggest_alpha(regu_dict)
        # We don't need to scale this, will have already been scaled earlier if
        # desired
    else:
        logging.info("No regularisation results, not applying regularisation.")

    logging.info(
        "Full Matrix Decomposition\n"
        "---------------------------\n"
        f"matrix        : {matrix}\n"
        f"seed          : {seed}\n"
        f"rank          : {rank}\n"
        f"max_iter      : {max_iter}\n"
        f"l1_ratio      : {l1_ratio}\n"
        f"alpha         : {best_alpha}\n"
        f"init          : {init}\n"
        f"beta_loss     : {beta_loss}"
    )

    decompositions: Dict[int, List[denovo.Decomposition]] = (
        denovo.decompositions(
            x=mat,
            ranks=[rank],
            random_starts=random_starts,
            top_n=random_starts,
            seed=seed,
            alpha=best_alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            init=init,
            beta_loss=beta_loss
        )
    )

    # Output the best decomposition
    best_d: denovo.Decomposition = decompositions[rank][0]
    best_d.save(output)

    # Calculate the stability based values signature similarity now, as only 
    # the best decomposition will be retained.
    sig_per_inch: float = 0.4
    width: float = 1 + (sig_per_inch * rank)
    logging.info("Calculating signature stability")
    sig_sim: pd.DataFrame = signature_stability(
        decompositions[rank]
    )
    output_pth: pathlib.Path = pathlib.Path(output)
    sig_sim.to_csv(output_pth / "signature_similarity.tsv", sep="\t")
    plt_sig_sim: plotnine.ggplot = plot_signature_stability(
        sig_sim,
        colors=best_d.colors,
        ncol=1
    ) + plotnine.theme(figure_size=(3, width))
    plt_sig_sim.save(output_pth / "signature_similarity.pdf")

    if stability:
        logging.info("Calculating dispersion")
        dispersion_s: pd.Series = denovo.dispersion(decompositions)
        logging.info("Calculating cophenetic correlation")
        cophenetic_s: pd.Series = denovo.cophenetic_correlation(decompositions)
        logging.info("Calculating signature similarity")
        similarity_s: pd.Series = denovo.signature_similarity(decompositions)
        logging.info("Writing stability rank selection results")
        stability_df: pd.DataFrame = pd.concat(
            [dispersion_s, cophenetic_s, similarity_s],
            axis=1
        )
        stability_df.to_csv(output_pth / "stability_ranksel_values.tsv", sep="\t")


if __name__ == "__main__":
    cli()