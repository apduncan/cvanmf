#!/usr/bin/env python
"""
Reapply existing Enterosignature models to new abundance data.

The easiest way to do this is through the :func:`reapply` function, which is
most flexible about parameter types. The other functions perform individual
steps, which are useful if you want fine control of a given step, but
probably not necessary for most uses.
"""
import abc
import logging
import os
import pathlib
import re
from collections import namedtuple
from typing import (Any, Callable, Collection, Dict, Iterator, List, NamedTuple,
                    Optional, Set, Tuple, Union, Protocol)

import click
import numpy as np
import pandas as pd
from sklearn.decomposition import non_negative_factorization
from sklearn.metrics.pairwise import cosine_similarity

# from cvanmf import models, denovo
# Compile regular expressions
RE_RANK: re.Pattern = re.compile("[a-zA-Z]__")
RE_SPLIT_GENUS: re.Pattern = re.compile(r"([\w]+)_\w$")
# Match the last named rank, plus any following ?s
RE_SHORTEN: re.Pattern = re.compile(r".*;([^\;?]+[;\?]*)$")


class EnteroException(Exception):
    """Exception raised when unable to proceed with enterosignature 
    transformation"""


class FeatureMapping():
    """Connect new data table features to those in the model.

    Manage the mappings from input features to model features. Source features
    are the features in the new abundance table we want to fit to the model;
    target features are the features in the model we're trying to match to.
    User defined mappings can be provided via `hard_map`, any subsequent
    mappings for a source taxon in `hard_map` will be ignored. New mappings are
    added via :meth:`add`. When mappings are fully defined the model w matrix
    and the new data table can be matched using :meth:`transform_w` and
    :meth:`transform_abundance`
    """

    def __init__(self,
                 target_features: Set[str],
                 source_features: Set[str],
                 hard_map: Optional[Dict[str, str]] = None):
        """
        :param target_features: Model features to map to
        :type target_features: Set[str]
        :param source_features: Input features to be mapped from
        :type source_features: Set[str]
        :param hard_map: User defined mappings, as a dictionary with source
            as key and target as value.
        :type hard_map: Optional[Dict[str, str]]
        """

        # hard_map are user provided mappings which will never be altered
        self.__target_features = target_features
        self.__source_features = source_features
        self.__hard_map = {} if hard_map is None else hard_map
        self.__map: Dict[str, List[str]] = dict()

    def add(self, feature_from: str, feature_to: str) -> None:
        """Add a mapping. If there is already a mapping from this feature, we
        will append this one. Use :meth:`conflicts` to identify where more than
        one mapping exists.

        :param feature_from: Feature in the new table
        :type feature_from: str
        :param feature_to: Model feature to map to
        :type feature_to: str
        :raises EnteroException: Feature not in the relevant sets
        """
        # If user has provided a hard mapping, ignore this mapping
        if feature_from in self.__hard_map:
            return
        # If the genus being mapped to is not valid, raise an exception
        if feature_to not in self.__target_features:
            raise EnteroException(
                f"Mapping of '{feature_from}' to target '{feature_to}' not "
                f"possible as target does not exist.")
        if feature_from not in self.__source_features:
            raise EnteroException(
                f"Mapping of '{feature_from}' to target '{feature_to}' not "
                f"possible as source does not exist.")
        if feature_from not in self.__map:
            self.__map[feature_from] = [feature_to]
        else:
            self.__map[feature_from].append(feature_to)

    def to_df(self) -> pd.DataFrame:
        """Produce a dataframe of the mapping. Where mappings are amibiguous,
        multiple rows will be included. Where mappings are missing, one row with
        a blank target will be included.

        :return: DataFrame with two columns, first source feature, second target
            feature.
        :rtype: pd.DataFrame
        """

        # Could be more nicely done with some maps, but take a simple approach
        df_src: List[Tuple[str, str]] = []
        for source, targets in self.mapping.items():
            for target in targets:
                df_src.append((source, target))
        missing: List[Tuple[str, str]] = list(
            zip(self.missing(), [''] * len(self.missing())))
        df_src = df_src + missing
        df: pd.DataFrame = pd.DataFrame(
            df_src, columns=['input_feature', 'model_feature'])
        return df

    def missing(self) -> Collection[str]:
        """Identify input features which currently have no mapping.

        :return: Source features which are not mapping to any model feature
        :rtype: Collection[str]
        """
        return self.__source_features.difference(set(self.mapping.keys()))

    def transform_abundance(self, abd_tbl: pd.DataFrame) -> pd.DataFrame:
        """Applying mapping to the input table.

        Make a table with renamed and combined rows based on the
        identified mappings.

        :param abd_tbl: New table, samples on columns
        :type abd_tbl: pd.DataFrame
        :return: Table with mappings applied
        :rtype: pd.DataFrame
        """

        # Probably we could do some smarter changes and groupbys
        # TODO(apduncan): Refactor, slow and ugly
        abd: pd.DataFrame = abd_tbl.copy()
        for input_feature, feature_maps in self.mapping.items():
            # Where there are conflicting mappings, use the first one
            feature_map = feature_maps[0]
            if input_feature == feature_map:
                continue
            # Add or sum
            if feature_map not in abd.index:
                abd.loc[feature_map] = abd.loc[input_feature]
            else:
                abd.loc[feature_map] += abd.loc[input_feature]
            abd = abd.drop(labels=[input_feature])
        return abd

    def transform_w(self,
                    w: pd.DataFrame,
                    abd_tbl: pd.DataFrame) -> pd.DataFrame:
        """Match the model w matrix to the new table.

        Make a W matrix which has features not in the abundance table removed,
        and rows added for features which are in the abundance table but not the
        model.

        :param w: Model W matrix
        :type w: pd.DataFrame
        :param abd_tbl: New matrix. Should `not` have been transformed
            with :meth:`transform_abundance`.
        :type abd_tbl: pd.DataFrame
        :return: W matrix matched to new table
        :rtype: pd.DataFrame
        """
        w_new: pd.DataFrame = w.copy()
        # Drop taxa which are not in the input matrix from the W matrix
        w_feat: Set[str] = set(w_new.index)
        w_new = w_new.drop(labels=w_feat.difference(set(abd_tbl.index)))
        # Add taxa which are in the input but not in the W matrix
        missing_features: Set[str] = set(abd_tbl.index).difference(w_feat)
        missing_df: pd.DataFrame = pd.DataFrame(
            np.zeros(shape=[len(missing_features), w.shape[1]]),
            index=list(missing_features),
            columns=w.columns
        )
        w_new = pd.concat([w_new, missing_df])
        return w_new

    @property
    def conflicts(self) -> List[Tuple[str, List[str]]]:
        """Features for which more than one target exists."""
        return list(filter(lambda x: len(x[1]) > 1, self.mapping.items()))

    @property
    def mapping(self) -> Dict[str, List[str]]:
        """Mapping from source to target features."""
        return {**self.__map, **{x: [y] for x, y in self.__hard_map.items()}}


class ReapplyResult(NamedTuple):
    """A named tuple containing results from transformation process."""
    w: pd.DataFrame
    """Model weights with index matched to input abundances."""
    h: pd.DataFrame
    """Enterosignature weights for each sample. Not these are not relative
    (do not sum to 1 per sample)."""
    abundance_table: pd.DataFrame
    """Abundances generated by matching input abundance taxa to taxa in the 
    model weights."""
    model_fit: pd.DataFrame
    """Model fit for each sample, ranging from 1 (good) to 0 (poor)."""
    taxon_mapping: FeatureMapping
    """Object defining how input abundance taxa were mapped to model taxa, 
    of type :class:`GenusMapping`."""


def _console_logger(message: Any) -> None:
    logging.info(message)


def _is_taxon_unknown(taxon: str) -> bool:
    # Remove ?, if all entries in lineage blank, the is unknown
    return all(
        map(
            lambda x: len(x.strip()) == 0,
            taxon.replace("?", "").split(';')
        )
    )


def _contain_rank_indicators(taxon: str) -> bool:
    # Check if this taxonomy contains rank indicators
    return any(map(
        lambda x: RE_RANK.search(x.strip()) is not None,
        taxon.split(';')
    ))


def _shorten_genus(
        genus: str,
        genera_list: Collection[str],
        short_genera_list: Collection[str]
) -> str:
    # Copying the operation of Clemence's shorten_genus function
    # Get last named rank, and subsequent ?
    # Genera list should the genera list, but with RE_SHORTEN applied, not 
    # applying each call as computationally inefficient (especially if we want 
    # to run on streamlit's servers)
    # Count occurences in the genera list
    short: str = RE_SHORTEN.sub(r"\1", genus)
    nb_occ: int = sum(map(lambda x: short in x, genera_list))
    nb_occ_s: int = sum(map(lambda x: short in x, short_genera_list))
    # shorten2 is the last three bits of the lineage, if we observed short in 
    # the short form genera list more than once, else we just keep short
    shorten2: str = ";".join(genus.split(';')[-3:]) if nb_occ_s > 1 else short
    final_shorten: str = short
    if nb_occ != 1:
        if nb_occ_s == 1:
            # Extract the final 2
            final_shorten = ";".join(genus.split(";")[-2:])
        else:
            final_shorten = shorten2
    return final_shorten


def _final_rank(taxon: str) -> str:
    # Exctract final non-blank or ? rank
    return next(filter(
        lambda x: len(x.strip()) != 0,
        reversed(taxon.replace("?", "").split(';'))
    ))


def _final_rank_equal(a: str, b: str) -> bool:
    return _final_rank(a) == _final_rank(b)


def _pad_lineage(
        taxon: str,
        delim: str = ";",
        length: int = 6,
        unknown: str = "") -> str:
    """Pad lineage out to the expected length using delimiters. Default is to
    genus level.

    :param str taxon: Lineage string for taxon
    :param str delim: Delimiter between ranks
    :param int length: Rank to truncate to
    :param str unknown: Text to fill in unknown ranks
    """

    lineage_parts: List[str] = taxon.split(delim)
    diff: int = length - len(lineage_parts)
    if diff > 0:
        lineage_parts += [unknown] * diff
    lineage_parts = lineage_parts[0:length]
    # Replace any length 0 entries with the unknown indicator
    lineage_parts = list(map(
        lambda x: x if len(x) > 0 else unknown,
        lineage_parts
    ))
    return delim.join(lineage_parts)


def validate_table(abd_tbl: pd.DataFrame,
                   **kwargs
                   ) -> pd.DataFrame:
    """Basic checks and transformations of the abundance table. 
    
    Some transformations may be made here, such as transposition. Any 
    transformation will be written out to inform the user. Transformations are 
    done in place.
    
    :param abd_tbl: Abundance table to check
    :type abd_tbl: pd.DataFrame
    :param logger: Function to report errors
    :type logger: Callable[[str], None]
    :returns: Validated, potentially transformed, dataframe
    :rtype: pd.DataFrame
    """

    # Check the dimensions make sense
    if abd_tbl.shape[0] < 2 or abd_tbl.shape[1] < 2:
        logging.critical(
            """Table has one or fewer columns or rows. Check delimiters and
            newline formats are correct.""")
        raise EnteroException("Table incorrect format.")

    # Check that there were column names in the file. We're going to assume
    # that if column names are all numeric values, there were no headers.
    # Unfortunate for people who only gave their samples numbers, but alas.
    all_numeric: bool = all(map(lambda x: str(x).isnumeric(), abd_tbl.columns))
    if all_numeric:
        logging.error(
            """Table appear to lack sample IDs in the first row. Add
            sample IDs, or ensure all sample IDs are not numeric.""")
        raise EnteroException("Table lacks sample IDs in first row.")

    # Check that taxa are on rows. We'll do that by looking for "Bacteria;"
    # in the column names.
    count_bac: int = len(
        list(filter(lambda x: "BACTERIA;" in x.upper(), abd_tbl.columns)))
    if (count_bac / len(abd_tbl.columns)) > 0.2:
        logging.warning("""Table appears to have taxa on columns, so we have
                 transposed it.""")
        abd_tbl = abd_tbl.T

    # Remove ? from lineages, unknown should be blank
    # abd_tbl.index = list(map(lambda x: x.replace("?", ""), abd_tbl.index))

    # See if there are rank indicators in the taxa names
    # We'll take a look at the first 10 taxa, and see if they do
    rank_indicated: bool = all(map(_contain_rank_indicators,
                                   abd_tbl.index[:10]))
    if rank_indicated:
        logging.info(
            """Taxa names appear to contain rank indicators (i.e k__, p__),
            these have been removed to match Enterosignature format.""")
        abd_tbl.index = map(
            lambda x: re.sub(RE_RANK, "", x),
            abd_tbl.index
        )

    # Delete any taxa which have numeric labels (basically dealing with
    # MATAFILER output which has a -1 row included). More useful for our
    # group, limited use for other maybe.
    numeric_taxa: List[str] = list(filter(
        lambda x: str(x).lstrip('-').isnumeric(), abd_tbl.index))
    if len(numeric_taxa) > 0:
        logging.info(
            f"{len(numeric_taxa)} taxa had numeric labels and were dropped.")
        abd_tbl = abd_tbl.drop(labels=numeric_taxa)

    # Pad lineages to standard length with semicolons
    new_index: pd.Index = pd.Index(map(_pad_lineage, abd_tbl.index))
    # If there are duplicated rows (same genus, or malformed strings) then
    # report an error
    duplicates: np.ndarray = new_index.duplicated(keep=False)
    if any(duplicates):
        orig_dups: List[str] = abd_tbl.index[duplicates]
        new_dups: List[str] = new_index[duplicates]
        logging.error(
            f"{len(orig_dups)} taxa are duplicates after trimming to "
           "genus length. This may be genuine duplicates, or could be "
           "due to erroneous semi-colons. Duplicates are:")
        for o, n in zip(orig_dups, new_dups):
            logging.error(f'{o} -> {n}')
        raise EnteroException("Duplicate taxa in input after lineage " +
                              "truncated.")

    # Remove any taxa which are completely unknown (all ? or blank)
    bad_taxa: List = list(filter(_is_taxon_unknown, abd_tbl.index))
    abd_tbl = abd_tbl.drop(labels=bad_taxa)
    if len(bad_taxa) > 0:
        logging.warning(f"Removed {len(bad_taxa)} unknown taxa: {bad_taxa}")

    # Remove any taxa which had 0 observations, and samples for which 0
    # taxa were observed (unlikely but check)
    zero_taxa: List = list(abd_tbl.loc[abd_tbl.sum(axis=1) == 0].index)
    zero_samples: List = list(
        abd_tbl.loc[:, abd_tbl.sum(axis=0) == 0].columns)
    if len(zero_taxa) > 0:
        abd_tbl = abd_tbl.drop(labels=zero_taxa)
        logging.warning(
            f"Dropped {len(zero_taxa)} taxa with no observations: {zero_taxa}")
    if len(zero_samples) > 0:
        abd_tbl = abd_tbl.drop(columns=zero_samples)
        logging.warning(
            f"Dropped {len(zero_samples)} sample with no observations")

    # Renormalise (TSS)
    abd_tbl = abd_tbl / abd_tbl.sum(axis=0)

    return abd_tbl


def match_genera(
        es_w: pd.DataFrame,
        abd_tbl: pd.DataFrame,
        hard_mapping: Optional[Dict[str, str]] = None,
        family_rollup: bool = True,
        **kwargs
) -> FeatureMapping:
    """Match taxonomic names in the input table and the Enterosignatures W 
    matrix.
    
    This function is currently based on the R script provided by Clemence in
    the Enterosignatures (ES) gitlab repo (prepare_matrices.R)
    https://gitlab.inria.fr/cfrioux/enterosignature-paper/. 
    This will attempt to match names, but if any input names cannot be matched 
    to Enterosignature names, the third entry returned will be a list of names 
    to be manually corrected. Mappings in the mapping parameter are new to ES 
    names, and will be applied before any other matches identified.
    
    :param es_w: Enterosignatures W matrix
    :type es_w: pd.DataFrame
    :param abd_tbl: Abundance table being transformed
    :type abd_tbl: pd.DataFrame
    :param hard_mapping: Mapping from input to ES name
    :type hard_mapping: Dict[str, str]
    :param family_rollup: Move abundance of genera which are not matched to
        the family level entry if one exists
    :type family_rollup: bool
    :param logger: Function to log messages
    :type logger: Callable[[Any], None]
    :returns: Transformed abundance table, es W matrix, and mapping object
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, List[str]]
    """

    # Set optional parameter values
    if hard_mapping is None:
        hard_mapping = dict()
    # Sets of taxa
    es_taxa: Set[str] = set(es_w.index)
    input_taxa: Set[str] = set(abd_tbl.index)
    unmatched: Set[str] = set(abd_tbl.index)

    # Create an object to collect all the mappings we identify
    # No tables will be altered until we've exhausted the ways in which we
    # could generate mappings
    mapping: FeatureMapping = FeatureMapping(
        hard_map=hard_mapping,
        source_features=input_taxa,
        target_features=es_taxa)

    # Add any exact matches. If there was an exact match, remove from unmatched
    # pool
    exact = es_taxa.intersection(input_taxa)
    logging.info(
        f"{len(exact)} of {len(input_taxa)} taxa names matched exactly")
    for genus in exact:
        mapping.add(genus, genus)
    unmatched = unmatched.difference(exact)

    # Merge some split genera (ie Clostridia_A -> Clostridia)
    trimmed: Set[str] = set()
    for taxon in filter(lambda x: "_" in x.split(';')[-1], unmatched):
        root: str = re.sub(RE_SPLIT_GENUS, r"\1", taxon)
        if root in es_taxa:
            trimmed.add(taxon)
            mapping.add(taxon, root)
    logging.info(
        f"{len(trimmed)} genera trimmed and matched (i.e Ruminococcus_C ->"
        " Ruminococcus)")
    unmatched = unmatched.difference(trimmed)

    # Homogenise taxa of input matrix and ES
    # This is adapted directly from Clemence's script, and identifies taxa
    # where the lowest known rank matches, but the preceding elements do not
    es_taxa_short: List[str] = list(map(
        lambda x: RE_SHORTEN.sub(r"\1", x), es_taxa))
    input_taxa = set(unmatched)
    input_taxa_short: List[str] = list(map(
        lambda x: RE_SHORTEN.sub(r"\1", x), input_taxa))
    es_only: Set[str] = es_taxa.difference(input_taxa)
    input_only: Set[str] = input_taxa.difference(es_taxa)

    es_only_genus: Set[str] = set(
        map(lambda x: _shorten_genus(
            x,
            genera_list=es_taxa,
            short_genera_list=es_taxa_short
        ),
            es_only
            )
    )
    input_only_genus: Set[str] = set(
        map(lambda x: _shorten_genus(
            x,
            genera_list=input_taxa,
            short_genera_list=input_taxa_short
        ),
            input_only
            )
    )

    # Which can be merged based on Clemence's critera
    fixable_taxa: Set[str] = es_only_genus.intersection(input_only_genus)
    clemence_merged: Set[str] = set()
    for taxon in fixable_taxa:
        t_from: List[str] = list(filter(lambda x: taxon in x, unmatched))
        t_to: List[str] = list(filter(lambda x: taxon in x, es_w))
        for f in t_from:
            for t in t_to:
                mapping.add(f, t)
            clemence_merged.add(f)
    logging.info(f"{len(clemence_merged)} taxa merged by final rank (e.g. "
                 "UBA1435, CAG-314)")
    unmatched = unmatched.difference(clemence_merged)

    # Final rank matching
    final_matched: Set[str] = set()
    for taxon in unmatched:
        final_match: Iterator[str] = filter(
            lambda x: _final_rank_equal(taxon, x),
            es_taxa
        )
        for match in final_match:
            mapping.add(taxon, match)
            final_matched.add(taxon)
    logging.info(
        f"{len(final_matched)} taxa matched by name of lowest rank alone")
    unmatched = unmatched.difference(final_matched)

    # Roll any genera which are unknown, but where we have a family entry,
    # up into the family entry
    if family_rollup:
        family_match: Set[str] = set()
        for taxon in unmatched:
            family: str = ";".join(taxon.split(";")[:-1]) + ";"
            if family in es_taxa:
                family_match.add(taxon)
                mapping.add(taxon, family)
        logging.info(
            f"{len(family_match)} genera rolled up to family "
            "(i.e Lachonspiraceae;CAG-95 -> Lachnospiraceae)")
        unmatched = unmatched.difference(family_match)

    # # Get updated abundance tables and W matrix
    # new_abd = mapping.transform_abundance(abd_tbl)
    # new_w = mapping.transform_w(es_w, new_abd)
    # # Match their ordering
    # new_abd = new_abd.loc[new_w.index]
    #
    # # Summarise loss of abundance and of W weights so user can assess whether
    # # this is acceptable
    # input_unique = new_w[new_w.sum(axis=1) == 0].index
    # input_abd_missed: float = new_abd.loc[input_unique].sum().sum()
    # input_abd_missed_prop: float = input_abd_missed / new_abd.sum().sum()
    # logger(f"{len(input_unique)} taxa unique to input, could not be matched" + \
    #        f" any entries in Enterosignatures matrix.")
    # logger(f"These unique taxa represent {input_abd_missed:.2f} abundance of" + \
    #        f" a total {new_abd.sum().sum():.2f} ({input_abd_missed_prop:.2%}).")
    #
    # # Loss of W weight
    # w_total: float = es_w.sum().sum()
    # new_w_total: float = new_w.sum().sum()
    # logger(f"Sum of weights in new W matrix is {new_w_total:.2f}; " + \
    #        f"Original Enterosignatures matrix sum is {w_total:.2f}. " + \
    #        f"{1 - (new_w_total / w_total):.2%} of ES matrix weight lost due to " + \
    #        "taxa mismatch.")
    return mapping


def model_fit(
        w: pd.DataFrame,
        h: pd.DataFrame,
        x: pd.DataFrame,
        logger: Callable[[str], None] = _console_logger) -> pd.DataFrame:
    """Model fit (cosine similarity) for each sample to the model WH.
    
    Model fit is a measure of how well the input data for a sample can be 
    fit to the feature weights in the decomposition. Model fit is measured as 
    cosine similarity between taxon abundances in input matrix X, and taxon 
    abundances in model product WH. Cosine similarity ranges between 1 (good
    fit) and 0 (poor fit) for this case. While cosine similaity can be negative,
    without our non-negativity constraint it will fall in the range 0 to 1.
    
    :param w: Taxon weight matrix (taxa x ES)
    :type w: pd.DataFrame
    :param h: Sample ES weight matrix (ES x sample)
    :type h: pd.DataFrame
    :param x: Input abundances. Here this will be output of
                            match_genera
    :return: Cosine similarity between WH and X for each sample
    :type: pd.DataFrame
    """

    # Obtain product of w x h
    wh: pd.DataFrame = w.dot(h)
    # Ensure abundance and model in matching order
    x = x.loc[wh.index, wh.columns]
    # Apply across matched columns
    # We don't want pairwise just specific pairs
    # TODO(apduncan): Calculate only desired pairs, rather than full matrix
    cos_sim: pd.DataFrame = pd.DataFrame(
        np.diag(cosine_similarity(wh.T, x.T)),
        index=wh.T.index,
        columns=["model_fit"]
    )
    return cos_sim


def nmf_transform(new_abd: pd.DataFrame,
                  new_w: pd.DataFrame,
                  logger: Callable[[Any], None] = _console_logger
                  ) -> pd.DataFrame:
    """Transform the input data into Enterosignature weights.
    
    Takes the matched up W matrix and abundance tables. Expects the row 
    ordering of W and abundance table to be the same. Any NA values will be 
    filled with 0.

    :param new_abd: Genus level abundances
    :type new_abd: pd.DataFrame
    :param new_w: Model weights
    :type new_w: pd.DataFrame
    :param logger:
    :type logger: Callable[[Any], None]
    :return: Enterosignature weights for the given model and abundances,
        note this is nor relative abundance (do not sum to 1)
    :rtype: pd.DataFrame
    """
    h, _, _ = non_negative_factorization(
        new_abd.T.values,
        n_components=new_w.shape[1],
        init="custom",  # Must be custom as providing H init
        verbose=False,
        solver="mu",
        max_iter=2000,
        random_state=None,
        l1_ratio=1,
        beta_loss="kullback-leibler",
        update_H=False,
        H=new_w.T.values
    )
    h_df: pd.DataFrame = pd.DataFrame(
        h,
        index=new_abd.columns,
        columns=new_w.columns
    )
    return h_df


def to_relative(result: ReapplyResult) -> ReapplyResult:
    """Covert results to relative abundance

    The output model weights for each sample do not sum to a constant values.
    For many analyses, you may want to consider the relative weight of each
    ES in each sample, by scaling so each sample sums to 1 (total sum scaling).
    This converts the h property to a relative value for each sample.
    Note when this is applied :math:`W \\times H \\not\\approx X` except in
    the unlikely case all samples summed close to 1 anyway.

    :param result: Result object
    :type result: ReapplyResult
    :returns: New result object with h converted to relative values.
    :rtype: ReapplyResult
    """
    h_rel: pd.DataFrame = (result.h.T / result.h.T.sum()).T
    return ReapplyResult(
        w=result.w, abundance_table=result.abundance_table,
        model_fit=result.model_fit, taxon_mapping=result.taxon_mapping,
        h=h_rel
    )


def transform_table(abundance: pd.DataFrame,
                    model_w: pd.DataFrame,
                    hard_mapping: Optional[Dict[str, str]] = None,
                    rollup: bool = True,
                    relative_abundance: bool = False,
                    logger: Callable[[str], None] = _console_logger
                    ) -> 'denovo.Decomposition':
    """Transform abundance DataFrame using model W DataFrame.

    The new data must be annotated against the same taxonomy the model uses.
    Currently this is GTDB r207 for the 5 Enterosignatures model.
    Taxon names will be automatically matched between the abundance table and
    model where possible, (see :func:`cvanmf.transform.match_genera`).
    Most of the work is done in :func:`cvanmf.transform.transform_table`,
    this mostly provides convenience of allowing parameters to be paths or
    dataframes etc.

    :param abundance: Table of genus level abundances to transform
    :type abundance: pd.DataFrame
    :param model_w: W matrix of model to use
    :type model_w: pd.DataFrame
    :param hard_mapping: Define matchups between taxon identifeiers in abundance
        and those in model_w. These will be used in preference of any automated
        matches.
    :type hard_mapping: Optional[Dict[str, str]]
    :param rollup: For genera in abundance which have no match in model_w,
        sum their abundance under the family level entry where available.
    :param relative_abundance: Whether to convert weights to relative abundance
        (all samples sum to 1)
    :type relative_abundance: bool
    :type rollup: bool
    """
    # Import locally to avoid circular imports. Most of this module does not
    # need to know about the denovo structures
    from cvanmf.denovo import Decomposition, NMFParameters

    abd_tbl: pd.DataFrame = validate_table(abundance, logger=logger)
    new_abd, new_w, mapping = match_genera(
        es_w=model_w, abd_tbl=abd_tbl, logger=logger, hard_mapping=hard_mapping,
        family_rollup=rollup
    )
    es: pd.DataFrame = nmf_transform(new_abd=new_abd, new_w=new_w,
                                     logger=logger)
    mf: pd.DataFrame = model_fit(w=new_w, h=es.T, x=new_abd)
    res: ReapplyResult = ReapplyResult(
        w=new_w, h=es, abundance_table=new_abd, model_fit=mf,
        taxon_mapping=mapping)
    res = to_relative(res) if relative_abundance else res
    decomp: Decomposition = denovo.Decomposition(
        w=new_w, h=es.T, feature_mapping=mapping,
        parameters=NMFParameters(
            x=new_abd,
            rank=new_w.shape[1],
            # TODO: Properly implement seeds for refitting
            # Shouldn't be needed as no randomisation
            seed='Generator',
        )
    )
    return decomp


class FeatureMatch(Protocol):
    """Signature for functions which perform feature matching."""

    def __call__(self,
                 w: pd.DataFrame,
                 y: pd.DataFrame,
                 **kwargs) -> FeatureMapping:
        ...


class InputValidation(Protocol):
    """Signature for functions which perform input validation."""

    def __call__(self,
                 y: pd.DataFrame,
                 **kwargs) -> pd.DataFrame:
        ...


def _reapply_model(
        y: pd.DataFrame,
        w: pd.DataFrame,
        colors: Optional[List[str]],
        input_validation: InputValidation,
        feature_match: FeatureMatch,
        **kwargs
) -> 'denovo.Decomposition':
    """Reapply a model to new data Y.

    New observations using the same features can be transformed to give
    signature weights for the new observations. However, for many types of
    data is unlikely the features will be in the exact same format, and
    that the exact same types of features will be observed.

    This function uses custom functions provided to validate and transform
    input, and then to match features. Note these are applied in this order if
    you apply any transformations such a log transform in validation this may
    have undesirable effects if features are combined and summed during
    feature matching.
    """
    from cvanmf.denovo import Decomposition, NMFParameters

    new_abd: pd.DataFrame = input_validation(y, **kwargs)
    mapping: FeatureMapping = feature_match(w, new_abd, **kwargs)

    # Make W' and Y' with matching row labels
    # Get updated abundance tables and W matrix
    new_abd = mapping.transform_abundance(new_abd)
    new_w = mapping.transform_w(w, new_abd)
    # Match their ordering
    new_abd = new_abd.loc[new_w.index]

    # Summarise loss of abundance and of W weights so user can assess whether
    # this is acceptable
    input_unique = new_w[new_w.sum(axis=1) == 0].index
    input_abd_missed: float = new_abd.loc[input_unique].sum().sum()
    input_abd_missed_prop: float = input_abd_missed / new_abd.sum().sum()
    logging.info(
        f"{len(input_unique)} features unique to input, could not be "
        f"matched any entries in signature W matrix.")
    logging.info(
        f"These unique features represent {input_abd_missed:.2f} weight of"
        f" a total {new_abd.sum().sum():.2f} ({input_abd_missed_prop:.2%}).")

    # Loss of W weight
    w_total: float = w.sum().sum()
    new_w_total: float = new_w.sum().sum()
    logging.info(
        f"Sum of weights in new W matrix is {new_w_total:.2f}; "
        f"Original signature matrix sum is {w_total:.2f}. "
        f"{1 - (new_w_total / w_total):.2%} of signature W matrix weight lost "
        f"due to feature mismatch.")

    # Transform to signature weights
    es: pd.DataFrame = nmf_transform(new_abd=new_abd,
                                     new_w=new_w)
    decomp: Decomposition = Decomposition(
        w=new_w, h=es.T, feature_mapping=mapping,
        parameters=NMFParameters(
            x=new_abd,
            rank=new_w.shape[1],
            seed='Reapply',
        )
    )
    return decomp


def reapply(abundance: Union[str, pd.DataFrame],
            model_w: Union[str, pd.DataFrame] = "5es",
            hard_mapping: Optional[Union[str, pd.DataFrame]] = None,
            rollup: bool = True,
            separator: str = "\t",
            output_dir: Optional[str] = None,
            relative_abundance: bool = False
            ) -> 'denovo.Decomposition':
    """Load and transform abundances to an existing Enterosignature model.

    The new data must be annotated against the same taxonomy the model uses.
    Currently this is GTDB r207 for the 5 Enterosignature model. Taxon names
    will be automatically matched between the abundance table and model where
    possible, (see :func:`match_genera`). Most of the work is done in
    :func:`transform_table`, this mostly provides convenience of allowing
    parameters to be paths or dataframes etc.

    :param abundance: Table of genus level abundances to transform. Can be a
        string giving path, or a DataFrame.
    :type abundance: Union[str, pd.DataFrame]
    :param model_w: W matrix of model to use. Can be a DataFrame, or the path to
        load matrix from. The default is the 5ES model of Frioux et al.
        (2023, https://doi.org/10.1016/j.chom.2023.05.024). Other models,
        once available, can be loaded via the models module.
    :type model_w: Union[str, pd.DataFrame]
    :param hard_mapping: Define matchups between taxon identifiers in abundance
        and those in model_w. These will be used in preference of any automated 
        matches. Should be a table with index being abundance identifier, 
        and first column the model_w identifier. Can be either a path, or 
        DataFrame.
    :type hard_mapping: Optional[Union[str, pd.DataFrame]]
    :param rollup: For genera in abundance which have no match in model_w,
        sum their abundance under the family level entry where available.
    :type rollup: bool
    :param separator: Separator to use when reading matrices.
    :type separator: str
    :param output_dir: Directory to write results to. Directory will be created 
        if it does not exist. Pass None for no output to disk.
    :type output_dir: Optional[str]
    :param relative_abundance: Whether to convert weights to relative abundance
        (all samples sum to 1). Note that if using relative abundance,
        product of w and h no longer approximates input.
    :type relative_abundance: bool
    :returns: Multiple results, see :class:`ReapplyResult`
    :rtype: ReapplyResult
    """

    # Load files if required
    abundance_df: pd.DataFrame
    if isinstance(abundance, str):
        abundance_df = pd.read_csv(abundance, sep=separator, index_col=0)
    else:
        abundance_df = abundance

    # Load model
    w_df: pd.DataFrame
    if isinstance(model_w, str):
        # This is either a known model identifier, or a location for model
        # Check for known models
        from cvanmf import models
        if model_w == "5es":
            logging.info("Loading 5ES model")
            w_df = models.five_es()
        else:
            logging.info("Loading model from %s", model_w)
            w_df = pd.read_csv(
                model_w, sep=separator, index_col=0
            )
    else:
        w_df = model_w

    # Load hard mapping
    mapping_dict: Dict[str, str]
    if hard_mapping is not None:
        if isinstance(hard_mapping, str):
            mapping_dict = pd.read_csv(hard_mapping, sep=separator).to_dict()
        else:
            mapping_dict = hard_mapping.to_dict()
    else:
        mapping_dict = {}

    res: 'denovo.Decomposition' = transform_table(
        abundance=abundance_df,
        rollup=rollup,
        model_w=w_df,
        hard_mapping=mapping_dict,
        relative_abundance=relative_abundance
    )

    if output_dir is not None:
        res.save(out_dir=pathlib.Path(output_dir), delim=separator)

    return res


@click.command()
@click.option("-a",
              "--abundance",
              required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="""Genus level relative abundance table, with genera on rows
              and samples on columns.""")
@click.option("-m",
              "--model_w",
              required=False,
              type=click.Path(exists=True, dir_okay=False),
              help="""Enterosignature W matrix. If not provided, attempts to 
               find or download the 5 ES model matrix""")
@click.option("-h",
              "--hard_mapping",
              required=False,
              type=click.Path(exists=True, dir_okay=False),
              help="""Mapping between taxa in abdundance table and ES W matrix.
               Provide as a csv, with first column taxa in input table, second 
               column the name in ES W to map to.""")
@click.option("--rollup/--no-rollup",
              default=True,
              help="""For genera in abundance table which do not match the 
               ES W matrix, add their abundance to a family level entry if one 
               exists.""")
@click.option("-s", "--separator",
              default="\t", type=str,
              help="""Separator used in input files.""")
@click.option("-o", "--output_dir",
              required=True, type=click.Path(file_okay=False),
              help="""Directory to write output to.""")
def cli(abundance: str,
        model_w: str,
        hard_mapping: str,
        rollup: bool,
        separator: str,
        output_dir: str) -> None:
    """Command line interfrace to fit new data to an existing Enterosignatures
    model. The new data must be annotated against the same taxonomy the model
    uses. Currently this is GTDB r207 for the 5 Enterosignatures model.
    
    For more on Enterosignatures see:

    *  Frioux et al. 2023 (doi:10.1016/j.chom.2023.05.024)

    *  https://enterosignatures.quadram.ac.uk
    """

    if model_w is None:
        model_w = '5es'

    # Use transform function
    res: 'denovo.Decomposition' = reapply(
        abundance=abundance,
        model_w=model_w,
        hard_mapping=hard_mapping,
        rollup=rollup,
        separator=separator,
        output_dir=output_dir
    )
