import argparse
import sys
from typing import (Any, Callable, Collection, Dict, Iterator, List, 
                    Optional, Set, Tuple)
import re
from sklearn.decomposition import non_negative_factorization
import numpy as np
import pandas as pd

# Compile regular expressions
RE_RANK: re.Pattern = re.compile("[a-zA-Z]__")
RE_SPLIT_GENUS: re.Pattern = re.compile(r"([\w]+)_\w$")
# Match the last named rank, plus any following ?s
RE_SHORTEN: re.Pattern = re.compile(r".*;([^\;?]+[;\?]*)$")

class EnteroException(Exception):
    """Exception raised when unable to proceed with enterosignature 
    transformation"""

class GenusMapping():
    """Manage the mappings from input genus to ES genus."""

    def __init__(self, 
                 target_taxa: Set[str],
                 source_taxa: Set[str],
                 hard_map: Optional[Dict[str, str]] = {}):
        # hard_map are user provided mappings which will never be altered
        self.__target_taxa = target_taxa
        self.__source_taxa = source_taxa
        self.__hard_map = {} if hard_map is None else hard_map
        self.__map: Dict[str, List[str]] = dict()

    def add(self, genus_from: str, genus_to: str) -> None:
        """Add a mapping. If there is already a mapping from this genus, we 
        will append this one. Use the conflits() methods to resolve where 
        more than one mapping exists"""
        # If user has provided a hard mapping, ignore this mapping
        if genus_from in self.__hard_map:
            return
        # If the genus being mapped to is not valid, raise an exception
        if genus_to not in self.__target_taxa:
            raise EnteroException(f"Mapping of '{genus_from}' to target " + \
                f"'{genus_to}' not possible as target does not exist.")
        if genus_from not in self.__source_taxa:
             raise EnteroException(f"Mapping of '{genus_from}' to target " + \
                f"'{genus_to}' not possible as source does not exist.")
        if genus_from not in self.__map:
            self.__map[genus_from] = [genus_to]
        else:
            self.__map[genus_from].append(genus_to)

    def to_df(self) -> pd.DataFrame:
        """Write the mapping used to TSV format. Where mappings are amibiguous, 
        multiple rows will be included. Where mappings are missing, on row with 
        a blank target will be included."""
        
        # Could be more nicely done with some maps, but take a simple appraoch
        df_src: List[Tuple[str, str]] = []
        for source, targets in self.mapping.items():
            for target in targets:
                df_src.append((source, target))
        missing: List[Tuple[str, str]] = list(
            zip(self.missing(), [''] * len(self.missing())))
        df_src = df_src + missing
        df: pd.DataFrame = pd.DataFrame(df_src,
                                        columns=['input_genus', 'es_genus'])
        return df

    def missing(self) -> Collection[str]:
        """Identify input taxa which currently have no mapping"""
        return self.__source_taxa.difference(set(self.mapping.keys()))

    def transform_abundance(self, abd_tbl: pd.DataFrame) -> pd.DataFrame:
        """Make an abundance table with renamed and combined rows based on the 
        identified mappings."""
        # Probably we could do some smarter changes and groupbys
        # TODO(apduncan): Refactor to be less ugly
        for input_taxon, es_maps in self.mapping.items():
            # Where there are conflicting mappings, use the first one
            es_map = es_maps[0]
            if input_taxon == es_map:
                continue
            # Add or sum
            if es_map not in abd_tbl.index:
                abd_tbl.loc[es_map] = abd_tbl.loc[input_taxon]
            else:
                abd_tbl.loc[es_map] += abd_tbl.loc[input_taxon]
            abd_tbl = abd_tbl.drop(labels=[input_taxon])
        return abd_tbl

    def transform_w(self, 
                    w: pd.DataFrame, 
                    abd_tbl: pd.DataFrame) -> pd.DataFrame:
        """Make a W matrix which has rows added for missing taxa."""
        w_new: pd.DataFrame = w.copy()
        # Drop taxa which are not in the input matrix from the W matrix
        w_tax: Set[str] = set(w_new.index)
        w_new = w_new.drop(labels=w_tax.difference(set(abd_tbl.index)))
        # Add taxa which are in the input but not in the W matrix
        missing_taxa: Set[str] = set(abd_tbl.index).difference(w_tax)
        missing_df: pd.DataFrame = pd.DataFrame(
            np.zeros(shape=[len(missing_taxa), w.shape[1]]),
            index = list(missing_taxa),
            columns = w.columns
        )
        w_new = pd.concat([w_new, missing_df])
        return w_new

    @property
    def conflicts(self) -> List[Tuple[str, List[str]]]:
        """Genera for which more than one target exists"""
        return list(filter(lambda x: len(x[1]) > 1, self.mapping.items()))

    @property
    def mapping(self) -> Dict[str, List[str]]:
        return self.__map

def _console_logger(message: Any) -> None:
    # Log messages to stderr
    print(message, file=sys.stderr)

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
    # I'm not entirely sure of the logic behind this function, and I would like 
    # to recheck it
    # TODO(apduncan): Ensure output of this method matches R implementation
    # Count occurences in the genera list
    # Clemence's script handles named with and without trailing ? separately
    # but I think we obtain the same results with an in
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

def validate_table(abd_tbl: pd.DataFrame,
                   logger: Callable[[Any], None] = _console_logger
                   ) -> pd.DataFrame:
    """Basic checks and transformations of the abundance table. 
    
    Some transformations may be made here, such as transposition. Any 
    transformation will be written out to inform the user. Transformations are 
    done in place.
    
    :param pd.DataFrame abd_tbl: Abundance table to check
    :param Callable[[str], None]: Function to report errors
    :returns: Validated, potentially transformed, dataframe
    :rtype: pd.DataFrame
    """
    # Check the dimensions make sense
    if abd_tbl.shape[0] < 2 or abd_tbl.shape[1] < 2:
        logger("""Table has one or fewer columns or rows. Check delimiters and 
                 newline formats are correct.""")
        raise EnteroException("Table incorrect format.")

    # Check that there were column names in the file. We're going to assume
    # that if column names are all numeric values, there were no headers.
    # Unfortunate for people who only gave their samples numbers, but alas.
    all_numeric: bool = all(map(lambda x: str(x).isnumeric(), abd_tbl.columns))
    if all_numeric:
        logger("""Table appear to lack sample IDs in the first row. Add
                 sample IDs, or ensure all sample IDs are not numeric.""")
        raise EnteroException("Table lacks sample IDs in first row.")

    # Check that taxa are on rows. We'll do that by looking for "Bacteria;"
    # in the column names.
    count_bac: int = len(
        list(filter(lambda x: "BACTERIA;" in x.upper(), abd_tbl.columns)))
    if (count_bac / len(abd_tbl.columns)) > 0.2:
        logger("""Table appears to have taxa on columns, so we have
                 transposed it.""")
        abd_tbl = abd_tbl.T
    
    # Remove ? from lineages, unknown should be blank
    abd_tbl.index = list(map(lambda x: x.replace("?", ""), abd_tbl.index))

    # See if there are rank indicators in the taxa names
    # We'll take a look at the first 10 taxa, and see if they do
    rank_indicated: bool = all(map(_contain_rank_indicators,
                               abd_tbl.index[:10]))
    if rank_indicated:
        logger("""Taxa names appear to contain rank indicators (i.e k__, p__),
               these have been removed to match Enterosignature format.""")
        abd_tbl.index = map(
            lambda x: re.sub(RE_RANK, "", x),
            abd_tbl.index
        )

    # TODO(apduncan): Pad lineages to standard length with semicolons

    # Remove any taxa which are completely unknown (all ? or blank)
    bad_taxa: List = list(filter(_is_taxon_unknown, abd_tbl.index))
    abd_tbl = abd_tbl.drop(labels = bad_taxa)
    if len(bad_taxa) > 0:
        logger(f"Removed {len(bad_taxa)} unknown taxa: {bad_taxa}")

    # Remove and taxa which had 0 observations, and samples for which 0
    # taxa were observed (unlikely but check)
    zero_taxa: List = list(abd_tbl.loc[abd_tbl.sum(axis = 1) == 0].index)
    zero_samples: List = list(
        abd_tbl.loc[:, abd_tbl.sum(axis = 0) == 0].columns)
    if len(zero_taxa) > 0:
        abd_tbl = abd_tbl.drop(labels = zero_taxa)
        logger(
            f"Dropped {len(zero_taxa)} taxa with no observations: {zero_taxa}")
    if len(zero_samples) > 0:
        abd_tbl = abd_tbl.drop(columns = zero_samples)
        logger(
            f"Dropped {len(zero_samples)} taxa with no observations: " + \
            f"{zero_samples}")

    # Renormalise (TSS)
    abd_tbl = abd_tbl / abd_tbl.sum(axis=0)

    return abd_tbl

def match_genera(
        es_w: pd.DataFrame,
        abd_tbl: pd.DataFrame,
        hard_mapping: Optional[Dict[str, str]] = {},
        family_rollup: bool = True,
        logger: Callable[[Any], None] = _console_logger
        ) -> Tuple[pd.DataFrame, pd.DataFrame, GenusMapping]:
    """Match taxonomic names in the input table and the Enterosignatures W 
    matrix.
    
    This function is currently based on the R script provided by Clemence in 
    the Enterosignatures (ES) gitlab repo (prepare_matrices.R)
    https://gitlab.inria.fr/cfrioux/enterosignature-paper/. 
    This will attempt to match names, but if any input names cannot be matched 
    to Enterosignature names, the third entry returned will be a list of names 
    to be manually corrected. Mappings in the mapping parameter are new to ES 
    names, and will be applied before any other matches identified.
    
    :param pd.DataFrame es_w: Enterosignatures W matrix
    :param pd.DataFrame abd_tbl: Abundance table being transformed
    :mapping Dict[str, str] mapping: Mapping from input to ES name
    :logger Callable[[Any], None]: Function to log messages
    :returns: Transformed ES W matrix, abundace table, and unmatched names
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, List[str]]
    """

    # Sets of taxa
    es_taxa: Set[str] = set(es_w.index)
    input_taxa: Set[str] = set(abd_tbl.index)
    unmatched: Set[str] = set(abd_tbl.index)

    # Create an object to collect all the mappings we identify
    # No tables will be altered until we've exhausted the ways in which we 
    # could generate mappings
    mapping: GenusMapping = GenusMapping(
        hard_map=hard_mapping,
        source_taxa=input_taxa,
        target_taxa=es_taxa)

    # Add any exact matches. If there was an exact match, remove from unmatched
    # pool
    exact = es_taxa.intersection(input_taxa)
    logger(f"{len(exact)} of {len(input_taxa)} taxa names matched exactly")
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
    logger(f"{len(trimmed)} genera trimmed and matched (i.e Ruminococcus_C ->" + \
           " Ruminococcus)")
    unmatched = unmatched.difference(trimmed)

    # Homogenise taxa of input matrix and ES
    # This is adapted directly from Clemence's script, and identifies taxa 
    # where the lowest known rank matches, but the preceding elements do not
    es_taxa_short: List[str]    = list(map(
        lambda x: RE_SHORTEN.sub(r"\1", x), es_taxa))
    input_taxa                  = set(unmatched)
    input_taxa_short: List[str] = list(map(
        lambda x: RE_SHORTEN.sub(r"\1", x), input_taxa))
    es_only: Set[str]           = es_taxa.difference(input_taxa)
    input_only: Set[str]        = input_taxa.difference(es_taxa)

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
        t_from: List[str]   = list(filter(lambda x: taxon in x, unmatched))
        t_to: List[str]     = list(filter(lambda x: taxon in x, es_w))
        for f in t_from:
            for t in t_to:
                mapping.add(f, t)
            clemence_merged.add(f)
    logger(f"{len(clemence_merged)} taxa merged by final rank (e.g. " + \
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
    logger(f"{len(final_matched)} taxa matched by name of lowest rank alone")
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
        logger(f"{len(family_match)} genera rolled up to family " + \
               "(i.e Lachonspiraceae;CAG-95 -> Lachnospiraceae)")
        unmatched = unmatched.difference(family_match)

    # Get updated abundance tables and W matrix
    new_abd = mapping.transform_abundance(abd_tbl)
    new_w = mapping.transform_w(es_w, new_abd)
    # Match their ordering
    new_abd = new_abd.loc[new_w.index]

    # Summarise loss of abundance and of W weights so user can assess whether 
    # this is acceptable
    input_unique = new_w[new_w.sum(axis=1) == 0].index
    input_abd_missed: float = new_abd.loc[input_unique].sum().sum()
    input_abd_missed_prop: float = input_abd_missed / new_abd.sum().sum()
    logger(f"{len(input_unique)} taxa unique to input, could not be matched" + \
           f" any entries in Enterosignatures matrix.")
    logger(f"These unique taxa represent {input_abd_missed:.2f} abundance of" + \
           f" a total {new_abd.sum().sum():.2f} ({input_abd_missed_prop:.2%}).")
    
    # Loss of W weight
    w_total: float = es_w.sum().sum()
    new_w_total: float = new_w.sum().sum()
    logger(f"Sum of weights in new W matrix is {new_w_total:.2f}; " + \
           f"Original Enterosignatures matrix sum is {w_total:.2f}. " + \
           f"{1 - (new_w_total / w_total):.2%} of ES matrix weight lost due to " + \
           "taxa mismatch.")
    return (new_abd, new_w, mapping)

def nmf_transform(new_abd: pd.DataFrame,
                  new_w: pd.DataFrame,
                  logger: Callable[[Any], None] = _console_logger
                  ) -> pd.DataFrame:
    """Transform the input data into Enterosignature weights.
    
    Takes the matched up W matrix and abundance tables. Expects the row 
    ordering of W and abundance table to be the same. Any NA values will be 
    filled with 0."""
    h, _, _ = non_negative_factorization(
        new_abd.T.values,
        n_components    = new_w.shape[1],
        init            = "custom",  # Must be custom as providing H init
        verbose         = False,
        solver          = "mu",
        max_iter        = 2000,
        random_state    = None,
        l1_ratio        = 1,
        beta_loss       = "kullback-leibler",
        update_H        = False,
        H               = new_w.T.values
    )
    h_df: pd.DataFrame = pd.DataFrame(
        h,
        index   = new_abd.columns,
        columns = new_w.columns
    )
    return h_df

if __name__ == "__main__":
    # Act as a standalone command line tool
    raise Exception("Command line not implemented")