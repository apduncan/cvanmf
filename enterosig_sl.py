"""
# Enterosignature Transformer
Transform GTDB taxonomic abundance tables to Enterosignature weights
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from entero_process import (EnteroException, validate_table, match_genera, 
                            nmf_transform)

# Constants for resource locations etc
ES_W_MATRIX: str = "data/ES5_W.tsv"

st.title("Enterosignature Transformer")

st.markdown("""This tool attempts to generate 
[Enterosignatures](https://doi.org/10.1016/j.chom.2023.05.024) from tables of 
relative taxonomic abundance which use the GTDB taxonomy as genus 
level. We attempt to match up your genera to those used in the 
Enterosignatures matrix, which allows your abundance table to be 
transformed to Enterosignature weights for each sample.

## Caveats

* Only GTDB r207 taxonomy is supported.
* Automated taxa matching may be inaccurate, you should check the 
mapping file to ensure you are happy with the matches.
* If taxa observed in your data are very different to those 
Enterosignatures was built from, results may be unreliable.
""")

# Removed text relating to command line as not imlemented
# ## Offline Use

# If you wanted to use this as part of a pipeline, a command line version of
# this same process is available in `entero_process.py` the GitHub repo.
            
abd_file = st.file_uploader(
    label = "Upload GTDB taxonomic relative abundance table",
    help = """Taxa should be on rows, and samples on columns.
    Taxa should be at genus level, and be the full lineage, separated by 
    semi-colons.
    The first column must be taxa lineages, the first row must be sample IDs.
    We cannot currently convert from other taxonomies to GTDB.
    We will do our best to match genera between GTDB versions, but any 
    unresolved mismatches will be presented for manual correction."""
)

if abd_file is not None:
    try:
        # Attempt to transform the data and return Enterosignatures
        # Any step which fails due to some problem with the data should
        # raise an EnteroException
        abd_tbl = pd.read_csv(abd_file, sep="\t", index_col=0)

        with st.expander("Data Processing Log", expanded=True):
            # Sanity check this abundance table
            abd_tbl = validate_table(abd_tbl, logger=st.write)

            # Attempt to match the genera
            es_w: pd.DataFrame = pd.read_csv(ES_W_MATRIX, index_col=0, sep="\t")
            new_abd, new_w, mapping  = match_genera(
                es_w,
                abd_tbl,
                logger = st.write
            )

            map_df = mapping.to_df().to_csv().encode("utf-8")
            st.download_button(label = "Download genus mapping",
                               data = map_df,
                               file_name = "es_mapping.csv")
        
        # Data has been transformed
        # TODO(apduncan): Allow custom mapping files to be provided
        # TODO(apduncan): UI for fixing mappings

        # Apply NMF with the new data - simple stuff now
        es: pd.DataFrame = nmf_transform(new_abd, new_w, st.write)
        
        st.write("## Enterosignatures")
        st.download_button(
            label = "Download Enterosignatures",
            data = es.to_csv().encode("utf-8"),
            file_name = "enterosignatures.csv"
        )
        # Provide a simple visualisations of the ES
        es_t: pd.DataFrame = es.T
        p_hmap: go.Figure = go.Figure(
            go.Heatmap(
                z = es_t.values,
                x = es_t.columns,
                y = es_t.index
            )
        )
        st.plotly_chart(
            p_hmap
        )
    except EnteroException as err:
        st.write(f"Unable to transform: {err}")
