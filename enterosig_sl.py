"""
# Enterosignature Transformer
Transform GTDB taxonomic abundance tables to Enterosignature weights
"""

from typing import Collection, Tuple, Union
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import zipfile as zf
import io

from entero_process import (EnteroException, validate_table, match_genera, 
                            nmf_transform, model_fit)

def _zip_items(items: Collection[Tuple[str, str]]
               ) -> io.BytesIO:
    # Create an in-memory zip file from a collection of objects, so it can be
    # offered to a user for download
    # Using suggestions from https://stackoverflow.com/questions/2463770/
    # python-in-memory-zip-library
    zbuffer: io.BytesIO = io.BytesIO()
    with zf.ZipFile(zbuffer, "a", zf.ZIP_DEFLATED, False) as z:
        for name, contents in items:
            z.writestr(name, contents)
        # Always add the readme
        z.write("data/README.txt", "README.txt")
    return zbuffer

# Constants for resource locations etc
ES_W_MATRIX: str = "data/ES5_W.tsv"

st.title("Reapply Enterosignatures")

st.markdown("""This tool attempts to fit genus level taxonomic abundance tables
which use the GTDB taxonomy to the 
[five Enterosignature model](https://doi.org/10.1016/j.chom.2023.05.024). 
We attempt to match up your genera to those observed in the data used to learn 
the five Enterosignatures, which allows your abundance table to be 
transformed to Enterosignature weights for each sample.
            
The output is the unnormalised weight for the five Enterosignatures in 
each sample.

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

        # Data has been transformed
        # TODO(apduncan): Allow custom mapping files to be provided
        # TODO(apduncan): UI for fixing mappings

        # Apply NMF with the new data - simple stuff now
        es: pd.DataFrame = nmf_transform(new_abd, new_w, st.write)

        # Model fit in the Enterosignatures paper is the cosine similarity
        # between genus abundance and model
        # So for sample i, between (WH)_i and (N_w)_i
        mf: pd.DataFrame = model_fit(new_w, es.T, new_abd, st.write)

        # TODO(apduncan): Sort out proper Streamlit caching
        # Zip up new W, new abundance, new H (enterosig weights), and model fit
        res_zip: io.BytesIO = _zip_items([
            ("w.tsv", new_w.to_csv(sep="\t")),
            ("h.tsv", es.to_csv(sep="\t")),
            ("abundance.tsv", new_abd.to_csv(sep="\t")),
            ("model_fit.tsv", mf.to_csv(sep="\t")),
            ("taxon_mapping.tsv", mapping.to_df().to_csv(sep="\t"))
        ])

        # TODO(apduncan): Add logger output to the zip
        st.download_button(
            label = "Download Enterosignatures",
            data = res_zip,
            file_name = "apply_es_results.zip"
        )

        st.markdown("### Enterosignature Weights")

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

        # Provide a simple visualisation of the model fit
        st.markdown("### Model Fit Distribution")
        p_hist: go.Figure = px.histogram(mf, x="model_fit")
        st.plotly_chart(
            p_hist
        )

    except EnteroException as err:
        st.write(f"Unable to transform: {err}")
