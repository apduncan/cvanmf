# enterosig_sl
Streamlit app to generate Enterosignature weights from GTDB abundance tables.
`entero_process.py` can be used a standalone command line script to generate 
Enterosignature weights, and does not import the streamlit module.

Currently only the 5 Enterosignatures module from Frioux et al. 2023 
(doi:10.1016/j.chom.2023.05.024) is provided, but the commandline is 
able to take other matrices.

The streamlit app is available at https://enterosignatures.streamlit.app

There are some features yet to be added:
* Toggle for family rollup option in Streamlit
* Option to upload hard mapping file in Streamlit