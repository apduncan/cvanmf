# Text to display in the Enterosignatures page. Separated out to make flow 
# of app clearer.
TITLE = "Reapply Enterosignatures"

SUMMARY = """This tool attempts to fit genus level taxonomic abundance tables
which use the GTDB taxonomy to the 
[five Enterosignature model](https://doi.org/10.1016/j.chom.2023.05.024). 
We attempt to match up your genera to those observed in the data used to learn 
the five Enterosignatures, which allows your abundance table to be 
transformed to Enterosignature weights for each sample. 
Source for this tool is available 
[at GitHub](https://github.com/apduncan/enterosig_sl)."""

INPUT_FORMAT = """### Input Format

Input is a table of relative abundances for GTDB genera.
Other taxonomies are not currently supported. 
Taxa should be on rows, and samples on columns.
Taxa should be at genus level, and be the full lineage, separated by 
semi-colons.
The first column must be taxa lineages, the first row must be sample IDs.
An [example dataset in this format is available here](https://gitlab.inria.fr/cfrioux/enterosignature-paper/-/blob/main/data/NonWestern_dataset/nonwestern_genus_abundance_normalised.tsv).
"""

CAVEATS = """### Caveats
            
* Only GTDB r207 taxonomy is supported.
* Automated taxa matching may be inaccurate, you should check the 
mapping file in the results to ensure you are happy with the matches.
* If taxa observed in your data are very different to those 
Enterosignatures was built from, results may be unreliable.
"""

RESULTS = """### Description of Results

Results are provided as a zip containing several matrices. 
Download using the button above.
The main matrix of interest is  `h.tsv` which gives the weight of each of the 
five enterosignatures (ES_Firm, ES_Bifi...) in each of your samples. 
After this, `model_fit.tsv` describes, roughly speaking, how well each sample 
can be described by the 5 Enterosignatures model, with 0 being poor and 1 being 
good.

The other files included are used in generating `h.tsv` and `model_fit.tsv`, and 
are included for completeness. A more detailed description of them is given in 
`README.txt` in the results."""

UPLOAD_LABEL = "Upload GTDB taxonomic relative abundance table"
UPLOAD_TOOLTIP = """Taxa should be on rows, and samples on columns.
    Taxa should be at genus level, and be the full lineage, separated by 
    semi-colons.
    The first column must be taxa lineages, the first row must be sample IDs.
    We cannot currently convert from other taxonomies to GTDB."""

LOG_LABEL = "Data Processing Log"

DOWNLOAD_LABEL = "Download Enterosignatures"

WEIGHT_PLOT_TITLE = "### Enterosignature Weights"
MODELFIT_PLOT_TITLE = "### Model Fit Distribution"

EXPANDER_UPLOAD = "Upload"
EXPANDER_LOG = "Data Processing Log"
EXPANDER_RESULTS = "Results"