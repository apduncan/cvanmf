# cvanmf
An implementation of bicrossvalidation for Non-negative Matrix Factorisation (NMF) rank selection, along with methods
for analysis and visualisation of NMF decomposition.

For details on the method, please see:
* Enterosignatures define common bacterial guilds in the human gut microbiome, Frioux, Clémence et al., Cell Host & Microbe, Volume 31, Issue 7, 1111 - 1125.e6 (https://doi.org/10.1016/j.chom.2023.05.024)

## Documentation
Documentation can be found at [readthedocs](https://cvanmf.readthedocs.io).

## Overview
NMF is an unsupervised machine learning techniques which provides a representation of a numeric input matrix $X$ as 
a mixture of $k$ of underlying parts. 
In this package we refer to each of these parts as a _signature_. 
Each signature can be described by how much each feature contributes to it.
For example, we can represent the abundance of bacteria in the human gut as a mixture of 5 signatures.

The number of signatures (or rank, $k$) has to specified when performing NMF, and selecting an appropriate value for 
$k$ is an important step.
We implement bicrossvalidation with Gabriel style holdouts.
Broadly speaking, this method holds out one block of the matrix ($A$) and makes an estimate of it ($A'$) using the 
remainder of the matrix.
How closely $A'$ resembles $A$ is used to identify and appropriate rank.

## Input
Any numeric matrix can be used as input, with samples on columns, and features on rows.
Each row should describe something similar, e.g. each is the abundance of a microbe, or abundance of a transcript.
A minimum of 2 samples is required.
When number of samples $n$ is close to the number of signatures $k$, signatures are likely to represent individual 
samples rather than broad patterns.

## Container
We provide a container image for linux/amd64 on through the Github Container Repository (GHCR), with the current
version being `ghcr.io/apduncan/cvanmf:latest/`.
This is intended either for running cvanmf command-line tools, or using as a container for using cvanmf within 
pipelines.
Please see the documentation for more details.

## References
If you use this tool please cite:
For details on the method, please see:
* Enterosignatures define common bacterial guilds in the human gut microbiome, Frioux, Clémence et al., Cell Host & Microbe, Volume 31, Issue 7, 1111 - 1125.e6 (https://doi.org/10.1016/j.chom.2023.05.024)

For background on NMF, see:
For background on  NMF see:
* Lee & Seung, 1999 (https://doi.org/10.1038/44565) for the paper introducing NMF
* Jiang et al, 2012 (https://doi.org/10.1007/s00285-011-0428-2) for a good description of the method and application to metagenomic data 
