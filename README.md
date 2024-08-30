# cvanmf
An implementation of bicrossvalidation for Non-negative Matrix Factorisation (NMF) rank selection, along with methods
for analysis and visualisation of NMF decomposition.

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