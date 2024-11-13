# Leaving Noone Behind

This library is made to make accurately estimating the privacy risk for records used to train synthetic datasets easy and clear. It contains methods for fast selection of high risk records and membership inference attacks against SDGs. The library is compatible with all SDGs implemented in the [reprosyn](https://github.com/alan-turing-institute/reprosyn) repository.

This library is based on work presented in the following papers:
1. [Lost in the Averages: A New Specific Setup to Evaluate Membership Inference Attacks Against Machine Learning Models](https://arxiv.org/abs/2405.15423)
2. [Achilles' Heels: Vulnerable Record Identification in Synthetic Data Publishing](https://arxiv.org/abs/2306.10308)

This repository also relies on [QuerySnout](https://github.com/computationalprivacy/querysnout), the repository containing the code for the paper ["QuerySnout: Automating the Discovery of Attribute Inference Attacks against Query-Based Systems"](https://dl.acm.org/doi/abs/10.1145/3548606.3560581) by Ana-Maria Cretu, Florimond Houssiau, Antoine Cully and Yves-Alexandre de Montjoye, published at ACM CCS 2022. We have included the necessary files, slightly modified for easier installation, in this repository.

## Installing the environment

To replicate our conda environment, we have provided a Makefile with predefined shortcuts for easy installation. To install all necessary packages and a conda environment and kernel, simply run the following command in your terminal:

`make setup-env`

## Getting started

See our [intro notebook](https://github.com/computationalprivacy/leaving-noone-behind/blob/master/notebooks/example.ipynb) to understand how to get started using our library!