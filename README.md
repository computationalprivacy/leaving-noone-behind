# Leaving Noone Behind

This library is made to make accurately estimating the privacy risk for records used to train synthetic datasets easy and clear. It contains methods for fast selection of high risk records and membership inference attacks against SDGs. The library is compatible with all SDGs implemented in the [reprosyn](https://github.com/alan-turing-institute/reprosyn) repository.

This library is based on work presented in the following papers:
1. [Lost in the Averages: : A New Specific Setup to Evaluate Membership Inference Attacks Against Machine Learning Models](https://arxiv.org/abs/2405.15423)
2. [Achilles' Heels: Vulnerable Record Identification in Synthetic Data Publishing](https://arxiv.org/abs/2306.10308)

## Installing the environment

To replicate our conda environment, run the following commands:

1. Create and activate the end:
    - `conda create --name mia_env python=3.9`
    - `conda activate mia_env`

2. Clone and install the requirements from the [reprosyn](https://github.com/alan-turing-institute/reprosyn) repository:
    - `git clone https://github.com/alan-turing-institute/reprosyn`
    - `cd reprosyn`
    - `curl -sSL https://install.python-poetry.org | python3 -`
    - `poetry install -E ektelo`. To install poetry on your system we refer to their [installation instructions](https://python-poetry.org/docs/#installing-with-the-official-installer)

3. Install the C-based optimized QBS:
    - `cd src/optimized_qbs/`
    - `python setup.py install`

4. Install remaining dependencies:
    - `pip install -r requirements.txt`

## Getting started

See our intro notebook to understand how to get started using our library!