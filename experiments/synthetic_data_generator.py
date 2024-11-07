import argparse
import datetime
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.classifiers import drop_zero_cols, fit_classifiers, scale_features
from src.data_prep import (
    discretize_dataset,
    get_target_record,
    normalize_cont_cols,
    read_data,
    read_metadata,
    select_columns,
)
from src.feature_extractors import (
    apply_feature_extractor,
    fit_ohe,
    get_feature_extractors,
)
from src.generators import get_generator
from src.shadow_data import (
    create_shadow_training_data_membership,
    create_shadow_training_data_membership_specific,
)
from src.utils import (
    blockPrint,
    enablePrint,
    ignore_depreciation,
    str2bool,
    str2list,
)

# ---------- Data Setup ------------ #
# Read the input of the user

parser = argparse.ArgumentParser()
parser.add_argument(
    "--target_record_id",
    type=int,
    default=None,
    help="the index of the target record to be considered in the attack",
)
parser.add_argument(
    "--path_to_data",
    type=str,
    ## FOR ADULT
    # default = 'data/Adult_dataset.csv',
    ## For UK census
    default="data/2011 Census Microdata Teaching File_OG.csv",
    help="path to all original data in csv format",
)
parser.add_argument(
    "--path_to_metadata",
    type=str,
    ## FOR ADULT
    # default = 'data/Adult_metadata_discretized.json',
    ## For UK census
    default="data/2011 Census Microdata Teaching Discretized.json",
    help="path to metadata of the csv in json format",
)
parser.add_argument(
    "--cols_to_select",
    type=str2list,
    default="['all']",
    help="if not all, specify a list of cols to include in the pipeline",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./results_experiments",
    help="path to dir to save the output in csv format",
)
parser.add_argument(
    "--name_generator",
    type=str,
    default="privbayes",
    help="name of the synthetic data generator",
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=1000.0,
    help="epsilon value for DP synthetic data generator",
)
parser.add_argument(
    "--n_aux",
    type=int,
    default=50000,
    help="number of records in the auxiliary data",
)
parser.add_argument(
    "--n_test",
    type=int,
    default=25000,
    help="number of records in the test data",
)
parser.add_argument(
    "--n_original",
    type=int,
    default=1000,
    help="number of records in the original data, from which synthetic data is generated",
)
parser.add_argument(
    "--n_synthetic",
    type=int,
    default=1000,
    help="number of records in the generated synthetic dataset",
)
parser.add_argument(
    "--n_pos_train",
    type=int,
    default=500,
    help="number of shadow datasets with a positive label for training, the total number of train shadow datasets is twice this number",
)
parser.add_argument(
    "--n_pos_test",
    type=int,
    default=100,
    help="number of shadow datasets with a positive label for testing, the total number of test shadow datasets is twice this number",
)
parser.add_argument(
    "--models",
    type=list,
    default=["random_forest"],
    help="a list of strings corresponding to the model types to be used",
)
parser.add_argument(
    "--cv",
    type=str2bool,
    default="False",
    help="whether or not cross validation should be applied",
)
parser.add_argument(
    "--feat_selection",
    type=str2bool,
    default="False",
    help="whether or not feature selection in the meta model should be applied",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the random seed to be applied for reproducibility",
)
parser.add_argument(
    "--path_1000",
    type=str,
    default="",
    help="path to save the 1000 targets if specific",
)

args = parser.parse_args()

PATH_TO_DATA = args.path_to_data
PATH_TO_METADATA = args.path_to_metadata
COLS_TO_SELECT = args.cols_to_select
OUTPUT_DIR = args.output_dir
TARGET_RECORD_ID = args.target_record_id
NAME_GENERATOR = args.name_generator
EPSILON = args.epsilon
N_AUX, N_TEST = args.n_aux, args.n_test
N_ORIGINAL, N_SYNTHETIC = args.n_original, args.n_synthetic
N_POS_TRAIN, N_POS_TEST = args.n_pos_train, args.n_pos_test
MODELS = args.models
CV = args.cv
FEAT_SELECTION = args.feat_selection
SEED = args.seed
PATH_TEST = args.path_1000
if PATH_TEST == "":
    SPECIFIC = False
else:
    SPECIFIC = True

# set the seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def main():
    start_time = time.time()
    # read data
    print("Reading and preparing the data...")
    meta_data_og, categorical_cols, continuous_cols = read_metadata(
        PATH_TO_METADATA
    )
    df = read_data(PATH_TO_DATA, categorical_cols, continuous_cols)
    df = discretize_dataset(df, categorical_cols)
    df = normalize_cont_cols(df, meta_data_og, df_aux=df)
    df, categorical_cols, continuous_cols, meta_data = select_columns(
        df, categorical_cols, continuous_cols, COLS_TO_SELECT, meta_data_og
    )

    # get a target record, for now just by selecting an index
    target_record = get_target_record(df, TARGET_RECORD_ID)

    # split data into auxiliary and test set
    with open(PATH_TEST, "rb") as f:
        indexs = pickle.load(f)

    # indices for D_target, with target record removed
    indices_target = indexs[0]
    indices_target.remove(TARGET_RECORD_ID)
    df_target = df.loc[indices_target]  # 999

    # indices for D_aux, used to train the MIA
    indices_train = indexs[1]
    df_aux = df.loc[indices_train]

    # indices for D_eval, for average evaluation setup, with target record removed
    indices_eval = [ind for ind in df.index if ind not in indices_train]
    indices_eval.remove(TARGET_RECORD_ID)
    df_eval = df.loc[indices_eval]

    # checks
    assert TARGET_RECORD_ID not in df_aux.index
    assert TARGET_RECORD_ID not in df_eval.index

    # specify a generator
    generator = get_generator(NAME_GENERATOR, epsilon=EPSILON)

    print("Creating shadow datasets...")
    blockPrint()

    # Prepare shadow datasets to train MIA
    train_seeds = list(range(N_POS_TRAIN * 2))

    (datasets_train, labels_train, _) = create_shadow_training_data_membership(
        df=df_aux,
        meta_data=meta_data,
        target_record=target_record,
        generator=generator,
        n_original=N_ORIGINAL,
        n_synth=N_SYNTHETIC,
        n_pos=N_POS_TRAIN,
        seeds=train_seeds,
    )

    # Prepare evaluation datasets to evaluate MIA
    test_seeds = list(
        range(N_POS_TRAIN * 2, N_POS_TRAIN * 2 + N_POS_TEST * 2)
    )  # make it non overlapping

    enablePrint()
    print("Generating evaluation datasets (specific setup)")
    blockPrint()

    (
        datasets_test_specific,
        labels_test_specific,
    ) = create_shadow_training_data_membership_specific(
        df_sub=df_target,
        meta_data=meta_data,
        target_record=target_record,
        generator=generator,
        n_original=N_ORIGINAL,
        n_synth=N_SYNTHETIC,
        n_pos=N_POS_TEST,
        seeds=test_seeds,
        df_test=df_eval,
    )

    enablePrint()
    print("Generating evaluation datasets (average setup)")
    blockPrint()
    (
        datasets_test_any,
        labels_test_any,
        _,
    ) = create_shadow_training_data_membership(
        df=df_eval,
        meta_data=meta_data,
        target_record=target_record,
        generator=generator,
        n_original=N_ORIGINAL,
        n_synth=N_SYNTHETIC,
        n_pos=N_POS_TEST,
        seeds=test_seeds,
    )

    enablePrint()

    # Fit one hot encoding for meta-classifier data
    ohe, ohe_column_names = fit_ohe(df_aux, categorical_cols, meta_data)

    # Query-based feature extraction
    print("Running query-based attack...")
    QUERY_FEATURE_EXTRACTORS = [
        (
            "query",
            range(1, df.shape[1] + 1),
            1e6,
            {"categorical": (1,), "continuous": (3,)},
        )
    ]

    feature_extractors, do_ohe = get_feature_extractors(
        QUERY_FEATURE_EXTRACTORS
    )
    ignore_depreciation()
    print("Preparing training data...")
    X_train, y_train = apply_feature_extractor(
        datasets=datasets_train.copy(),
        target_record=target_record,
        labels=labels_train,
        ohe=ohe,
        ohe_columns=categorical_cols,
        ohe_column_names=ohe_column_names,
        continuous_cols=continuous_cols,
        feature_extractors=feature_extractors,
        do_ohe=do_ohe,
    )
    print("Preparing test data (specific)...")

    X_test_specific, y_test_specific = apply_feature_extractor(
        datasets=datasets_test_specific.copy(),
        target_record=target_record,
        labels=labels_test_specific,
        ohe=ohe,
        ohe_columns=categorical_cols,
        ohe_column_names=ohe_column_names,
        continuous_cols=continuous_cols,
        feature_extractors=feature_extractors,
        do_ohe=do_ohe,
    )

    print("Preparing test data (average)...")

    X_test_any, y_test_any = apply_feature_extractor(
        datasets=datasets_test_any.copy(),
        target_record=target_record,
        labels=labels_test_any,
        ohe=ohe,
        ohe_columns=categorical_cols,
        ohe_column_names=ohe_column_names,
        continuous_cols=continuous_cols,
        feature_extractors=feature_extractors,
        do_ohe=do_ohe,
    )

    X_train, X_test_any, X_test_specific = drop_zero_cols(
        X_train, X_test_any, X_test_specific
    )
    X_train, X_test_any, X_test_specific = scale_features(
        X_train, X_test_any, X_test_specific
    )

    print("Data prepared")

    trained_models = fit_classifiers(X_train, y_train, cv=CV, models=MODELS)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    SPEC_PREDS_OUTPUT_PATH = (
        f"{OUTPUT_DIR}/{TARGET_RECORD_ID}_specific_preds.pkl"
    )
    SPEC_LABELS_OUTPUT_PATH = (
        f"{OUTPUT_DIR}/{TARGET_RECORD_ID}_specific_labels.pkl"
    )

    TRAIN_PREDS_OUTPUT_PATH = f"{OUTPUT_DIR}/{TARGET_RECORD_ID}_train_preds.pkl"
    TRAIN_LABELS_OUTPUT_PATH = (
        f"{OUTPUT_DIR}/{TARGET_RECORD_ID}_train_labels.pkl"
    )

    for i, model in enumerate(trained_models):
        specific_test_preds = model.predict_proba(X_test_specific)
        train_preds = model.predict_proba(X_train)

        ANY_PREDS_OUTPUT_PATH = f"{OUTPUT_DIR}/{TARGET_RECORD_ID}_any_preds.pkl"
        ANY_LABELS_OUTPUT_PATH = (
            f"{OUTPUT_DIR}/{TARGET_RECORD_ID}_any_labels.pkl"
        )
        any_test_preds = model.predict_proba(X_test_any)
        with open(ANY_PREDS_OUTPUT_PATH, "wb") as f:
            pickle.dump(any_test_preds, f)
        with open(ANY_LABELS_OUTPUT_PATH, "wb") as f:
            pickle.dump(y_test_any, f)

        with open(SPEC_PREDS_OUTPUT_PATH, "wb") as f:
            pickle.dump(specific_test_preds, f)
        with open(SPEC_LABELS_OUTPUT_PATH, "wb") as f:
            pickle.dump(y_test_specific, f)

        with open(TRAIN_PREDS_OUTPUT_PATH, "wb") as f:
            pickle.dump(train_preds, f)
        with open(TRAIN_LABELS_OUTPUT_PATH, "wb") as f:
            pickle.dump(y_train, f)

    end_time = time.time()
    print(f"Time needed for target {TARGET_RECORD_ID} : {start_time-end_time}")
    print("success!")


if __name__ == "__main__":
    main()
