import json
import pickle

import numpy as np
import pandas as pd


def read_metadata(metadata_path: str) -> tuple:
    """
    Read metadata from a json file (is necessary for the reprosyn generators)
    """
    with open(metadata_path, encoding="utf-8") as f:
        meta_data = json.load(f)

    categorical_cols = [
        col["name"] for col in meta_data if col["type"] == "finite"
    ]
    continous_cols = [
        col["name"] for col in meta_data if col["type"] in ("Integer", "Float")
    ]
    return meta_data, categorical_cols, continous_cols


def read_data(
    data_path: str, categorical_cols: list, continuous_cols: list
) -> pd.DataFrame:
    """
    Read given file_path (csv) and return a pd dataframe.
    If all categorical, make sure data all column values are strings
    """
    df = pd.read_csv(data_path)
    if "Person ID" in df.columns:
        df = df.drop("Person ID", axis=1)

    df[categorical_cols] = df[categorical_cols].astype(str)
    df[continuous_cols] = df[continuous_cols].astype(float)

    return df


def normalize_cont_cols(
    df: pd.DataFrame,
    meta_data: list,
    df_aux: pd.DataFrame,
    types: tuple = ("Float",),
) -> pd.DataFrame:
    norm_cols = [col["name"] for col in meta_data if col["type"] in types]

    if len(norm_cols) != 0:
        for col in norm_cols:
            df[col] = (df[col] - df_aux[col].min()) / (
                df_aux[col].max() - df_aux[col].min()
            )
    return df


def select_columns(
    df: pd.DataFrame,
    categorical_cols: list,
    continuous_cols: list,
    cols_to_select: list,
    meta_data_og: list,
) -> tuple:
    if cols_to_select[0] == "all":
        return df, categorical_cols, continuous_cols, meta_data_og
    df = df[cols_to_select]
    categorical_cols = [
        col for col in categorical_cols if col in cols_to_select
    ]
    continuous_cols = [col for col in continuous_cols if col in cols_to_select]
    meta_data_columns = [col["name"] for col in meta_data_og]
    meta_data_selected = []
    for col in cols_to_select:
        meta_data_selected.append(meta_data_og[meta_data_columns.index(col)])
    return df, categorical_cols, continuous_cols, meta_data_selected


def discretize_dataset(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Convert the dataset to one where all categories in categorical columns are integers
    instead of class name strings
    """
    value_mapping = {}
    discrete_df = df.copy()
    for column in columns:
        # Compute a mapping value -> integer.
        mapper = {v: str(i) for i, v in enumerate(sorted(df[column].unique()))}
        mapping = {str(i): v for i, v in mapper.items()}
        value_mapping[column] = mapping
        discrete_df[column] = [mapper[x] for x in df[column]]
    return discrete_df


def get_target_record(df: pd.DataFrame, index: int) -> pd.DataFrame:
    """
    Given an index, return the 1-record dataframe corresponding to the index
    """
    return df.loc[index:index]


def merge_datasets(df_secant: pd.DataFrame) -> pd.DataFrame:
    """
    Will merge the list of dataset given in entry into a global dataset
    """
    # The list of datasets has to be non-empty
    assert len(df_secant) > 0
    # All datasets need to share the same set of columns
    assert np.all(
        np.array([df_secant[0].columns == df_.columns for df_ in df_secant])
    )
    df = pd.DataFrame(columns=df_secant[0].columns)
    for df_ in df_secant:
        df.concat([df, df_], ignore_index=True)
    return df


def load_data(
    path_to_data: str, path_to_metadata: str, cols_to_select: list = ["all"]
):
    meta_data_og, categorical_cols, continuous_cols = read_metadata(
        path_to_metadata
    )
    df = read_data(path_to_data, categorical_cols, continuous_cols)
    # print(df)
    df = discretize_dataset(df, categorical_cols)
    df = normalize_cont_cols(df, meta_data_og, df_aux=df)
    df, categorical_cols, continuous_cols, meta_data = select_columns(
        df, categorical_cols, continuous_cols, cols_to_select, meta_data_og
    )
    return df, categorical_cols, continuous_cols, meta_data


def split_data(df: pd.DataFrame, path_to_ids: str):
    with open(path_to_ids, "rb") as f:
        ids = pickle.load(f)

    df_aux = df.loc[ids[1]]
    ids_eval = np.setdiff1d(df.index, ids[1])
    df_eval = df.loc[ids_eval]
    df_target = df.loc[ids[0]]

    return df_aux, df_eval, df_target
