import asyncio

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from lnb.feature_extractors import apply_ohe, fit_ohe


def top_n_vulnerable_records(distances: dict, n: int) -> list:
    """
    Return the ids of the top n vulnerable records
    """
    top_n = {
        k: v
        for k, v in sorted(
            distances.items(), key=lambda item: item[1], reverse=True
        )[0:n]
    }
    return list(top_n.keys())


def top_n_vulnerable_dists(distances: dict, n: int) -> list:
    """
    Return the Achilles scores of the top n vulnerable records
    """
    top_n = {
        k: v
        for k, v in sorted(
            distances.items(), key=lambda item: item[1], reverse=True
        )[0:n]
    }
    return list(top_n.values())


async def compute_achilles_one_record(
    df: pd.DataFrame,
    record_id: int,
    ohe_cat_indices: list,
    n_cat_cols: int,
    cont_indices: list,
    n_cont_cols: int,
    all_distances: dict,
    n_to_save: int,
):
    record = df.loc[record_id].values
    values = df.values

    # calculate distance for categorical features
    cat_dist = (
        1
        - cosine_similarity(
            record[ohe_cat_indices].reshape(1, -1), values[:, ohe_cat_indices]
        ).flatten()
    )
    cat_dist = [n_cat_cols / (n_cat_cols + n_cont_cols) * k for k in cat_dist]

    # if there are only categorical columns, we can return this
    if n_cont_cols == 0:
        return cat_dist

    # calculate distance for continuous features
    cont_dist = (
        1
        - cosine_similarity(
            record[cont_indices].reshape(1, -1), values[:, cont_indices]
        ).flatten()
    )
    cont_dist = [
        n_cont_cols / (n_cat_cols + n_cont_cols) * k for k in cont_dist
    ]

    # finally, return the weighted average, weighted by number of respective cols
    mean_dist = np.mean(
        np.sort([cat_dist[i] + cont_dist[i] for i in range(len(cont_dist))])[
            :n_to_save
        ]
    )
    all_distances[record_id] = mean_dist
    # print(all_distances[record_id])
    return None


async def compute_achilles_parallel(
    df: pd.DataFrame,
    categorical_cols: list,
    continuous_cols: list,
    meta_data: list,
    n_to_save: int,
):
    ohe, ohe_column_names = fit_ohe(df, categorical_cols, meta_data)
    df_ohe = apply_ohe(
        df.copy(), ohe, categorical_cols, ohe_column_names, continuous_cols
    )

    all_columns = list(df_ohe.columns)
    ohe_cat_indices = [all_columns.index(col) for col in ohe_column_names]
    cont_indices = [all_columns.index(col) for col in continuous_cols]
    n_cat_cols = len(categorical_cols)
    n_cont_cols = len(continuous_cols)

    all_distances = dict()
    n_to_save = n_to_save

    tasks = list()

    print("creating tasks")
    for i, r in tqdm(enumerate(df.index)):
        tasks.append(
            asyncio.create_task(
                compute_achilles_one_record(
                    df_ohe,
                    r,
                    ohe_cat_indices,
                    n_cat_cols,
                    cont_indices,
                    n_cont_cols,
                    all_distances,
                    n_to_save,
                )
            )
        )

    print("computing achilles")
    for i in tqdm(range(len(df))):
        await tasks[i]
    return all_distances


def compute_achilles(
    df: pd.DataFrame,
    categorical_cols: list,
    continuous_cols: list,
    meta_data: list,
    n_to_save: int,
):
    return asyncio.run(
        compute_achilles_parallel(
            df, categorical_cols, continuous_cols, meta_data, n_to_save
        )
    )


def compute_achilles_seq(
    df: pd.DataFrame,
    categorical_cols: list,
    continuous_cols: list,
    meta_data: list,
    n_to_save: int,
):
    ohe, ohe_column_names = fit_ohe(df, categorical_cols, meta_data)
    df_ohe = apply_ohe(
        df.copy(), ohe, categorical_cols, ohe_column_names, continuous_cols
    )

    all_columns = list(df_ohe.columns)
    ohe_cat_indices = [all_columns.index(col) for col in ohe_column_names]
    cont_indices = [all_columns.index(col) for col in continuous_cols]

    n_cat_cols = len(categorical_cols)
    n_cont_cols = len(continuous_cols)

    all_distances = dict()
    n_to_save = n_to_save

    for record_id in tqdm(df.index):
        record = df_ohe.loc[record_id].to_numpy()
        values = df_ohe.values

        # calculate distance for categorical features
        cat_dist = (
            1
            - cosine_similarity(
                record[ohe_cat_indices].reshape(1, -1),
                values[:, ohe_cat_indices],
            ).flatten()
        )
        cat_dist = [
            n_cat_cols / (n_cat_cols + n_cont_cols) * k for k in cat_dist
        ]

        # if there are only categorical columns, we can return this
        if n_cont_cols == 0:
            return cat_dist

        # calculate distance for continuous features
        cont_dist = (
            1
            - cosine_similarity(
                record[cont_indices].reshape(1, -1), values[:, cont_indices]
            ).flatten()
        )
        cont_dist = [
            n_cont_cols / (n_cat_cols + n_cont_cols) * k for k in cont_dist
        ]

        # finally, return the weighted average, weighted by number of respective cols
        mean_dist = np.mean(
            np.sort(
                [cat_dist[i] + cont_dist[i] for i in range(len(cont_dist))]
            )[:n_to_save]
        )
        all_distances[record_id] = mean_dist
    return all_distances
