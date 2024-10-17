### add feature extractors
import itertools
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

#from src.optimized_qbs import qbs


def fit_ohe(df: pd.DataFrame, categorical_cols: list, metadata: dict) -> tuple:

    # first extract all categories from the metadata
    meta_data_columns = [col['name'] for col in metadata]
    categories = []
    for col in categorical_cols:
        categories.append(metadata[meta_data_columns.index(col)]['representation'])

    ohe = OneHotEncoder(categories=categories)
    ohe.fit(df[categorical_cols])

    ohe_column_names = []
    all_categories = ohe.categories_
    for i, categories in enumerate(all_categories):
        for category in categories:
            ohe_column_names.append(categorical_cols[i] + '_' + str(category))

    return ohe, ohe_column_names

def apply_ohe( df: pd.DataFrame, ohe: OneHotEncoder, categorical_cols: list,
               ohe_column_names: list, continous_cols: list) -> pd.DataFrame:
    ohe_values = ohe.transform(df[categorical_cols]).toarray()
    ohe_df = pd.DataFrame(data = ohe_values, columns = ohe_column_names, index = df.index)
    results_df = df[continous_cols].merge(ohe_df, left_index=True, right_index=True)

    return results_df

def extract_naive_features(synthetic_df: pd.DataFrame, categorical_cols: list,
               ohe_column_names: list, continuous_cols: list, target_record=pd.DataFrame) -> tuple:
    '''Compute the Naive method as described in Groundhod (Usenix 2022)'''

    ## (1) For each continuous col, extract the mean, median and variance
    # get mean, median and var for each col
    means = [np.mean(synthetic_df[col]) for col in continuous_cols]
    medians = [np.median(synthetic_df[col]) for col in continuous_cols]
    varians = [np.var(synthetic_df[col]) for col in continuous_cols]
    features = means + medians + varians
    # get col names
    col_names = ['mean_' + col for col in continuous_cols]
    col_names += ['median_' + col for col in continuous_cols]
    col_names += ['var_' + col for col in continuous_cols]

    ## (2) For each categorical col, extract the the number of distinct categories plus the most and least frequent category
    for cat_col in categorical_cols:
        all_ohe_cols = [i for i in ohe_column_names if i.split('_')[0] == cat_col]
        all_summed = synthetic_df[all_ohe_cols].sum()
        distinct = sum(all_summed > 0)
        most_freq = int(all_summed.index[np.argmax(all_summed.values)].split('_')[1])
        least_freq = int(all_summed.index[np.argmin(all_summed.values)].split('_')[1])
        features += [distinct, most_freq, least_freq]
        col_names += [f'{cat_col}_distinct', f'{cat_col}_most_freq', f'{cat_col}_least_freq']

    return features, col_names

def extract_correlation_features(synthetic_df: pd.DataFrame, categorical_cols: list,
               ohe_column_names: list, continuous_cols: list, target_record=pd.DataFrame) -> tuple:
    corr_matrix = synthetic_df.corr()
    # replace nan values with 0
    corr_matrix = corr_matrix.fillna(0.0)
    # Remove redundant entries from the symmetrical matrix.
    above_diagonal = np.triu_indices(corr_matrix.shape[0], 1)
    features = list(corr_matrix.values[above_diagonal])

    # get col names
    col_names = ['corr_' + str(i) for i in range(len(features))]

    return features, col_names


def get_queries(orders, categorical_indices: list, continous_indices: list,
                num_cols : int, number: int,
                cat_condition_options: tuple = (-1, 1), cont_condition_options: tuple = (3, -3),
                random_state: int = 42) -> list:
    '''
    Condition options:
               0  ->  no condition on this attribute;
		       1  ->  ==
		      -1  ->  !=
		       2  ->  >
		       3  ->  >=
		      -2  ->  <
		      -3  ->  <=
    '''

    all_combinations = []

    for order in orders:
        all_indices = list(itertools.combinations(range(num_cols), order))
        for indices in all_indices:
            indices_combinations = []
            for i, index in enumerate(indices):
                if index in categorical_indices:
                    index_options = cat_condition_options
                else:
                    index_options = cont_condition_options
                if i == 0:
                    for index_option in index_options:
                        base_tup = np.array([0] * num_cols)
                        base_tup[index] = index_option
                        indices_combinations.append(base_tup)
                else:
                    for j, index_option in enumerate(index_options):
                        if j == 0:
                            for base_tup in indices_combinations:
                                base_tup[index] = index_option
                        else:
                            indices_combinations_c = deepcopy(indices_combinations)
                            for base_tup in indices_combinations_c:
                                base_tup[index] = index_option
                            indices_combinations += indices_combinations_c
            for combo in indices_combinations:
                all_combinations.append(tuple(combo))

    if number < len(all_combinations):
        np.random.seed(random_state)
        indices = np.random.choice(
            len(all_combinations), replace=False, size=(number,)
        )
        queries = [all_combinations[idx] for idx in indices]
    else:
        queries = all_combinations

    return queries

def feature_extractor_queries_CQBS(synthetic_df: pd.DataFrame, target_record: pd.DataFrame,
                                   queries: list):
    # set up qbs of synthetic dataframe and define target values
    qbs_data = qbs.SimpleQBS(synthetic_df.itertuples(index=False, name=None))
    target_values = [tuple(target_record.values[0])]

    # get features by batch-quering using the queries and qbs
    features = qbs_data.query(target_values * len(queries), queries)

    # get feature names
    og_data_columns = synthetic_df.columns
    col_names = ['_'.join([f'{cond}_{og_data_columns[i]}' for i, cond in enumerate(conditions) if cond != 0]) for
                 conditions in queries]

    return features, col_names

def feature_extractor_topX_full(synthetic_df: pd.DataFrame, target_record_ohe: pd.DataFrame,
                                top_X: int = 50):
    all_cos_sim = np.array([cosine_similarity(synthetic_df.iloc[i].values.reshape(1, -1),
                                              target_record_ohe.values.reshape(1, -1))[0][0] for i in
                            range(len(synthetic_df))])
    ordered_idx = np.argsort(all_cos_sim)[::-1]

    top_x_data = synthetic_df.iloc[ordered_idx[:top_X]]

    features = list(top_x_data.values.flatten())
    col_names = []

    for i in range(top_X):
        col_names += [k + '_top_X=' + str(i) for k in top_x_data.columns]

    return features, col_names

def feature_extractor_distances(synthetic_df: pd.DataFrame, target_record_ohe: pd.DataFrame):
    all_cos_sim = np.array([cosine_similarity(synthetic_df.iloc[i].values.reshape(1, -1),
                                              target_record_ohe.values.reshape(1, -1))[0][0] for i in
                            range(len(synthetic_df))])
    ordered_vals = np.sort(all_cos_sim)[::-1]

    features = list(ordered_vals)
    col_names = ['distance_X=' + str(k) for k in range(len(features))]

    return features, col_names

def get_feature_extractors(feature_extractor_names: list) -> tuple:
    '''
    given a list of strings specifying the feature extractors to be used,
    create a list of the corresponding functions and parameters
    '''
    feature_extractors, do_ohe = [], []
    for feat in feature_extractor_names:
        if isinstance(feat, str):
            if feat == 'naive':
                feature_extractors.append(extract_naive_features)
                do_ohe.append(True)
            elif feat == 'correlation':
                feature_extractors.append(extract_correlation_features)
                do_ohe.append(True)
            elif feat == 'closest_X_full':
                feature_extractors.append(feature_extractor_topX_full)
                do_ohe.append(True)
            elif feat == 'all_distances':
                feature_extractors.append(feature_extractor_distances)
                do_ohe.append(True)
            else:
                print('Not a valid feature extractor')
        elif isinstance(feat, tuple):
            name, orders, number, conditions = feat
            if name == 'query':
                feature_extractors.append((feature_extractor_queries_CQBS, orders, number, conditions))
                do_ohe.append(False)
            else:
                print('Not a valid feature extractor')
        else:
            print('Not a valid feature extractor')

    return feature_extractors, do_ohe

def apply_feature_extractor(datasets: list, target_record: pd.DataFrame, labels: list, ohe: OneHotEncoder,
                            ohe_columns: list, ohe_column_names: list,
                            continuous_cols: list, feature_extractors: list, do_ohe: list) -> tuple:
    '''
    Given a list of feature extractor functions and synthetic datasets, extract all features and
    create a new dataframe with all features per dataset as individual records
    :param datasets: a list of shadow synthetic datasets
    :param target_record: dataframe of one record with the target record, potentially to be used by feature extractor
    :param labels: a list of labels corresponding to the datasets
    :param ohe: a fitted one hote encoder instance
    :param ohe_columns: the columns on which the ohe should be applied
    :param ohe_column_names: the names of the columns of the ohe result
    :param continuous_cols: the columns that are continuous
    :param feature_extractors: a list of feature extractor functions. All functions have as input
    a dataset and output a list of features and a list of column names. If more than one feature extractor is
    specified all features are extracted and appended
    :param do_ohe: a list of length equal to feature_extractors, this contains booleans
    whether the feature extractor function requires the dataset to be one hot encoded or not
    :return: a dataframe with all features per dataset and the corresponding labels
    '''
    all_features = []

    #for k, dataset in tqdm(enumerate(datasets)):
    for k, dataset in enumerate(datasets):
        if sum(do_ohe) != 0:
            data_ohe = apply_ohe(dataset, ohe, ohe_columns, ohe_column_names, continuous_cols)
            target_ohe = apply_ohe(target_record, ohe, ohe_columns, ohe_column_names, continuous_cols)
        all_feature_one_ds = []
        all_feature_names = []
        for i, feature_extractor in enumerate(feature_extractors):
            if isinstance(feature_extractor, tuple):
                # then we know it's query extracting with additional params
                query_extractor, orders, number, conditions = feature_extractor

                # make sure to compute the queries only once
                if k == 0:
                    all_columns = list(dataset.columns)
                    categorical_indices = [all_columns.index(col) for col in ohe_columns]
                    continous_indices = [all_columns.index(col) for col in continuous_cols]
                    queries = get_queries(orders=orders, categorical_indices=categorical_indices,
                                          continous_indices=continous_indices, num_cols=dataset.shape[1],
                                          number=number, cat_condition_options=conditions['categorical'],
                                          cont_condition_options=conditions['continuous'])
                # for C QBS we need int for categorical
                dataset_int = dataset.copy()
                dataset_int[ohe_columns] = dataset[ohe_columns].astype(int)
                target_record_int = target_record.copy()
                target_record_int[ohe_columns] = target_record_int[ohe_columns].astype(int)
                features, col_names = query_extractor(dataset_int, target_record_int, queries)
            else:
                if do_ohe[i]:
                    features, col_names = feature_extractor(data_ohe, ohe_columns, ohe_column_names,
                                                            continuous_cols, target_ohe)
                else:
                    features, col_names = feature_extractor(dataset, ohe_columns, ohe_column_names,
                                                            continuous_cols, target_record)
            all_feature_one_ds += features
            all_feature_names += col_names
        all_features.append(all_feature_one_ds)

    shadow_train_X = pd.DataFrame(data=np.array(all_features), columns=all_feature_names)

    return shadow_train_X, labels
