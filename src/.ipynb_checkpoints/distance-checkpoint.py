import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

def compute_distances(record: np.array, values: np.array,
                      ohe_cat_indices: list, continous_indices: list,
                      n_cat_cols: int, n_cont_cols: int,
                      method: str = 'cosine', p=None):
    '''
    Compute the generalized distance between the given record and the provided values (collection of other records)
    :param record: Given record, with:
                    - The categorical columns are one-hot-encoded
                    - The continuous columns are normalized (minus min, divided by max - min)
    :param values: A numpy array with all other records, with respect to which the distance will be computed
    :param ohe_cat_indices: A list of indices of all one-hot-encoded values in record and values
    :param continous_indices: A list of indices of all continuous values in record and values
    :param n_cat_cols: Number of categorical attributes
    :param n_cont_cols: Number of continuous attributes
    :param method: The distance method to be used, by default 'cosine'
    :param p: If method is 'minkowski', provide the associated value for p
    :return: a list of distances for the given record to all the given values
    '''
    # first define distance based on categorical
    if method == 'cosine':
        cat_dist = 1 - cosine_similarity(record[ohe_cat_indices].reshape(1, -1),
                                         values[:, ohe_cat_indices]).flatten()
    elif method == 'minkowski':
        assert p is not None
        cat_dist = [distance.minkowski(record[ohe_cat_indices], value[ohe_cat_indices], p=p) for value in values]

    cat_dist = [n_cat_cols / (n_cat_cols + n_cont_cols) * k for k in cat_dist]

    # if there are only categorical columns, we can return this
    if n_cont_cols == 0:
        return cat_dist

    # then define it based on continuous
    if method == 'cosine':
        cont_dist = 1 - cosine_similarity(record[continous_indices].reshape(1, -1),
                                          values[:, continous_indices]).flatten()
    elif method == 'minkowski':
        assert p is not None
        cont_dist = [distance.minkowski(record[continous_indices], value[continous_indices], p=p) for value in values]

    cont_dist = [n_cont_cols / (n_cat_cols + n_cont_cols) * k for k in cont_dist]

    # finally, return the weighted average, weighted by number of respective cols
    return [cat_dist[i] + cont_dist[i] for i in range(len(cont_dist))]

