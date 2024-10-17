from copulas.multivariate.gaussian import GaussianMultivariate
import numpy as np
from src.marginals import ArbitraryMarginal, compute_histogram

def generate_synthetic_dataset_using_copulas(nbr_samples, univariates,
        correlation_matrix, name_columns):
    """
    Method to generate a synthetic dataset using Gaussian copulas.


    Takes as input the number of samples to generate, as well as the 
    marginal distributions and the correlation matrix.
    """
    copula = GaussianMultivariate()
    
    copula.correlation = correlation_matrix

    copula.columns = name_columns
    copula.fitted = True
    copula.univariates = univariates

    #Sample the synthetic dataset.
    synthetic_dataset = copula.sample(nbr_samples)
    return synthetic_dataset

def project(M):
    vparray,P=np.linalg.eig(M)
    for i in range(len(vparray)) :
        if vparray[i] < 0 :
            vparray[i]=0.
    #this will ensure 
    projection = P.dot(np.diag(vparray)).dot(P.T)
    return projection

def is_pos_def(M):
    return np.all(np.linalg.eigvals(M) >= 0)


def return_correlation_after_copula(univariates,correlation_matrix, name_columns):
    synth = generate_synthetic_dataset_using_copulas(1000, univariates, correlation_matrix, name_columns)
    corr = synth.corr().to_numpy() #Corr(Xi,Y)
    return corr

def compute_distances(real_correlation, shifted_correlation, univariates, name_columns):
    correlation_after_copula = return_correlation_after_copula(univariates,shifted_correlation, name_columns)
    diff = []
    diff_dict = {}
    for i in range(len(univariates)):
        for j in range(i+1,len(univariates)):
            distance = real_correlation[i][j] - correlation_after_copula[i][j]
            if shifted_correlation[i][j] != 0.999999 and shifted_correlation[i][j] != -0.999999 :
                diff.append(np.abs(distance))
            diff_dict[(i,j)] = distance
    return diff,diff_dict

def compute_correlation_copula(univariates, real_correlation, epsilon, name_columns, steps=1000):
    shifted_correlation = return_correlation_after_copula(univariates,
            real_correlation, name_columns)
    diff, diff_dict = compute_distances(real_correlation,
            real_correlation,
            univariates, name_columns)
    etape = 0
    while max(diff) >= epsilon and etape <= steps:
        for i in range(len(univariates)):
            for j in range(i+1,len(univariates)):
                if np.abs(diff_dict[(i,j)]) >= epsilon:
                    new_corr = np.maximum(-0.999999, 
                            np.minimum(0.999999, 
                                shifted_correlation[i][j]+diff_dict[(i,j)]/2))
                    shifted_correlation[i][j] = new_corr
                    shifted_correlation[j][i] = new_corr
        if not is_pos_def(shifted_correlation):
            shifted_correlation = project(shifted_correlation)
        diff, diff_dict = compute_distances(real_correlation,
                shifted_correlation,univariates, name_columns)
        etape += 1
        if etape%100 == 0:
            print(etape)
        if diff == []:
            break
    return shifted_correlation

def init_univariates(dataset, K, continuous_columns, categorical_columns):
    """
    Set `y_binary` to True if the last column of the dataset (by default, the
    output variable `y`) is binary.
    """
    univariates = []
    for i, column in enumerate(dataset.columns):
        print(column)
        if column in continuous_columns:
            bins, cdf = compute_histogram(dataset[column], K, False)
            univariates.append(ArbitraryMarginal(bins, cdf, False))
        else :
            bins, cdf = compute_histogram(dataset[column], K, True)
            univariates.append(ArbitraryMarginal(bins, cdf, True, np.min(dataset[column]), list(set(dataset[column]))))
    return univariates