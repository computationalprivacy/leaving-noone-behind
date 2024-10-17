import numpy as np
import pandas as pd


class ArbitraryMarginal(object):
    def __init__(self, bins, cdf, categorical, mini=None, values=None):
        self.bins = bins
        self.cdf = cdf
        self.categorical = categorical
        if categorical:
            #assert len(self.cdf) == 2
            self.n_pos = len(self.cdf)
            self.min = mini
            self.values = values
        else:
            assert len(self.cdf) >= 3 and self.cdf[0] == 0 and self.cdf[-1] == 1


    def uniform(self, a, b):
        assert a < b
        u = np.random.random()
        return a + (b-a) * u
    

    def sample(self, nbr_samples):
        samples = []
        for _ in range(nbr_samples):
            random = np.random.random()
            samples.append(self.get_inversion(random))
        return samples

   
    def binary_lookup(self, u):
        if np.abs(u - self.cdf[-1]) < 1e-6:
            return len(self.cdf)-1, self.bins[-1]
        begin, end = 0, len(self.cdf)-1
        #print('Begin', begin, 'End', end)
        while begin < end:
            middle = (begin + end) // 2
            #print('Middle', middle)
            if self.cdf[middle] <= u < self.cdf[middle + 1]:
                #print('I am here 1')
                return middle, np.random.uniform(self.bins[middle], self.bins[middle+1])
            elif begin < end - 1:
                if self.cdf[middle] <= u:
                    #print('I am here 2')
                    begin = middle
                else:
                    end = middle
            #print('Middle', middle, 'Begin', begin, 'End', end)
        return end, self.bins[end]
    

    def get_inversion(self, u):
        assert 0 <= u <= 1
        if self.categorical:
            for i in range(self.n_pos):
                if u < self.cdf[i] :
                    return self.values[i]
            return self.values[len(self.values)-1]
            # if u < self.cdf[0]:
            #     return 0
            # else:
            #     return 1
        else:
            return self.binary_lookup(u)[1]
            #for i in range(len(self.cdf)-1):
            #    if self.cdf[i] <= u < self.cdf[i+1]:
            #         return self.uniform(self.bins[i], self.bins[i+1])
            #return self.bins[-1]

    #fonction needed to ensure that copulas are working with our new class
    def percent_point(self, cdf):
        output = []
        for u in cdf:
            output.append(self.get_inversion(u))
        return output


def compute_histogram(column, K, categorical=False):
    # The intervals need to be [-inf, min(column)); 
    # [min(column), max(column))/K, [max_column, inf) to match the empirical
    # distribution.
    assert K > 0, f'ERROR: Invalid value for {K}.'
    if isinstance(column, pd.core.series.Series):
        column = column.to_numpy()
    elif isinstance(column, list):
        column = np.array(column)
    else:
        assert isinstance(column, np.ndarray), 'ERROR: Invalid column format.'
    if categorical:
        m, M = np.min(column), np.max(column)
        # Check that y is indeed binary.
        #c0 = np.sum(column==0)
        #c1 = np.sum(column==1)
        #assert c0 + c1 == len(column), \
        #        'ERROR: y is not a binary variable.'
        c0 = np.sum(column==m)
        pi = c0/len(column)
        probs=[pi]
        for i in range(m+1,int(M)+1):
            ci = np.sum(column==i)
            if ci != 0 :
                pi += ci/len(column)
                probs.append(pi)
        return None, probs
    else:
        m, M = np.min(column), np.max(column)
        #print(m,M)
        assert m < M, f'ERROR: This column has only one possible value.'
        bins = np.linspace(m, M, K+1)
        #print(bins)
        hist, _  = np.histogram(column, bins=bins)
        pdf = hist / np.sum(hist)
        #print(bins, pdf)
        #assert np.abs(np.sum(pdf)-1) < 1e-6, pdf
        cdf = [0] + list(np.cumsum(pdf))
        cdf[-1] = 1
        return bins, cdf


# class ArbitraryMarginalOld(object): #=myunivariate
#     def __init__(self, random_intervals):
#         #random_intervals : dic{ (1,p_1) : X_1,..., (n,p_n) : X_n } p_i of being in interval X_i
#         #Note pour le moment : (-3,3)/K + (-inf,-3) = X_1 ,(3,+inf) = X_n
#         self.random_intervals = random_intervals
#         #map [0,1] -> \mathbb{R}
#         #p_i : probability of being in the X_i interval. 
#         #bini = X_i = [begin_i, end_i]
#         mapping = {}
#         begin = 0 
#         for (i,pi) in random_intervals :
#             end = begin + pi 
#             mapping[(begin,end)] = random_intervals[(i,pi)]
#             begin = end
#         #we need to cover every possibility. Due to the inaccuracy on the sum of two float (to be equal to 1), then we take a decent approximation of 1. 
#         assert end >= 0.999, f'Error : your sum of p_i is not equal to 1'
#         self.mapping = mapping
           

#     def get_interval(self,u):#pb
#         for (begin,end) in self.mapping :
#             if u > end : 
#                 continue
#             else :
#                 index = list(self.mapping.keys()).index((begin,end))
#                 break
#         return index
    

#     def get_inversion(self,index):
#         #on tire dans l'intervalle rendu par F-1 (self.mapping)
#         (begin,end) = self.mapping[list(self.mapping.keys())[index]] #F-1[p_1+...+p_index,p_1+...+p_index+p_index+1]
#         if np.inf == end : #pos inf -> poisson
#             sample = np.random.poisson(1) + (3) #m=1
#         elif - np.inf == begin : #neg inf -> poisson
#             sample = - np.random.poisson(1) - (3) #m=1
#         else :# no inf -> uniforme
#             random_sample = np.random.random() #drawn from a continuous distribution over (0,1)
#             sample = (end-begin)*random_sample + begin
#         return sample
    
#     #fonction needed to ensure that copulas are working with our new class
#     def percent_point(self,cdf):
#         output = []
#         for u in cdf :
#             output.append(self.get_inversion(self.get_interval(u)))
#         return output
