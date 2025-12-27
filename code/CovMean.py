import numpy as np
import scipy.linalg as alg
def CovMean(cov_list):
    '''
    compute the mean of covariance features
    '''
    #rows, cols = cov_list.shape
    num = len(cov_list)
    if num>0:
        sum_m = alg.logm(cov_list[0])
    else:
        return np.array([])

    for m in cov_list[1:]:
        lm = alg.logm(m)
        sum_m = sum_m + lm
    sum_m /= num
    return alg.expm(sum_m)
