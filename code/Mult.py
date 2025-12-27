import scipy.linalg as alg
def Mult (a,b):
    '''
    compute the multiplication of two covariance matrix
    '''
    return alg.expm(alg.logm(a)+alg.logm(b))
