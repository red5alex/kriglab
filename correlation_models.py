import numpy as np

from svmodels import SV
from helpers import opt


def cvmodel(P, model, hs, bw, Cn=None, svrange=None, C0=None):
    '''
    Input:  (P)      ndarray, data
            (model)  modeling function
                      - spherical
                      - exponential
                      - gaussian
            (hs)     distances
            (bw)     bandwidth
    Output: (covfct) function modeling the covariance
    '''
    if Cn is None:
        Cn = N(P, hs, bw)

    if type(C0) == float:
        C0 = np.float64(C0)

    # calculate the semivariogram
    sv = SV(P, hs, bw)
    # calculate the sill
    if C0 is None:
        C0 = C(P, hs[0], bw)
    # calculate the optimal parameters
    if svrange is None:
        svrange = opt(model, sv[0], sv[1], C0)
    # return a covariance function
    covfct = lambda h, a=svrange: C0 - list(model(h, a, C0, Cn=Cn))
    return covfct