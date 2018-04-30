# See the companion Jupyter Notebook "Semivariogram Models" for detailed information

import numpy as np
from scipy.spatial.distance import squareform, pdist
import collections
from math import *

# Standard Semivariogram Models

def gaussian( h, a, C0, Cn=0., **kwargs ):
    '''
    Gaussian model of the semivariogram
       sv(h) = Cn+(C0-Cn) * (1 - exp(-3*h**2/a**2))
    with:
       h = euclidean distance between a pair of points
       a = range
       C0 = sill
       Cn = nugget
    '''
    if isinstance(h, collections.Iterable):
        # calculate the gaussian function for all elements
        h = np.array(h)
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        Cn = np.ones(h.size) * Cn
        return map( gaussian, h, a, C0, Cn )
    else:
        # calculate the gaussian function
        return Cn+(C0-Cn) * (1 - exp(-3*h**2/a**2))

    
def spherical( h, a, C0, Cn=0., **kwargs):
    '''
    Spherical model of the semivariogram
       sv(h) = Cn + (C0-Cn)*( 1.5*h/a - 0.5*(h/a)**3.0 ) if h <= a, else C0
    with:
       h = euclidean distance between a pair of points
       a = range
       C0 = sill
       Cn = nugget
    '''
    if isinstance(h, collections.Iterable):
        # calculate the gaussian function for all elements
        h = np.array(h)
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        Cn = np.ones(h.size) * Cn
        return map( spherical, h, a, C0, Cn )
    else:
        # calculate the spherical function
        if h <= a:
            return Cn + (C0-Cn)*( 1.5*h/a - 0.5*(h/a)**3.0 )
        else:
            return C0
    

def exponential( h, a, C0, Cn=0., **kwargs):
    '''
    Exponential model of the semivariogram
       sv(h) = Cn+(C0-Cn) * (1 - exp(-3*h/a))
    with:
       h = euclidean distance between a pair of points
       a = range
       C0 = sill
       Cn = nugget
    '''
    if isinstance(h, collections.Iterable):
        # calculate the gaussian function for all elements
        h = np.array(h)
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        Cn = np.ones(h.size) * Cn
        return map( exponential, h, a, C0, Cn )
    else:
        # calculate the gaussian function
        return Cn+(C0-Cn) * (1 - exp(-3*h/a))

# business functions

def instance_of(model, **kwargs):
    """
    return an instance of the variogram function.
    """
    svmfct = lambda h: model( h, **kwargs)
    return svmfct

def get_empirical_semivariogram(X, Y, Z, max_distance, bandwidth):
    '''
    Experimental variogram for a collection of lags
    '''
    
    # handle datetime objects
    if np.issubdtype(X.dtype, np.datetime64):
        # define a reference date and convert arrays to float-type
        assert len(X.shape) == 1, "dimension for dtype array greater than 1 not supported"
        T0 = X[0]  
        X = np.array(X - T0, dtype=(float)) / (1e9 * 60 * 60 * 24)
        
    P = np.stack( [X, Y, Z] ).T
    pd = squareform( pdist( P[:,:2] ) )
    N = pd.shape[0]
    hs = np.arange(0, max_distance, bandwidth)
    
    sv = list()
    for h in hs:  
        Z = list()
        for i in range(N):
            for j in range(i+1,N):
                if( pd[i,j] >= h-bandwidth )and( pd[i,j] <= h+bandwidth ):
                    Z.append( ( P[i,2] - P[j,2] )**2.0 )
        sumz = np.sum( Z ) / ( 2.0 * len( Z ) )
        sv.append( sumz )
    sv = [ [ hs[i], sv[i] ] for i in range( len( hs ) ) if sv[i] > 0 ]
    return np.array( sv ).T
    

# Non-Standard Semivariogram functions
def hole (h, a, C0, Cn=0., **kwargs ):
    '''
    hole effect model of the semivariogram
    Should be used in 1D kriging only.
    h = euclidean distance between a pair of points
    a = range
    C0 = sill
    Cn = nugget
    '''
    if type(h) == np.float64:
        # calculate the hole function from Kitanidis (book, 1997)
        return Cn+(C0-Cn)*(1-(1-h/a) * exp(-h/a) )

    # if h is an iterable
    else:
        # calcualte the hole function for all elements
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        Cn = np.ones(h.size) * Cn
        return map( hole, h, a, C0, Cn )

def hole_N (h, a, C0, Cn=0., **kwargs ):
    '''
    Hole effect model of the semivariogram 
    (Triki et al. p.1600 / Dowdall et al. 2003)
    h = euclidean distance between a pair of points
    a = range
    C0 = sill
    Cn = nugget
    '''
    # from Triki et al. p.1600 (Dowdall et al. 2003)
    if type(h) == np.float64:
        # calculate the hole function
        if h == 0:
            return Cn
        else: 
            return Cn+(C0-Cn)*( 1-(sin(h/a ))/(h/a) )

    # if h is an iterable
    else:
        # calcualte the hole function for all elements
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        Cn = np.ones(h.size) * Cn
        return map( hole_N, h, a, C0, Cn )
    
