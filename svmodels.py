import numpy as np
from math import *

def gaussian( h, a, C0, Cn=0 ):
    '''
    Gaussian model of the semivariogram
    h = euclidean distance between a par of points
    a = range
    C0 = sill
    Cn = nugget / measurement error
    '''
    # if h is a single digit
    if type(h) == np.float64:
        # calculate the gaussian function
        return Cn+(C0-Cn) * (1 - exp(-3*h**2/a**2))
        
    # if h is an iterable
    else:
        # calcualte the gaussian function for all elements
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        Cn = np.ones(h.size) * Cn
        return map( gaussian, h, a, C0, Cn )
    
def spherical( h, a, C0, Cn=0 ):
    '''
    Spherical model of the semivariogram
    '''
    # if h is a single digit
    if type(h) == np.float64:
        # calculate the spherical function
        if h <= a:
            return Cn + (C0-Cn)*( 1.5*h/a - 0.5*(h/a)**3.0 )
        else:
            return C0
    # if h is an iterable
    else:
        # calcualte the spherical function for all elements
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        Cn = np.ones(h.size) * Cn
        return map( spherical, h, a, C0, Cn )
    
def exponential( h, a, C0, Cn=0 ):
    '''
    Exponential model of the semivariogram
    '''
    # if h is a single digit
    if type(h) == np.float64:
        
        # calculate the exponential function
        return Cn+(C0-Cn) * (1 - exp(-3*h/a))
        
    # if h is an iterable
    else:
        # calcualte the exponential function for all elements
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        Cn = np.ones(h.size) * Cn
        return map( exponential, h, a, C0, Cn )
    
def hole (h, a, C0, Cn=0):
    '''
    hole effect model of the semivariogram
    only valid for 1D kriging!!
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

def hole_N (h, a, C0, Cn=0):
    '''
    Hole effect model of the semivariogram
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