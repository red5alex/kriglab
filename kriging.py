import numpy as np
from scipy.spatial.distance import pdist, squareform

def kriging_simple( X, F, covfct, u, n, mu=None, V=0., trendf=None ):
    '''
    Input  (X)       Cartesian Coordinates of sample points (array-like, n x dim)
           (F)       Sampled values (array-like, n x 1)
           (covfct)  covariance modelling function (function handler, F(distance) )
           (u)       coordinates of unsampled point (array-like, dim) 
           (N)       number of neighboring points to consider during computation
           (mu)      known mean of values 
                     Default = mean of Z
           (V)       known variance of sample values Z.
                     - if a scalar, V is applied to all sample points. Default = 0.
                     - if array-like, is applied as individual variance
                       for each sample point  (must be of same size as Z)
           (trendf)  trend function handler
    '''
    
    ## Parameter handling ##
    
    # reduce number of neighbors if exceeding sample points
    if n > len(F):
        n = len(F)
    
    # if variance is a scalar, apply equally to all samples
    if np.isscalar(V):
        V = np.ones_like(F) * V
    
    # if mean is not given, use mean of the samples
    if mu is None:
        mu = np.mean( F )
        
    # if no trend function is provided, trend is zero
    if trendf is None:
        trendf = lambda x: mu
    
    # cast array-like input to arrays
    X = np.asarray(X)
    F = np.asarray(F)
    u = np.asarray(u)
    V = np.asarray(V)
    
    # convert any 1D arrays to one-columned 2D matrices
    if len(X.shape) == 1: 
        X = X.reshape(X.shape[0],1)
    if len(F.shape) == 1:
        F = F.reshape(F.shape[0],1)
    if len(V.shape) == 1:
        V = V.reshape(V.shape[0],1)

    # check correct dimensionality of input
    if X.shape[1] != len(u):
        raise ValueError("dimensions of X and u differ ({} != {})".format(X.shape[1], len(u)))
    if X.shape[0] != F.shape[0]:
        raise ValueError("X and F must be of equal Length ({} != {})".format(X.shape[0], X.shape[1]))
    if F.shape != V.shape:
        raise ValueError("F and V must be of equal shape ({} != {})".format(F.shape, V.shape))
    
    # handle datetime objects
    if np.issubdtype(X.dtype, np.datetime64):
        
        # check that unsampled point is datetime as well
        if not np.issubdtype(u.dtype, np.datetime64):
            ValueError("dtype mismatch of X and u")
        
        # define a reference date and convert arrays to float-type
        T0 = X[0]  
        X = np.array(X - T0, dtype=(float)) / (1e9 * 60 * 60 * 24)  
        u = np.array(u - T0, dtype=(float)) / (1e9 * 60 * 60 * 24)
    
    ## Simple Kriging ##
    
    # distance between u and each sample point
    D = np.linalg.norm(X - u, axis=1)
   
    # consolidate all vectors in a single matrix P = [X, Z, d, V] 
    P = np.hstack([X, F, D.reshape(D.shape[0],1), V])
    
    # sort P by distances and take the closest N neighbors 
    P = P[D.argsort()[:n]]
    
    # apply the covariance model to the distances
    xdim = X.shape[1]
    k = covfct( P[:, xdim+1] )
    # cast as a matrix
    k = np.matrix( k ).T
 
    # calculate the trend function at sample points X and unsampled point u
    trendu = trendf(u)
    T = np.apply_along_axis(trendf, 1, P[:,:xdim])
 
    # form a matrix of distances between sample points
    K = squareform( pdist( P[:,:xdim] ) )

    # apply the covariance model to these distances
    K = covfct( K.ravel() )

    # re-cast as a NumPy array -- thanks M.L.
    K = np.array( K )

    # reshape into an array
    K = K.reshape(n,n)

    # cast as a matrix
    K = np.matrix( K )
    
    # add the individual noise variance to the principal diagonal of K 
    K += np.identity(K.shape[0]) * P[:, xdim+2]

    # calculate the kriging weights
    weights = np.linalg.inv( K ) * k
    weights = np.array( weights )
 
    if len(T.shape) == 1:
        T = T.reshape(T.shape[0],1)
 
    # calculate the residuals to trend
    residuals = P[:,xdim] - T[:,0] #- trendu

    # calculate the estimation
    estimate = np.dot( weights.T, residuals ) + trendu

   # calculate the Kriging variance
    assert (K.diagonal().max() - K.diagonal().min()) < (K.diagonal().mean() * 1e-9) , "Diagonal of K is not homogeneous! "+str(K.diagonal())
    C0 = np.diag(K).mean()  # K.diag()  is supposed to contain the sill
    variance = C0 - np.dot(weights.T, k)
    variance = max(0, variance**2) # round-off errors may cause negatives close to zero
    
    return estimate, variance, {"weights" : weights, "K" : K, "k": k, "P" : P, "mu" : mu}