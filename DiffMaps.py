'''
A simple implementation of Diffusion Maps in Python.

v0.0
*** NEW AND UNTESTED ***


Project repo: https://github.com/GeometryOfData/DiffusionMaps

@author: Roy Lederman
http://roy.lederman.name
'''

import numpy as np
import scipy.spatial.distance as scdst



#
#
#
def affinityFromDist(d,numneigh=10,myeps=None,islocal=0,isepsviasquare=0):
    # Computes an affinity matrix from distances.
    #
    # input: 
    #  * d is distance (not squared).
    #  * myeps is the scaling factor in the exponential. 
    #    if myeps=None (default), then numneigh, islocal, isepsviasquare are used, otherwise they are ignored.
    #  * numneigh : number of neighbors to use in computing epsilon (if myeps isn't set)
    #  * islocal  : Should I use a local epsilon (see below)
    #  * Should epsilon be computed by averaging the squares of the distances to the nearest neighbors.
    #
    # output:
    #  * A: affinity matrix 
    #  * dMean 
    #  * dMeanVec
    #
    # Global version (myeps is set, or islocal=0)
    #     A_{ij} = exp( - d_{ij}^2 / eps^2 )    
    #
    # Local version (myeps is not set, AND islocal=1)
    #     A_{ij} = exp( - d_{ij}^2 / (eps(i) eps(j) ) )    
    #
    
    #
    # check input
    #
    n=d.shape[0]
    assert(d.ndim==2)
    assert(d.shape[1]==n)
    assert(numneigh>0)
    assert(numneigh<n-1)
    
    #
    # Compute epsilon
    #
    if (myeps!=None):
        islocal=0
        dMeanVec = np.zeros(n,1) 
        dMean = myeps
    else:
        if (isepsviasquare==0):
            dSort = np.sort( d, 1)[:,1:numneigh+1]
            dMeanVec = np.mean( dSort,1 )
            dMean    = np.mean( dSort )
        elif (isepsviasquare==1):
            dSort = np.sort( d**2, 1)[:,1:numneigh+1]
            dMeanVec = np.sqrt(np.mean( dSort,1 ))
            dMean    = np.sqrt(np.mean( dSort ))
        else:
            raise Exception('invalid isepsviasquare')
        
            
    if (dMean<=0):
        raise Exception('eps = 0','May have overlapping points')
      
    #
    # Compute affinity
    #
    d2 = d**2
    if (islocal==0):
        d2 = d2/(dMean**2)
    elif (islocal==1):
        if (np.min(dMeanVec)<=0):
            raise Exception('eps vec contains 0','May have overlapping points')
        for j1 in range(n):
            d2[:,j1]=d2[:,j1]/dMeanVec[j1]
            d2[j1,:]=d2[j1,:]/dMeanVec[j1]
    else:
        raise Exception('invalid islocal')
    
    #
    # A
    #
    A = np.exp(-d2)
    
    return A, dMean, dMeanVec




#
#
#
def diffusionKerFromAffinity( A ):
    #   DiffusionKerFromAffinity generate a diffusion kernel K2 for diffusion
    #maps, from a given symmetric affinity matrix A. 
    #
    #For more details about the algorithm, see the algorithm for 
    #approximating the  Laplace-Beltrami operator in
    #Lafon, S. , ``Diffusion Maps and Geometric Harmonics.''
    #
    #       s0(j)   = \sum_k ( A_{kj} )
    #       K1_{ij} = A_{ij} / ( s0(j) s0(i) )
    #       s1(j)   = \sum_k ( K1_{kj} )
    #       K2_{ij} = K1_{ij} / ( sqrt{s1(j)}  \sqrt{1(i)} )
    #
    #  
    #   Input:
    #    * A  : A (n x n) affinity Matrix. 
    #
    #   Output:
    #    * K2 : The (n x n) diffusion kernel for diffusion maps.
    #    * K1, s1, s0 are also given as part of the output, but they can be
    #        ignored.
    #

    #
    # prechecks
    #
    n=A.shape[0]
    assert(A.ndim==2)
    assert(A.shape[1]==n)
    
    #
    # compute the kernel 
    #
    s0 = np.sum(A,1)
    #print(s0.shape)
    #print(s0)
    #print(s0[0])
    k1 = np.copy(A)
    #print(k1.shape)
    for j1 in range(n):
        k1[:,j1]=k1[:,j1]/s0[j1]
        k1[j1,:]=k1[j1,:]/s0[j1]
        
    s1 = np.sum(k1,1)
    k2 = np.copy(k1)
    for j1 in range(n):
        k2[:,j1]=k2[:,j1]/np.sqrt(s1[j1])
        k2[j1,:]=k2[j1,:]/np.sqrt(s1[j1])
        
    return k2,k1



#
#
#
def diffusionMapsFromKer( K , t ):
    #   DiffusionMapsFromKer computes diffusion maps from a diffusion kernel. 
    #
    #   Gives the embedding and diffusion distances for a given time t,
    #   but also the eigenfunctions that can be used to construct the embedding
    #   and distance for any other diffusion time. 
    #
    #   For more details about the algorithm, see the algorithm for 
    #   approximating the  Laplace-Beltrami operator in
    #      Lafon, S. , ``Diffusion Maps and Geometric Harmonics.''
    #
    # 
    #   Input:
    #    * K  : An (n x n) diffusion kernel ( typically constructed 
    #           by DiffusionKerFromAffinity ). 
    #           Assumed to be symmetric, positive semi-definite.
    #    * t  : The diffusion time
    #
    #   Output:
    #    * MapEmbd  :  The diffusion maps embedding for time t.
    #                  The i-th row MapEmbd(i,:) is the vector corresponding to
    #                  the i-th point,
    #                  The j-th column is the j-th coordinate of the
    #                  embedding. 
    #                  Note, that the first column is a ``trivial''
    #                  coordinate, and should be all ones. This coordinate can
    #                  be ignored. 
    #    * UWeighted : The weighted eigenfunctions (in the columns), i.e. 
    #                  the approximate eigenfunctions of the 
    #                  Laplace-Beltrami operator.
    #                  Note that the first column is the trivial eigenfunctions.
    #    * svals     : the singular values (eigenvalues) of the diffusion
    #                   kernel K.
    #    * U,V       : The right and left singular vectors (eigenvectors) of K,
    #                  such that 
    #
    #   The vectors are sorted in the columns of UWeighted, U,V in the order
    #   of descending singular values (eigenvalues).
    #
    #   To compute the diffusion distance at a different time t', 
    #   compute the embedding at a different time: 
    #      M = UWeighted * diag( svals.^t' )
    #   and then, the diffusion distance at time t', between point i and j is
    #   Euclidean distance between the corresponding rows of M:
    #      d_{t'}(i,j) = || M(i,:) - M(j,:) ||_2
    #
    
    n=K.shape[0]
    assert(K.ndim==2)
    assert(K.shape[1]==n)

    
    #
    #   Eigenvalue decomposition of the kernel (assumed to be symmetric, positive semidefinite)
    #
    U,svals,V = np.linalg.svd(K);
    V=V.T
    
    #
    #   Compute the approximate eigenfunctions of the Laplace-Beltrami operator.
    #
    UWeighted = np.copy(U);
    for j1 in range(n):
        UWeighted[j1,:] = UWeighted[j1,:]/U[j1,0]
    
    #
    #   Compute the diffusion maps embedding at time t.
    #
    MapEmbd = np.copy(UWeighted)
    for j1 in range(n):
        MapEmbd[:,j1] = MapEmbd[:,j1] * svals[j1]**t 

        
    return MapEmbd,UWeighted,svals, U,V


#
#
#
def vecToDiffMap( xx , t, numneigh=10,myeps=None,islocal=0,isepsviasquare=0):

    myd = scdst.squareform(scdst.pdist(xx))    
    A, dMean, dMeanVec = affinityFromDist(myd,numneigh=numneigh,islocal=0,isepsviasquare=1)
    ker2, ker1 = diffusionKerFromAffinity( A )
    MapEmbd,UWeighted,svals, U,V = diffusionMapsFromKer( ker2 , t )

    return MapEmbd, UWeighted, myd, svals, ker2, A



########################################################

#
# Test functions, compute a diffusion map of a fake dataset. 
#    
    

if __name__ == "__main__":

    
    def test_data(n,dim):
    #
    # Fake dataset
    #
        x=np.zeros([n,dim])
        #tmp = np.arange(n) / (n)    
        tmp = np.random.rand(n)    
        x[:,0]= np.sin(tmp*2*np.pi)
        x[:,1]= np.cos(tmp*2*np.pi)
        v = np.zeros([n,2])
        v[:,0] = tmp
        return x , v


    def test_data_2D(n,dim):
    #
    # Another fake dataset
    #
        x=np.random.rand(n,dim)
        x[:,2:] = 0
        v = np.zeros([n,2])
        v[:,0] = x[:,0]
        return x , v
    
    

    import matplotlib.pyplot as plt

    # Dataset: x is data, v is oracle values
    #xx,vv= test_data(500,3)
    xx,vv= test_data_2D(500,3)
    #
    myd = scdst.squareform(scdst.pdist(xx))
    
    # diffusion parameters
    numneigh = 10
    t=1
    
    # plot data
    plt.scatter(xx[:,0], xx[:,1], c=vv[:,0])
    plt.show()
    
    # Get Diffusion Map
    MapEmbd, UWeighted, myd, svals, ker2, A = vecToDiffMap( xx , t, numneigh=numneigh,myeps=None,islocal=0,isepsviasquare=0)
    plt.scatter(MapEmbd[:,1], MapEmbd[:,2], c=vv[:,0])
    plt.show()
    
    