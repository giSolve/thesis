
import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
def embedding_quality(X, Z, classes, knn=10, knn_classes=10, subsetsize=1000):
    # ensure data is in the form of a numpy array for this function 
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    Z = np.array(Z) if not isinstance(Z, np.ndarray) else Z

    # measuring local embedding quality 
    nbrs1 = NearestNeighbors(n_neighbors=knn).fit(X)
    ind1 = nbrs1.kneighbors(return_distance=False)

    nbrs2 = NearestNeighbors(n_neighbors=knn).fit(Z)
    ind2 = nbrs2.kneighbors(return_distance=False)

    intersections = 0.0
    for i in range(X.shape[0]):
        intersections += len(set(ind1[i]) & set(ind2[i]))
    mnn = intersections / X.shape[0] / knn
    
    # measuring class preservation 
    cl, cl_inv = np.unique(classes, return_inverse=True)
    C = cl.size
    # to make sure we can handle less than 10 classes as well 
    effective_knn_classes = min(knn_classes, C - 1) 
    mu1 = np.zeros((C, X.shape[1]))
    mu2 = np.zeros((C, Z.shape[1]))
    for c in range(C):
        mu1[c,:] = np.mean(X[cl_inv==c,:], axis=0)
        mu2[c,:] = np.mean(Z[cl_inv==c,:], axis=0)
        
    nbrs1 = NearestNeighbors(n_neighbors=effective_knn_classes).fit(mu1)
    ind1 = nbrs1.kneighbors(return_distance=False)
    nbrs2 = NearestNeighbors(n_neighbors=effective_knn_classes).fit(mu2)
    ind2 = nbrs2.kneighbors(return_distance=False)
    
    intersections = 0.0
    for i in range(C):
        intersections += len(set(ind1[i]) & set(ind2[i]))
    mnn_global = intersections / C / knn_classes
    
    # measuring global embedding quality 
    size_of_subset = min(X.shape[0], subsetsize)
    subset = np.random.choice(X.shape[0], size=size_of_subset, replace=False)
    d1 = pdist(X[subset,:])
    d2 = pdist(Z[subset,:])
    rho = scipy.stats.spearmanr(d1[:,None],d2[:,None]).correlation
    
    return (mnn, mnn_global, rho)