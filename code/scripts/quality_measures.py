import numpy as np 
import scipy
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist

def embedding_quality(X, Z, classes, knn=10, knn_classes=3, subsetsize=1000):
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
    # print(f"effective number of classes: {effective_knn_classes}")
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
    
    num_iterations = 10  # Number of times to compute Spearman correlation
    size_of_subset = min(X.shape[0], subsetsize)

    rho_values = []
    for _ in range(num_iterations):
        subset = np.random.choice(X.shape[0], size=size_of_subset, replace=False)
        d1 = pdist(X[subset, :])
        d2 = pdist(Z[subset, :])
        rho = scipy.stats.spearmanr(d1, d2).correlation
        rho_values.append(rho)

    rho = np.mean(rho_values) 
    # measuring global embedding quality 
    
    #subset = np.random.choice(X.shape[0], size=size_of_subset, replace=False)
    #d1 = pdist(X[subset,:])
    #d2 = pdist(Z[subset,:])
    #rho = scipy.stats.spearmanr(d1[:,None],d2[:,None]).correlation
    
    return (mnn, mnn_global, rho)

def compute_quality_results(embedding_dict, datasets, knn=10, subsetsize=1000):
    """
    Computes embedding quality measures for each embedding and organizes results by iteration length.
    
    Parameters:
        embedding_dict (dict): Dictionary with keys (T, dataset_index) mapping to 
                               (embedding, labels, kld_values) as returned by run_tsne_with_callbacks_and_timing.
        datasets (list): List of tuples (data, labels) that were used to compute the embeddings.
        knn (int): Number of nearest neighbors for local quality measure.
        subsetsize (int): Number of points to sample for computing the Spearman correlation.
    
    Returns:
        quality_results (dict): Dictionary with iteration length as keys. Each value is a dictionary
                                mapping dataset_index to a tuple (mnn, mnn_global, rho).
                                Structure example:
                                {
                                  250: { 0: (mnn, mnn_global, rho),
                                         1: (mnn, mnn_global, rho),
                                         ... },
                                  500: { ... },
                                  ...
                                }
    """
    quality_results = {}
    
    for (T, dataset_index), (embedding, _, _) in embedding_dict.items():
        # Retrieve the original data and labels for this dataset.
        data, labels = datasets[dataset_index]
        # Ensure data is a numpy array.
        if hasattr(data, "values"):
            X = data.values.astype(float)
        else:
            X = np.array(data)
        
        # The embedding (Z) is already a numpy array, but we ensure it.
        Z = np.array(embedding)
        
        # Determine the number of unique classes.
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)
        # Set knn_classes to one-third of the number of classes, rounded to a whole number, with a minimum of 1.
        knn_classes = max(1, int(round(n_classes / 3)))
        
        # Compute quality measures using your provided embedding_quality function.
        quality = embedding_quality(X, Z, labels, knn=knn, knn_classes=knn_classes, subsetsize=subsetsize)
        # quality is a tuple: (mnn, mnn_global, rho)
        
        # Organize results by iteration length.
        if T not in quality_results:
            quality_results[T] = {}
        quality_results[T][dataset_index] = quality
        
    return quality_results
