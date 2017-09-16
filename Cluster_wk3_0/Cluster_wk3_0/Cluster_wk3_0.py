import graphlab
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy.sparse import csr_matrix


'''Check GraphLab Create version'''
from distutils.version import StrictVersion
assert (StrictVersion(graphlab.version) >= StrictVersion('1.8.5')), 'GraphLab Create must be version 1.8.5 or later.'

from sklearn.preprocessing import normalize

def get_initial_centroids(data, k, seed=None):
    '''Randomly choose k data points as initial centroids'''
    if seed is not None: # useful for obtaining consistent results
        np.random.seed(seed)
    n = data.shape[0] # number of data points
        
    # Pick K indices from range [0, N).
    rand_indices = np.random.randint(0, n, k)
    
    # Keep centroids as dense format, as many entries will be nonzero due to averaging.
    # As long as at least one document in a cluster contains a word,
    # it will carry a nonzero weight in the TF-IDF vector of the centroid.
    centroids = data[rand_indices,:].toarray()
    
    return centroids
from sklearn.metrics import pairwise_distances


def assign_clusters(data, centroids):
    
    # Compute distances between each data point and the set of centroids:
    # Fill in the blank (RHS only)
    distances_from_centroids = pairwise_distances(data,centroids,metric='euclidean')
    
    # Compute cluster assignments for each data point:
    # Fill in the blank (RHS only)
    cluster_assignment = np.argmin(distances_from_centroids,axis=1)
    
    return cluster_assignment


def revise_centroids(data, k, cluster_assignment):
    new_centroids = []
    for i in xrange(k):
        # Select all data points that belong to cluster i. Fill in the blank (RHS only)
        member_data_points = data[cluster_assignment==i]
        # Compute the mean of the data points. Fill in the blank (RHS only)
        centroid = member_data_points.mean(axis=0)
        #print type(centroid)
        print 'centriod = ', centroid
        # Convert numpy.matrix type to numpy.ndarray type
        #centroid = centroid.A1
        new_centroids.append(centroid)
    new_centroids = np.array(new_centroids)
    
    return new_centroids


def compute_heterogeneity(data, k, centroids, cluster_assignment):
    
    heterogeneity = 0.0
    for i in xrange(k):
        
        # Select all data points that belong to cluster i. Fill in the blank (RHS only)
        member_data_points = data[cluster_assignment==i, :]
        
        if member_data_points.shape[0] > 0: # check if i-th cluster is non-empty
            # Compute distances from centroid to data points (RHS only)
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances**2
            heterogeneity += np.sum(squared_distances)
        
    return heterogeneity

compute_heterogeneity(data, 2, centroids, cluster_assignment)

# Fill in the blanks
def kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False):
    '''This function runs k-means on given data and initial set of centroids.
       maxiter: maximum number of iterations to run.
       record_heterogeneity: (optional) a list, to store the history of heterogeneity as function of iterations
                             if None, do not store the history.
       verbose: if True, print how many data points changed their cluster labels in each iteration'''
    centroids = initial_centroids[:]
    prev_cluster_assignment = None
    
    for itr in xrange(maxiter):        
        if verbose:
            print(itr)
        
        # 1. Make cluster assignments using nearest centroids
        # YOUR CODE HERE
        cluster_assignment = assign_clusters(data,centroids)
            
        # 2. Compute a new centroid for each of the k clusters, averaging all data points assigned to that cluster.
        # YOUR CODE HERE
        centroids = revise_centroids(data,k,cluster_assignment)
            
        # Check for convergence: if none of the assignments changed, stop
        if prev_cluster_assignment is not None and (prev_cluster_assignment==cluster_assignment).all():
            break
        
        # Print number of new assignments 
        if prev_cluster_assignment is not None:
            num_changed = np.sum(prev_cluster_assignment!=cluster_assignment)
            if verbose:
                print('    {0:5d} elements changed their cluster assignment.'.format(num_changed))   
        
        # Record heterogeneity convergence metric
        if record_heterogeneity is not None:
            # YOUR CODE HERE
            score = compute_heterogeneity(data,k,centroids,cluster_assignment)
            record_heterogeneity.append(score)
        
        prev_cluster_assignment = cluster_assignment[:]
        
    return centroids, cluster_assignment

def plot_heterogeneity(heterogeneity, k):
    plt.figure(figsize=(7,4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.show()
    plt.close()

data = np.array([[-1.88, 2.05],
                 [-0.71, 0.42],
                 [2.41, -0.67],
                 [1.85,-3.8],
                 [-3.69,-1.33]])
centroids = np.array([[2., 2.],
                      [-2., -2.]])

cluster_assignment = assign_clusters(data, centroids)
print cluster_assignment

k = 2
heterogeneity = []
#initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
#centroids, cluster_assignment = kmeans(data, k, centroids, maxiter=400,
#                                       record_heterogeneity=heterogeneity, verbose=True)
#plot_heterogeneity(heterogeneity, k)
#np.bincount(cluster_assignment)

 
#centroids = initial_centroids[:]
prev_cluster_assignment = None
maxiter = 20
record_heterogeneity=heterogeneity
verbose=True    

for itr in xrange(maxiter):        
    if verbose:
        print(itr)
        
    # 1. Make cluster assignments using nearest centroids
    # YOUR CODE HERE
    cluster_assignment = assign_clusters(data,centroids)
    print 'cluster_assignment =', cluster_assignment
            
    # 2. Compute a new centroid for each of the k clusters, averaging all data points assigned to that cluster.
    # YOUR CODE HERE
    centroids = revise_centroids(data,k,cluster_assignment)
            
    # Check for convergence: if none of the assignments changed, stop
    if prev_cluster_assignment is not None and (prev_cluster_assignment==cluster_assignment).all():
        break
        
    # Print number of new assignments 
    if prev_cluster_assignment is not None:
        num_changed = np.sum(prev_cluster_assignment!=cluster_assignment)
        if verbose:
            print('    {0:5d} elements changed their cluster assignment.'.format(num_changed))   
        
    # Record heterogeneity convergence metric
    if record_heterogeneity is not None:
        # YOUR CODE HERE
        score = compute_heterogeneity(data,k,centroids,cluster_assignment)
        record_heterogeneity.append(score)
        
    prev_cluster_assignment = cluster_assignment[:]

