import numpy as np
import matplotlib as mpl
import os
from matplotlib import pyplot
import cv2
import pandas as pd
from sklearn.cluster import KMeans 


def findClosestCentroids(X, centroids):
   
    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly.
    idx = np.zeros(X.shape[0], dtype=int)

    # ====================== YOUR CODE HERE ======================
    m=X.shape[0]
    for i in range(m):
        closest = 10000000000  # initial large value
        index = 0
        for j in range(K):
            dist = np.sum(np.square(X[i] - centroids[j]))
            if dist < closest:
                closest = dist
                index = j
        idx[i] = index
    
    # =============================================================
    return idx



def computeCentroids(X, idx, K):
    
    # Useful variables
    m, n = X.shape
    
    centroids = np.zeros((K, n))

    train_egsum = np.zeros((K,n))
    count = np.zeros(K)
    for i in range(m):
        found = 0
        for j in range(K):
            if j == idx[i]:
                found = j
        count[found] += 1
        train_egsum[found] += X[i]
    
    for i in range(K):
        centroids[i] = train_egsum[i]/count[i]

    return centroids


def runkMeans(X, centroids, findClosestCentroids, computeCentroids,
              max_iters=10, plot_progress=False):

    K = centroids.shape[0]
    idx = None
    idx_history = []
    centroid_history = []

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)

        if plot_progress:
            idx_history.append(idx)
            centroid_history.append(centroids)

        centroids = computeCentroids(X, idx, K)

    return centroids, idx


def kMeansInitCentroids(X, K):
    
    m, n = X.shape

    centroid = np.zeros((K, n))

    for i in range(K):
        centroid[i] = X[i] # simply initializing first k training examples as centroids

    return centroid






