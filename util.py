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

def img_color_pallete(original_image, k=8):
    img=resizing(original_image)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    w, h, d = original_shape = tuple(img.shape)
    image_array = np.reshape(img, (w * h, d))
    kmeans = KMeans(n_clusters=k).fit(image_array)
    labels = kmeans.predict(image_array)
    df=pd.DataFrame({"labels":labels})
    label_orders=df['labels'].value_counts().to_dict()
    center_colors = list(kmeans.cluster_centers_)
    colors_ordering=[center_colors[i]/255 for i in label_orders.keys()]
    color_labels = [rgbtohex(colors_ordering[i]*255) for i in label_orders.keys()]
    return img,label_orders,colors_ordering,color_labels

def rgbtohex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def resizing(img):
    src = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    width=int(src.shape[1]) 
    height = int(src.shape[0])
    if width>500 or height>500:
        return cv2.resize(src,(500,500))
    return src

