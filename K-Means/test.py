import numpy as np
import matplotlib.pyplot as plt
from kmeans import *

'''K-means with kmeans++'''
X = np.load("data/data.npy") # 1000 4-dimension points, expected 8 clusters

m = 16
# plotting the loss curve with number of centroids.
for i in range(4):
    losses = elbow_method(m, X)
    plt.subplot(2, 2, i+1)
    plt.plot(losses)
plt.tight_layout()
plt.savefig('data/elbows.png')

m = 8 #number of centroid#
i = 100

def allocator(X, L, c):
    cluster = []
    for i in range(L.shape[0]):
        if np.array_equal(L[i, :], c):
            cluster.append(X[i, :])
    return np.asarray(cluster)

colours = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'lime', 'wheat', 'fuchsia', 'pink']
for j in range(9):
    C_final, L_final = kmeans(X, m, i)
    for k in range(m):
        plt.subplot(3, 3, j+1)
        cluster = allocator(X, L_final, C_final[k, :])
        plt.scatter(cluster[:,2], 
                    cluster[:,3], 
                    c=colours[k])

plt.tight_layout()
plt.savefig('data/clusters.png')