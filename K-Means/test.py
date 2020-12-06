import numpy as np
from matplotlib import pyplot as plt
from kmeans import *

X = np.load("data/data.npy")

m = 16
# plotting the loss curve with number of centroids.
for i in range(4):
    losses = elbow_method(m, X)
    plt.subplot(2, 2, i+1)
    plt.plot(losses)
plt.tight_layout()
plt.savefig('data/elbows.png')
plt.show()

m = 8
methods = ['random','k-means++']

for method in methods:
    model = KMeans(n_clusters = m, init = method, n_init = 10)
    
    colours = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'lime', 'wheat', 'fuchsia', 'pink']
    for j in range(9):
        model.fit(X)
        labels = model.predict(X)
        for k in range(m):
            plt.subplot(3, 3, j+1)
            cluster = X[labels == k]
            plt.scatter(cluster[:,2], 
                        cluster[:,3], 
                        c=colours[k])
    
    plt.tight_layout()
    plt.savefig('data/result_with_'+method+'.png')
    plt.show()