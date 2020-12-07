import numpy as np
import matplotlib.pyplot as plt
from GMM import *

def test():
    X = np.load("data.npy")[:,:3]
    K = 4
    model = GaussianMixture(K)
    model.fit(X)
    labels = model.predict(X)
    
    colours = ['r', 'g', 'b', 'y']
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    for k in range(K):
        cluster = X[labels == k]
        ax.scatter(cluster[:,0], cluster[:,1], cluster[:, 2], c=colours[k])
    plt.savefig('result.png')

if __name__ == "__main__":
	test()