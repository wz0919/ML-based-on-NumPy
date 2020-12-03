Both elbow method and kmeans method are based on the approaches of finding new centroid in kmeans++

elbow method result
---
Run elbow method for 4 times, result:

![Aaron Swartz](https://raw.githubusercontent.com/wz0919/ML-based-on-NumPy/main/K-Means/data/elbows.png)

Quite stable having elbow at k = 8

kmeans result
---
Run kmeans for 9 times, result:

![Aaron Swartz](https://raw.githubusercontent.com/wz0919/ML-based-on-NumPy/main/K-Means/data/clusters.png)

We can see kmeans++ are fairly likely to have better clusters

In this example even in all the 9 times, the algorithm converged to the global optimal point
