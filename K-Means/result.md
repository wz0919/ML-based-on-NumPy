elbow method result
---
Run elbow method for 4 times, result:

![Aaron Swartz](https://raw.githubusercontent.com/wz0919/ML-based-on-NumPy/main/K-Means/data/elbows.png)

Quite stable having elbow at k = 8

kmeans result
---
Run kmeans for 9 times with max_iter 10, result:

with k-means++

![Aaron Swartz](https://raw.githubusercontent.com/wz0919/ML-based-on-NumPy/main/K-Means/data/result_with_k-means%2B%2B.png)

All results are global optimal.  In fact, with smaller max_iter kmeans++ will aslo has these result. 

with randomly initialization:

![Aaron Swartz](https://raw.githubusercontent.com/wz0919/ML-based-on-NumPy/main/K-Means/data/result_with_random.png)

Even if with max_iter 10, there're still some local optimal points.

