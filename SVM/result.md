Result when c = 10:

![Aaron Swartz](https://raw.githubusercontent.com/wz0919/ML-based-on-NumPy/main/SVM/data/result_when_c%3D10.png)

We can see the decision boundray seperates data very well and because c softens the margin, the circled support vectors are raletively farer from the boundary.

—————————

Result when c = 1e7:

![Aaron Swartz](https://raw.githubusercontent.com/wz0919/ML-based-on-NumPy/main/SVM/data/result_when_c%3D10000000.0.png)

When c is very big, this should almost like a hard margin SVM. In our data the positive samples and negative samples are very close so the dicision boundary
should almost exactly seperate the data and the support vectors should be very close to the boundary.

We can see the decision boundray seperates data very well and the circled support vectors are very closed to the boundary.
