import numpy as np
from matplotlib import pyplot as plt
from kernel import *
from svm import *
import time

def add_intercept_fn(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x

def load_csv(csv_path, label_col='y', add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 'l').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    # Load headers
    with open(csv_path, 'r', newline='') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def main(c):
    t1 = time.time()
    x, y = load_csv('data/ds1_b.csv', add_intercept=False)
    
    c = c
    # simple inner product: order = 1, bias = 0
    state = train_svm_smo(x, y, c, polynomial_kernel, 300, 1, 0)
    alpha = state['alpha']
    b = state['b']
    omiga = (y*alpha).dot(x)
    
    # plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == -1, -2], x[y == -1, -1], 'go', linewidth=2)
    
    # Plot decision boundary (found by solving for theta^T x + b = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(b / omiga[1] + omiga[0] / omiga[1] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)
    
    # circle support vectors
    indices = np.where((alpha > 0.001)*(alpha < c-0.001) == 1)
    support_vector = x[indices]
    sv_x1 = support_vector[:,0]
    sv_x2 = support_vector[:,1]
    plt.scatter(sv_x1, sv_x2, s=150, c='none', alpha=0.7,
            linewidth=1.5, edgecolor='#AB3319')
    
    # Add labels
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig('data/result_when_c={}.png'.format(c))
    plt.show()
    
    t2 = time.time()
    print('total time: %1f s'%(t2 - t1))
    
for c in [10,1e7]:
    main(c)
