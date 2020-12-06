import numpy as np

def initialise_parameters(X, K):
    '''
    split X into K groups
    calulate the mean and variance of each group
    as the initialized parameter
    
    expect X to be 2d (N, m)
    
    return parameters
    '''

    N, m = X.shape
    sigma = np.zeros((K, m, m))
    mu = np.zeros((K, m))
    pi = np.zeros(K)

    indices = np.arange(N)
    np.random.shuffle(indices)
    # find the K groups
    groups = np.array_split(X[indices,:m], K, axis=0)

    for i in range(K):
      g = groups[i]
      mu[i] = g.mean(axis=0)
      sigma[i] = (g - mu[i]).T@(g - mu[i])/(g.shape[0])
    
    pi = np.ones(K)/K

    return sigma, mu, pi

def E_step(pi, mu, sigma, X):
    '''
    calculate the responsibility matrix, i.e. 
    P(point i comes from j clusters| point i and i's Gaussian)
    
    return responsibility matrix
    '''
    N, m = X.shape
    K = len(pi)
    r = np.zeros((N, K))
    
    from scipy.stats import multivariate_normal
    for i in range(K):
      r[:,i] = multivariate_normal.pdf(X[:,],mu[i],sigma[i])
    r = r * pi[None,:]
    scale = r.sum(1)
    r = r/scale[:,None]
    return r

def M_step(r, X):
    '''
    update the parameters by their MLE
    
    return parameters
    '''
    K = r.shape[1]
    N, m = X.shape
    mu = np.zeros((K, m))
    sigma = np.zeros((K, m, m))
    pi = np.zeros(K)
    
    x = X[:,:m]
    sum_p = r.sum(0)
    pi = sum_p/N
    mu = r.T @ x/sum_p[:,None]
    for i in range(K):
      sigma[i] = (x - mu[i]).T@((x - mu[i])*r[:,i:(i+1)])/sum_p[i]
    return mu, sigma, pi

def classify(pi, mu, sigma, x, calculating_nll = False):
    '''
    classify points by their responsibilities
    
    if calculating_nll == True, also caculate the nll
    
    return assigned cluster's index and nll
    '''
    K = len(pi)
    nll = None
    m = 1 if x.ndim == 1 else x.shape[0]
    prob = np.tile(np.zeros_like(pi), (m,1))
    from scipy.stats import multivariate_normal
    for i in range(K):
      prob[:,i] = multivariate_normal.pdf(x,mu[i],sigma[i])*pi[i]
    ind = prob.argmax(1)
    if calculating_nll == True:
        nll = -np.log(prob.sum(1)).sum()

    return ind, nll

def EM(X, K, iterations):
    '''
    EM method for optimizing GMM
    '''
    N, m = X.shape
    mu = np.zeros((K, m))
    sigma = np.zeros((K, m, m))
    pi = np.zeros(K)

    pre_r = np.zeros((X.shape[0],K))
    sigma, mu, pi = initialise_parameters(X, K)
    for i in range(iterations):
      r = E_step(pi, mu, sigma, X)
      if np.linalg.norm(pre_r - r) < 1e-10:
        break
      mu, sigma, pi = M_step(r, X)
      pre_r = r

    return mu, sigma, pi