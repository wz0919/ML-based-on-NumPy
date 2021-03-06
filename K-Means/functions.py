import numpy as np

def initialise_parameters(X, m, method = 'k-means++'):
    '''
    k-means++ initialization, m centroids
    C: centroids matrix
    '''
    if method == 'k-means++': 
        C = []
        ind = np.random.choice(X.shape[0])
        C.append(X[ind,:])
        for i in range(m-1):
          # number of local trials
          n_local_trials = 2 + int(np.log(m))
    
          # Choose center candidates by sampling with probability proportional
          # to the squared distance to the closest existing center
          d = (((X[None,:] - np.array(C)[:,None,:])**2).sum(2)).min(0)
          ind = np.random.choice(X.shape[0], n_local_trials, p = d/d.sum())
    
          # choosing best new center among center candidates, i.e. minimize variances of clusters
          new_d = ((X[None,:] - X[ind,None,:])**2).sum(2)
          np.minimum(d, new_d, out = new_d)
          ind = ind[new_d.sum(1).argmin()]
          C.append(X[ind,:])
    
        C = np.array(C)
    elif method == 'random':
        ind = np.random.randint(0, X.shape[0], m)
        C = X[ind,:]
    else:
        print('init method wrong!')

    return C

def E_step(C, X):
    '''
    update assignments
    labels: assigning result
    '''
    L = np.zeros(X.shape)
    labels = ((X[None,:] - C[:,None,:])**2).sum(2).argmin(0)
   
    return labels

def M_step(C, X, labels):
    '''update centroids'''
    L = C[labels,:]
    for i in range(C.shape[0]):
      ind = np.mean(L == C[i,:], axis = 1).astype(bool)
      C[i,:] = np.mean(X[ind,:], axis = 0)
      
    return C

def elbow_method(m, data):
  '''
  elbow method for finding the best K in K-Means
  
  Potential numbers of centroids are 1,2,...,m.
  Each time we add a centriod by k-means++, run k-means 
  until converge and record the final loss.

  P.S. Seems start with 0, i.e., plt.plot(range(m),losses)
  will give a better estimated elbow when plotting the losses
  Args:
    m: potential maximal numbers of centroids

  Returns:
    losses: list of recorded final loss for each number.
  '''

  # initalization
  X = data
  m_times = m
  i = 100
  ind = np.random.choice(X.shape[0])
  C= X[ind:(ind+1),:]
  losses = [np.linalg.norm(X - C)]

  for j in range(m_times-1):

    # k-means++ initialization
    n_local_trials = 2 + int(np.log(m_times))
    d = (((X[None,:] - np.array(C)[:,None,:])**2).sum(2)).min(0)
    ind = np.random.choice(X.shape[0], n_local_trials, p = d/d.sum())
    new_d = ((X[None,:] - X[ind,None,:])**2).sum(2)
    np.minimum(d, new_d, out = new_d)
    ind = ind[new_d.sum(1).argmin()]

    # k-means after adding new center
    C = np.vstack((C,X[ind:(ind+1),:]))
    m,n = C.shape
    iter = 0
    pre_C = np.zeros((m,n))
    while not (pre_C == C).all() and iter <= i:
      pre_C = C
      labels = E_step(C, X)
      C = M_step(C, X, labels)
      iter += 1

    # record loss
    losses.append(np.linalg.norm(X - C[labels,:]))
      
  return losses

def kmeans(X, m, method = 'k-means++', max_iter = 300):
    '''clustering using kmeans'''
    C = np.zeros((m, X.shape[1]))
    m,_ = C.shape
    i = 0
    pre_C = C
    C = initialise_parameters(X, m, method)
    while not (pre_C == C).all() and i <= max_iter:
      pre_C = C.copy()
      labels = E_step(C, X)
      C = M_step(C, X, labels)
      i += 1

    return C, labels