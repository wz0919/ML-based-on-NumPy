import numpy as np

'''sequential minimal optimization with heuristics'''

class smo_para_and_func():
    '''
    a class to store the parameters and intermediate calculations
    '''
    def __init__(self, X, Y, c, tol, kernel, **para):
        self.m, _ = X.shape
        self.b = 0
        self.alpha = np.zeros(self.m)
        self.c = c
        self.K = kernel(X, X, **para)
        self.Y = Y
        self.tol = tol
        self.dists = (self.alpha*Y).dot(self.K) + self.b - Y
        
def take_step(i,j, p):
    '''
    optimize the two-variable quadratic problem
    reuturn 1 (means alpha changed) if alpha changed else 0 (means alpha didn't change)
    '''
    eta = p.K[i,i] + p.K[j,j] - 2*p.K[i,j]
    unclipped_alphaj = p.alpha[j] + p.Y[j]*(p.dists[i] - p.dists[j])/eta
    
    if p.Y[i] != p.Y[j]:
        L = np.max([0, p.alpha[j] - p.alpha[i]])
        H = np.min([p.c, p.c + p.alpha[j] - p.alpha[i]])

    else:
        L = np.max([0, p.alpha[j] + p.alpha[i] - p.c])
        H = np.min([p.c, p.alpha[j] + p.alpha[i]])
    if L == H:
        return 0
    
    clipped_alphaj = H if unclipped_alphaj > H else L if unclipped_alphaj < L else unclipped_alphaj
    clipped_alphai = p.alpha[i] + p.Y[i]*p.Y[j]*(p.alpha[j] - clipped_alphaj)
    
    if abs(clipped_alphai - p.alpha[i]) < p.tol:
        return 0
    bj = -p.dists[j] - p.Y[i]*p.K[i,j]*(clipped_alphai - p.alpha[i]) - p.Y[j]*p.K[j,j]*(clipped_alphaj - p.alpha[j]) + p.b
    bi = -p.dists[i] - p.Y[i]*p.K[i,i]*(clipped_alphai - p.alpha[i]) - p.Y[j]*p.K[i,j]*(clipped_alphaj - p.alpha[j]) + p.b
    if 0 < clipped_alphai < p.c:
        p.b = bi
    elif 0 < clipped_alphaj < p.c:
        p.b = bj
    else:
        p.b = (bi + bj)/2 
    p.alpha[i] = clipped_alphai
    p.alpha[j] = clipped_alphaj
    p.dists = (p.alpha*p.Y).dot(p.K) + p.b - p.Y
    # print_loss(p)
    return 1
    
def examine_example(i, p):
    '''
    given the first (alpha) example, examine the potential second example
    if found a good second example, return 1, else 0
    '''
    # if violate KKT conditions, go on, else return 0 (means we don't want to use this example) 
    if ((p.dists[i] < 0) and (p.alpha[i] < p.c)) or ((p.dists[i] > 0) and (p.alpha[i] > 0)):
        # hueristics1: find alpha j maximize |E_i - E_j|
        candidate_j = np.where((p.alpha<p.c)*(p.alpha>0)*(p.alpha!=p.alpha[i]))[0]
        if len(candidate_j)>0:
            j = np.argmax(abs(p.dists-p.dists[i]))
            if take_step(i, j, p):
                return 1
        # hueristics2: loop over unbound examples
            np.random.shuffle(candidate_j)
            for j in candidate_j:
                if take_step(i, j, p):
                    return 1
        # hueristics3: loop over all examples
        candidate_j = list(set(range(p.m)) - set(candidate_j))
        candidate_j.remove(i)
        np.random.shuffle(candidate_j)
        for j in candidate_j:
            if take_step(i, j, p):
                return 1
    return 0

def main_routine(p, max_iter=5):
    num_changed = 0
    examine_all = 1
    passes = 0
    while(passes <= max_iter):
        num_changed = 0
        if (examine_all == 1):
            for i in range(p.m):
                num_changed += examine_example(i, p)
        else:
            candidate_i = np.where((p.alpha>0)*(p.alpha<p.c))[0]
            for i in candidate_i:
                # since alpha may change so we needs to check unbound at every step
                if (p.alpha[i] > 0) and (p.alpha[i] < p.c):
                    num_changed += examine_example(i, p)
        if (num_changed == 0):
            passes += 1
        # elif (num_changed > 0):
        #     passes = 0
        if (examine_all == 1):
            examine_all = 0
        elif (num_changed == 0):
            examine_all = 1
        # print(num_changed)
        
def print_loss(p):
    Y1 = p.Y[:,None]
    alpha1 = p.alpha[:,None]
    Loss = np.sum(alpha1) - 1/2*np.sum(Y1.dot(Y1.T)*alpha1.dot(alpha1.T)*p.K)
    print(Loss)
    
def check_step(i,j, p):
    '''
    This function only give a determination if i and j are suitable but won't change alpha and b
    Only used to debug
    '''
    alpha, K, Y, dists, c, tol, b = p.alpha.copy(), p.K, p.Y, p.dists.copy(), p.c, p.tol, p.b
    eta = K[i,i] + K[j,j] - 2*K[i,j]
    unclipped_alphaj = alpha[j] + Y[j]*(dists[i] - dists[j])/eta
    
    if Y[i] != Y[j]:
        L = np.max([0, alpha[j] - alpha[i]])
        H = np.min([c, c + alpha[j] - alpha[i]])

    else:
        L = np.max([0, alpha[j] + alpha[i] - c])
        H = np.min([c, alpha[j] + alpha[i]])
    if L == H:
        return 0
    
    clipped_alphaj = H if unclipped_alphaj > H else L if unclipped_alphaj < L else unclipped_alphaj
    clipped_alphai = alpha[i] + Y[i]*Y[j]*(alpha[j] - clipped_alphaj)
    
    if abs(clipped_alphai - alpha[i]) < tol:
        return 0
    bj = -dists[j] - Y[i]*K[i,j]*(clipped_alphai - alpha[i]) - Y[j]*K[j,j]*(clipped_alphaj - alpha[j]) + b
    bi = -dists[i] - Y[i]*K[i,i]*(clipped_alphai - alpha[i]) - Y[j]*K[i,j]*(clipped_alphaj - alpha[j]) + b
    if 0 < clipped_alphai < c:
        b = bi
    elif 0 < clipped_alphaj < c:
        b = bj
    else:
        b = (bi + bj)/2 
    alpha[i] = clipped_alphai
    alpha[j] = clipped_alphaj
    dists = (alpha*Y).dot(K) + b - Y

    return 1