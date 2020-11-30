import numpy as np

'''sequential minimal optimization with heuristics'''

def train_svm_smo(X, Y, c, kernel, max_iter, *para):
    '''
    Train SVM by SMO.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - Y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    - c: A float number repesenting the penalty c of soft margin SVM.
    - kernel: A function calculating kernel matrix.
    - max_iter: A int number representing the maximal iterations. If the 
      parameter don't change for max_iter-time iterations, end the loop.
    - *para； parameters may be used to calculate kernel matrix.
      
    Output:
    - alpha: the optimal alpha for the dual problem
    - b: the optimal bias for the primal problem
    '''
    # initialize parameters
    state = {}
    m, n = X.shape
    K = kernel(X, X, *para)
    alpha = np.zeros(m)
    b = 0
    it = 0
    
    # if didn't change for max_iter iterations break
    while it < max_iter:
        # pair_changed: parameter to record whether the mutiplier pair changed
        pair_changed = 0
        # initialize clipped_alphaj to make abs(clipped_alphaj - alpha[j0]) > 0.001 True
        clipped_alphaj = 1
        dists = (alpha*Y).dot(K) + b - Y
        # first find alphas which violate KKT conditions(complentary slackness) most
        viol_KKT_i = range(m) if it ==0 else np.where(dists[(alpha>0)*(alpha<c)] != 0)[0] 
        for i in viol_KKT_i:
            dists = (alpha*Y).dot(K) + b - Y
            # second find alphas which gives a best step size.
            j0 = np.argmax(abs(dists-dists[i]))
            # if this change nothing, choose unbounded alpha_j
            j = j0 if abs(clipped_alphaj - alpha[j0]) > 0.001 else\
                np.random.choice(np.where((alpha>0)*(alpha<c)*(alpha!=i))[0],1)
            eta = K[i,i] + K[j,j] - 2*K[i,j]
            if eta <= 0:
                  print('WARNING  eta <= 0')
                  continue
            unclipped_alphaj = alpha[j] + Y[j]*(dists[i] - dists[j])/eta
            
            if Y[i] != Y[j]:
                L = np.max([0, alpha[j] - alpha[i]])
                H = np.min([c, c + alpha[j] - alpha[i]])

            else:
                L = np.max([0, alpha[j] + alpha[i] - c])
                H = np.min([c, alpha[j] + alpha[i]])

            clipped_alphaj = H if unclipped_alphaj > H else L if unclipped_alphaj < L else unclipped_alphaj
            clipped_alphai = alpha[i] + Y[i]*Y[j]*(alpha[j] - clipped_alphaj)
            # if change nothing, continue
            if abs(clipped_alphaj - alpha[j]) < 0.001:
                #print('WARNING   alpha_j not moving enough')
                continue

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
            Y1 = Y[:,None]
            alpha1 = alpha[:,None]
            # Loss = np.sum(alpha1) - 1/2*np.sum(Y1.dot(Y1.T)*alpha1.dot(alpha1.T)*K)
            # print(it, Loss, np.sum(alpha > 0.001), np.sum(Y*alpha))
            
            pair_changed += 1
            print('iteration:{}  i:{}  pair_changed:{}'.format(it, i, pair_changed))
        
        # it: parameter to record how many times pair didn't change
        if pair_changed == 0:
            it += 1
        else:
            it = 0
        print('iteration number: {}'.format(it))
    
    # another way to compute b
    # ind = np.argmax(alpha)
    # b = Y[ind] - (alpha*Y).dot(K[ind,:])
    state['alpha'] = alpha
    state['b'] = b
    state['Xtrain'] = X
    state['Ytrain'] = Y
    state['kernel'] = kernel
    return state


def predict_svm(state, X1, *para):
    '''
    Make prediction of svm.

    Parameters
    ----------
    state : A dictionary of trained parameters
    'alpha' is trained alpha
    'b' is trained b
    'Xtrain' is the training data will be used to caculate kernel
    'Y' is the label of training data used to caculate kernel
    'kernel' is the used kernel in training.
    X1 : A numpy array of shape (num_test, D) containing testing data
      consisting of num_test samples each of dimension D.
    *para : parameters may be used to calculate kernel matrix.

    Returns
    -------
    y_pred : A numpy array of shape (num_test,) containing predicted
    value (directional margin) of svm

    '''
    alpha, b, X, Y, kernel = state.values()
    K = kernel(X, X1, *para)
    y_pred = (alpha*Y).dot(K) + b
    
    return y_pred
    
