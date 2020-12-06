from kernel import *
from smo import *

class SVM():
    def __init__(self, C=1.0, kernel='rbf', max_iter= 5, tol = 0.001):
        '''
        C: the parameter c in objective \|w\|^2 + C\sum_{i = 1}^n \ξi
        kernel: the used kernel
        max_iter: after max_iter times no update aplphas, algorithm converges.
        '''
        self.c = C
        self.max_iter = max_iter
        self.tol = tol
        if kernel == 'rbf':
            self.kernel = gaussian_kernel
        if kernel == 'poly':
            self.kernel = polynomial_kernel
        self.alpha = None
        self.b = None
        
    def fit(self, X, Y, **para):
        '''
        expect X: (N,D)
        expect Y: (N,), +-1 label
        **para: necessary parameter for kernels
        '''
        p = smo_para_and_func(X, Y, self.c, self.tol, self.kernel, **para)
        main_routine(p, self.max_iter)
        self.para = para
        self.alpha = p.alpha
        self.b = p.b
        self.X = X
        self.Y = p.Y
        
    def predict(self, X1):
        K = self.kernel(self.X, X1, **self.para)
        y_pred = (self.alpha*self.Y).dot(K) + self.b
        
        return y_pred
    
    

        
               