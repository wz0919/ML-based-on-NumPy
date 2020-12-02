import numpy as np

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):     
        if type(kernel_size) == int:
            height, width = kernel_size, kernel_size
        elif type(kernel_size) == tuple:
            height, width = kernel_size
        self.stride = stride
        self.padding = padding
        he = np.sqrt(2/(height*width*in_channels))
        self.weights = he*np.random.randn(out_channels, in_channels, height, width)
        self.bias = he*np.random.randn(out_channels)
        self.w_velocity = np.zeros_like(self.weights)
        self.b_velocity = np.zeros_like(self.bias)
        
    def forward(self, x):
        weights = self.weights
        pad = self.padding
        stride = self.stride
        b = self.bias
        
        N, C, H, W = x.shape
        F, C, HH, WW = weights.shape
        padded_x = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
    
        H_new = int(1 + (H + 2 * pad - HH) / stride)
        W_new = int(1 + (W + 2 * pad - WW) / stride)
    
        out = np.zeros((N, F, H_new, W_new))
        for h in range(H_new):
          for w in range(W_new):
            out[:,:,h,w] = np.sum(padded_x[:,None,:,h*stride:h*stride+HH,\
              w*stride:w*stride+WW] * weights[None,:], axis = (2,3,4)) + b
        cache = x
        
        return out, cache
    
    def backward(self, dout, cache):
        x = cache
        weights = self.weights
        pad = self.padding
        stride = self.stride
        b = self.bias
        
        _, _, W_out, H_out = dout.shape
        F, C, HH, WW = weights.shape
        padded_x = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
        dw = np.zeros(weights.shape)
        dx = np.zeros(x.shape)
        db = np.zeros(b.shape)
        dpad_x = np.zeros(padded_x.shape)
        for w in range(W_out):
          for h in range(H_out):
            dw += np.sum(padded_x[:,None,:,h*stride:h*stride+HH,w*stride:w*stride+WW] * \
              dout[:,:,None,h:h+1,w:w+1], axis = 0)
            dpad_x[:,:,h*stride:h*stride+HH,w*stride:w*stride+WW] += \
              np.sum(weights[None,:] * dout[:,:,None,h:h+1,w:w+1], axis = 1)
            db += np.sum(dout[:,:,w,h], axis = 0)
        _, _, H, W = x.shape
        dx = dpad_x[:,:,pad:H+pad,pad:W+pad]
    
        return dx, dw, db
    
    def update(self, dw, db, lr = 0.01, momentum = 0.9):
        self.w_velocity = self.w_velocity*momentum - lr*dw
        self.b_velocity = self.b_velocity*momentum - lr*db
        self.weights += self.w_velocity
        self.bias += self.b_velocity

class ReLU():
    def forward(self, x):
        out = np.maximum(x, 0)
        cache = x
        return out, cache

    def backward(self, dout, cache):
        x = cache
        dx = dout * (np.maximum(x, 0) > 0)
        return dx
    
class BatchNorm1d():
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = 1*np.random.randn(num_features)
        self.beta = 1*np.random.randn(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.zeros(num_features)
        self.g_velocity = np.zeros_like(self.gamma)
        self.b_velocity = np.zeros_like(self.beta)
        
    
    def forward(self, x, train = True):
        cache = None
        if train == True:
            momentum = self.momentum
            sample_mean = np.mean(x, axis = 0, keepdims = True)
            sample_var = np.var(x, axis = 0, keepdims = True)
            x_n = (x - sample_mean)/(np.sqrt(sample_var + self.eps))
            out = self.gamma * x_n + self.beta
            self.running_mean = momentum * self.running_mean + (1 - momentum) * sample_mean
            self.running_var = momentum * self.running_var + (1 - momentum) * sample_var
            cache = x_n, self.gamma, np.sqrt(sample_var + self.eps)
        else:
            x_n = (x - self.running_mean)/(np.sqrt(self.running_var) + self.eps)
            out = self.gamma * x_n + self.beta
            
        return out, cache

    def backward(self, dout, cache):
        x_n, gamma, _std = cache
        N = x_n.shape[0]
        dgamma = np.sum(dout * x_n, axis = 0)
        dbeta = np.sum(dout, axis = 0)
        dx_hat = dout * gamma 
        dsigma = -0.5 * np.sum(dx_hat * x_n, axis= 0 ) / _std**2
        dmu = -np.sum(dx_hat / _std, axis= 0)
        dx = dx_hat / _std + 2.0 * dsigma * x_n * _std / N + dmu /N

        return dx, dgamma, dbeta
    
    def update(self, dgamma, dbeta, lr = 0.01, momentum = 0.9):
        self.g_velocity = self.g_velocity*momentum - lr*dgamma
        self.b_velocity = self.b_velocity*momentum - lr*dbeta
        self.gamma += self.g_velocity
        self.beta += self.b_velocity
    
class BatchNorm2d():
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        self.batch = BatchNorm1d(num_features, eps, momentum)
        self.eps = self.batch.eps
        self.momentum = self.batch.momentum
        self.gamma = self.batch.gamma
        self.beta = self.batch.beta
        self.running_mean = self.batch.running_mean
        self.running_var = self.batch.running_var
        self.g_velocity = np.zeros_like(self.gamma)
        self.b_velocity = np.zeros_like(self.beta)
    
    def forward(self, x, train = True):
        n, c, h, w = x.shape
        out, cache = self.batch.forward(x.transpose((0, 2, 3, 1)).reshape(-1, c),train)
        out = out.reshape(n, h, w, c).transpose(0, 3, 1, 2)

        return out, cache
    
    def backward(self, dout, cache):
        n, c, h, w = dout.shape
        dx, dgamma, dbeta = self.batch.backward(dout.transpose((0,2,3,1)).reshape((-1,c)), cache)
        dx = dx.reshape(n, h, w, c).transpose(0, 3, 1, 2)
        
        return dx, dgamma, dbeta
    
    def update(self, dgamma, dbeta, lr = 0.01, momentum = 0.9):
        self.g_velocity = self.g_velocity*momentum - lr*dgamma
        self.b_velocity = self.b_velocity*momentum - lr*dbeta
        self.gamma += self.g_velocity
        self.beta += self.b_velocity
    
class MaxPool2d():
    def __init__(self, kernel_size, stride = 2):     
        if type(kernel_size) == int:
            self.height, self.width = kernel_size, kernel_size
        elif type(kernel_size) == tuple:
            self.height, self.width = kernel_size
        self.stride = stride
        
    def forward(self, x):
        N, C, H, W = x.shape
        stride = self.stride
        H_new = int(1 + (H - self.height) / stride)
        W_new = int(1 + (W - self.width) / stride)
    
        out = np.zeros((N, C, H_new, W_new))
        for h in range(H_new):
          for w in range(W_new):
            out[:,:,h,w] = np.max(x[:,:,h*stride:h*stride+self.height,\
              w*stride:w*stride+self.width], axis = (2,3))
    
        cache = x
        return out, cache
    
    def backward(self, dout, cache):
        x = cache
        stride = self.stride
        _, _, H_out, W_out = dout.shape
    
        dx = np.zeros(x.shape)
        for h in range(H_out):
          for w in range(W_out):
            part_x = x[:,:,h*stride:h*stride+self.height, w*stride:w*stride+self.width]
            dx[:,:,h*stride:h*stride+self.height,w*stride:w*stride+self.width] += (part_x\
             == np.max(part_x, axis = (2,3), keepdims = True))*dout[:,:,h:h+1,w:w+1]
    
        return dx
    
class Dropout():
    def __init__(self, p = 0.5):     
        self.p = p
    
    def forward(self, x, train = True):
        mask = None
        if train == True:
            mask = (np.random.rand(*x.shape) < self.p) / self.p
            out = x * mask
        else:
            out = x

        cache = mask
        out = out.astype(x.dtype, copy=False)
        return out, cache
    
    def backward(self, dout, cache):
        mask = cache
        dx = dout * mask
        
        return dx
    
class Linear():
    def __init__(self, in_features, out_features):
        he = np.sqrt(2/(in_features))
        self.weights = he*np.random.randn(in_features, out_features)
        self.bias = he*np.random.randn(out_features)
        self.w_velocity = np.zeros_like(self.weights)
        self.b_velocity = np.zeros_like(self.bias)
        
    def forward(self, x):
        N = x.shape[0]
        flattened_x = np.reshape(x, (N, -1))
        out = np.dot(flattened_x, self.weights) + self.bias
        
        cache = x
        return out, cache
        
    def backward(self,dout, cache):
        x = cache
        N = x.shape[0]
        dx = np.reshape(dout.dot(self.weights.T), x.shape)
        dw = x.reshape((N, -1)).T.dot(dout)
        db = dout.T.dot(np.ones(N))

        return dx, dw, db
    
    def update(self, dw, db, lr = 0.01, momentum = 0.9):
        self.w_velocity = self.w_velocity*momentum - lr*dw
        self.b_velocity = self.b_velocity*momentum - lr*db
        self.weights += self.w_velocity
        self.bias += self.b_velocity


def cross_entropy(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    
    return loss, dx