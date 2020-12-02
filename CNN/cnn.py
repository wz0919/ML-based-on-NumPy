from layers import *

class Block():
    '''
    Basic block for cnn.
    
    Architecture:
    conv - relu - bn - max pool
    '''
    def __init__(self, in_channels, out_channels):
        self.conv = Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn   = BatchNorm2d(out_channels)
        self.relu = ReLU()
        self.pool = MaxPool2d(2)
        
    def forward(self, x, train = True):
        out, conv_cache = self.conv.forward(x)
        out, bn_cache   = self.bn.forward(out, train)
        out, relu_cache = self.relu.forward(out)
        out, pool_cache = self.pool.forward(out)
        cache = conv_cache, bn_cache, relu_cache, pool_cache
        
        return out, cache
    
    def backward(self, dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        dout = self.pool.backward(dout, pool_cache)
        dout = self.relu.backward(dout, relu_cache)
        dout, dgamma, dbeta = self.bn.backward(dout, bn_cache)
        dout, dw, db = self.conv.backward(dout, conv_cache)
        grad = dgamma, dbeta, dw, db
        
        return dout, grad
    
    def update(self, grad, lr = 0.01):
        dgamma, dbeta, dw, db = grad
        self.conv.update(dw, db, lr)
        self.bn.update(dgamma, dbeta, lr)
    
        
class miniCNN():
    '''
    A miniCNN used to test the code.
    
    Architecture:
    [conv - relu - bn - max pool ] * 3
    affine - relu - affine - softmax

    expect input size: (N,3,32,32)
    
    class number is 10
    expect output size: (N,10)
    '''
    def __init__(self):
        self.block1 = Block(3,8)
        self.block2 = Block(8,16)
        self.block3 = Block(16,32)
        self.fc1     = Linear(512, 64)
        self.relu   = ReLU()
        self.fc2    = Linear(64, 10)
        
    def forward(self, x, train = True):
        out, block1_cache = self.block1.forward(x, train)
        out, block2_cache = self.block2.forward(out, train)
        out, block3_cache = self.block3.forward(out, train)
        out, fc1_cache    = self.fc1.forward(out)
        out, relu_cache   = self.relu.forward(out)
        out, fc2_cache    = self.fc2.forward(out)
        cache = block1_cache, block2_cache, block3_cache, fc1_cache, relu_cache, fc2_cache
        
        return out, cache
    
    def backward(self, dout, cache):
        block1_cache, block2_cache, block3_cache, fc1_cache, relu_cache, fc2_cache = cache
        dout, dw2, db2 = self.fc2.backward(dout, fc2_cache)
        dout           = self.relu.backward(dout, relu_cache)
        dout, dw1, db1 = self.fc1.backward(dout, fc1_cache)
        dout, grad_bk3 = self.block3.backward(dout, block3_cache)
        dout, grad_bk2 = self.block2.backward(dout, block2_cache)
        dout, grad_bk1 = self.block1.backward(dout, block1_cache)
        grad = dw1, db1, dw2, db2, grad_bk3, grad_bk2, grad_bk1
        
        return dout, grad
    
    def update(self, grad, lr = 0.01):
        dw1, db1, dw2, db2, grad_bk3, grad_bk2, grad_bk1 = grad
        self.block1.update(grad_bk1, lr)
        self.block2.update(grad_bk2, lr)
        self.block3.update(grad_bk3, lr)
        self.fc1.update(dw1, db1, lr)
        self.fc2.update(dw2, db2, lr)