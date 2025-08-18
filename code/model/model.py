import numpy as np
import cupy as cp

class Layer:
    def forward(self, inputs):
        ''' hàm lan truyền xuôi
        '''
        raise NotImplementedError
    
    def backward(self, grad_out):
        '''
        hàm lan truyền ngược
        grad_out: kết quả đạo hàm của tầng trên
        '''
        raise NotImplementedError
    
    def step(self):
        '''
        hàm cập cập tham số theo ct tối ưu
        '''
        raise NotImplementedError



class Dense(Layer):
    '''
    layer linear Z = X * W  + b
    '''
    
    def __init__(self, in_features, out_features, optimzer):
        '''
        in_features: chiều dữ liệu đầu vào (chiều của data)
        out_features: chiều dữ liệu đầu ra (chiều của hidden)
        '''
        self.W = np.random.rand(in_features, out_features) * 0.01 # trọng số của layer có chiều [in_features, out_features]
        self.b = np.zeros((1, out_features)) # bias
        self.optimzer = optimzer
        
    def forward(self, inputs):
        '''
        input: [batch_size, in_features]
        output: [batch_size, out_features] = inputs x W + b
        '''
        self.inputs = inputs
        outputs = inputs @ self.W + self.b
        return outputs
    
    def backward(self, grad_output):
        self.dW = self.inputs.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        
        # gradient trả về cho layer phía trước
        grad_input = grad_output @ self.W.T
        return grad_input
    
    def step(self):
        self.W = self.optimzer.update(self.W, self.dW)
        self.b = self.optimzer.update(self.b, self.db)
   
   
   
class Relu(Layer):
    '''
    layer activetion relu để kích hoạt phi tuyến sau layer linear
    Z = max(X, 0)
    '''
    
    def forward(self, inputs):
        self.inputs = inputs
        output = np.maximum(inputs, 0)
        return output
    
    def backward(self, grad_out):
        return grad_out * (self.inputs > 0)
    
    def step(self):
        pass
    
    
    
class Sigmoid(Layer):
    
    '''
    layer activertion sigmoid để kích hoạt phi tuyến
    Z = 1/ (1 + exp(-Z))
    trong mạng thì Z  = Dense()
    '''
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(self.inputs))
        return self.outputs
    
    def backward(self, grad_out):
        return grad_out * self.outputs * (1 - self.outputs)
    
    def step(self):
        pass
        