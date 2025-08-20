import numpy as np
import cupy as cp


class Sequential:
    
    '''
    class triển khai sequential giống pytorch và keras
    thuận tiện triển khai model 
    '''
    
    def __init__(self, layers):
        self.layers = []  # list các layer
        for i, layer in enumerate(layers):
            # tự động đặt tên các layer nếu chưa có tên
            if not hasattr(layer, "name") or layer.name is None:
                layer.name = f"{layer.__class__.__name__}_{i}"
            self.layers.append(layer)
        
    
    def forward(self, x):
        '''
        forward qua từng layer trong list
        '''
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    
    def backward(self, grad_out):
        '''
        backward qua từng layer
        grad_out: kết quả của đạo hàm ở tầng trước
        '''
        for layer in self.layers:
            grad_out = layer.backward(grad_out)
        return grad_out
    
    
    def step(self):
        '''
        cập nhập tham số cho từng layer nếu có
        '''
        for layer in self.layers:
            layer.step()
            
    
    def summary(self):
        '''
        hàm show ra 
        '''