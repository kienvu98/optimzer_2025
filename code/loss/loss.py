import numpy as np
import cupy as cp
from abc import ABC, abstractmethod


class Loss(ABC):
    
    @abstractmethod
    def forward(self, predictions, targets):
        ''''
        tính toán giá trị hàm mục tiêu
        '''
        pass
    
    @abstractmethod
    def backward(self):
        '''
        tính đạo hàm của hàm mục tiêu
        '''
        pass



class MSELoss(Loss):
    
    '''
    class triển khai loss mean square error
    loss = mean ((predicts - targets) ** 2)
    '''
    
    def forward(self, predictions, targets):
        return super().forward(predictions, targets)
    
    
    def backward(self):
        return super().backward()
    
    

class BinaryCrossEntropy(Loss):
    
    '''
    class triển khai loss binary cross entropy phân loại nhị phân
    loss = mean ( tagrets * log(predicts) + (1 - targets) * log(1 - predicts))
    '''
    
    def forward(self, predictions, targets):
        return super().forward(predictions, targets)
    
    def backward(self):
        return super().backward()