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
    
    def forward(self, predictions, targets):
        return super().forward(predictions, targets)
    
    
    def backward(self):
        return super().backward()
    
    

class BinartCrossEntropy(Loss):
    
    def forward(self, predictions, targets):
        return super().forward(predictions, targets)
    
    def backward(self):
        return super().backward()

    
    
