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
        self.predictions = predictions
        self.targets = targets
        return np.mean((self.targets - self.predictions) ** 2)
    
    
    def backward(self):
        return 2 * (self.predictions - self.targets) / self.targets.shape[0]
    
    

class BinaryCrossEntropy(Loss):
    
    '''
    class triển khai loss binary cross entropy phân loại nhị phân
    loss = mean ( tagrets * log(predicts) + (1 - targets) * log(1 - predicts))
    '''
    
    def forward(self, predictions, targets):
        self.predictions = np.clip(predictions, 1e-7, 1 - 1e-7) # giá trị nhỏ nhất là 1e-7 và lớn nhất là 0,9999 tránh việc log cho 0 và 1
        self.targets = targets
        loss = - np.mean(self.targets * np.log(self.predictions) + (1 - self.targets) * np.log(1 - self.predictions))
        return loss
    
    def backward(self):
        return (self.predictions - self.targets) / self.targets.shape[0]



class CrossEntropy(Loss):
    
    '''
    class triển khai loss cross entropy cho phân loại nhiều đối tượng
    loss = 
    '''
    
    def forward(self, predictions, targets):
        return super().forward(predictions, targets)
    
    def backward(self):
        return super().backward()
    
