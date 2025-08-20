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
    loss = - mean ( tagrets * log(predicts) + (1 - targets) * log(1 - predicts))
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
    class triển khai loss cross entropy cho phân loại nhiều đối lớp
    loss = - mean(log (exp{z_i, y_i} / exp{z_ij}))
    '''
    
    def forward(self, logits, targets):
        self.logits = logits
        self.targets = targets
        
        # zj = zj - max(zk) trừ để tránh tràn số khi mũ
        shifted_logits = self.logits - np.max(self.logits, axis=1, keepdims=True)
        
        # tính tổng các dự đoán sum(exp{zj})
        log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=1, keepdims=True))
        
        log_probs = shifted_logits - log_sum_exp # vì log 2 vế exp{zj/sum(zj)} --> zj - sum(zj)
        
        batch_size = logits.shape[0]
        
        if targets.ndim == 2: # nếu targets là one-hot vector
            correct_log_probs  = np.sum(targets * log_probs, axis=1)
        else: 
            correct_log_probs = log_probs[np.arange(batch_size), targets]
            
        loss = -np.mean(correct_log_probs)
        
        # lưu softmax lại để phục vụ backward
        self.probs =  np.exp(log_probs)
        return loss
    
    
    def backward(self):
        grad = self.probs.copy()
        batch_size = self.logits.shape[0]
        
        if self.targets.ndim == 2: # nếu targets là one-hot vector
            grad -= self.targets
        else:
            grad[np.arange(batch_size), self.targets] -= 1
            
        return grad / batch_size
    
