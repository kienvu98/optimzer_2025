from optimzer_project.code.backend.backend import xp as np

from abc import ABC, abstractmethod


class Optimzer(ABC):
    
    '''
    class trừu tượng, abstraclass
    '''
    
    def __init__(self, lr=0.01):
        self.lr = lr
    
    @abstractmethod
    def update(self, param, grad, key=None):
        pass
    
  
  
class GD(Optimzer):
    
    '''
    class triển khai thuật toán gradient descent
    '''
        
    def update(self, param, grad, key=None):
        return param - self.lr * grad
    


class SGD(Optimzer):
    
    '''
    class triển khai thuật toán SGD
    '''
    
    def update(self, param, grad, key=None):
        return param - self.lr * grad
    
 
    
class Momentum(Optimzer):
    
    '''
    class triển khai thuật toán Momentun
    ct thuật toán
    1. cập nhập velocity
        v(t) = momentum * v(t-1) - lr * gradient(t)
    2. cập nhập tham số
        x(t+1) = x(t) + v(t)
    '''
    
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity = {}
        
    def update(self, param, grad, key=None):
        if key is None:
            raise ValueError("Momentum optimzer requires a unique key.") # truyền để phân biệt là tính velocity cho W hay b
        
        # khởi tạo velocity
        if key not in self.velocity:
            self.velocity[key] = np.zeros_like(grad)
        
        # cập nhập velocity
        self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grad
        
        # cập nhập tham số
        return param + self.velocity[key]
    
    
    
class Adam(Optimzer):
    
    '''
    class triển khai thuật toán adam
    ct thuật toán
    1. cập nhập momentum bậc 1
        m(t) = beta_1 * m(t-1) + (1 - beta_1) * gradient(t)
    2. cập nhập momentum bậc 2
        v(t) = beat_2 * v(t-1) + (1 - beta_1) * gradient(t) ** 2
    3. bias correction
        m_hat(t) = m(t) / (1 - beta_1 ** t)
        v_hat(t) = v(t) / (1 - beta_2 ** t)
    4. cập nhập tham số
        x(t+1) = x(t) - lr * m_hat(t) / (sqrt(v_hat(t) + epsilon))
    '''
    
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = {} # momemtum bậc 1
        self.v = {} # momemtum bậc 2
        self.t = {} # thời gian
        
    def update(self, param, grad, key=None):
        if key is None:
            raise ValueError("Adam optimzer requires a unique key.") # truyền để phân biệt là tính velocity cho W hay b
        
        # Khởi tạo m, v, t
        if key not in self.m:
            self.m[key] = np.zeros_like(grad)
            self.v[key] = np.zeros_like(grad)
            self.t[key] = 0
            
        self.t[key] += 1
        
        # tính momentum bậc 1
        self.m[key] = self.beta_1 * self.m[key] + (1 - self.beta_1) * grad
        
        # tính momentum bậc 2
        self.v[key] = self.beta_2 * self.v[key] + (1 - self.beta_2) * (grad ** 2)
        
        # bias correction
        m_hat = self.m[key] / (1 - self.beta_1 ** self.t[key])
        v_hat = self.v[key] / (1 - self.beta_2 ** self.t[key])
        
        # cập nhập tham số
        return param - self.lr * (m_hat /(np.sqrt(v_hat) + self.epsilon))
    


    