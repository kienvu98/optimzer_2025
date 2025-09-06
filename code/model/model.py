from optimzer_project.code.backend.backend import xp as np

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
    
    def state_dict(self):
        '''
        hàm để lưu thông tin của từng layer --> lưu model
        '''
        
    def load_state_dict(self, state):
        '''
        hàm load thông tin của layer, set lại trọng số cho layer --> load model
        '''



class Dense(Layer):
    '''
    layer linear Z = X * W  + b
    '''
    
    def __init__(self, in_features, out_features, optimzer, name=None):
        '''
        in_features: chiều dữ liệu đầu vào (chiều của data)
        out_features: chiều dữ liệu đầu ra (chiều của hidden)
        optimzer: đối tượng thuật toán tối ưu
        '''
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features) # trọng số của layer ma trận có kích cỡ [in_features, out_features]
        self.b = np.zeros((1, out_features)) # bias
        self.optimzer = optimzer
        self.name = name or f"Dense_{id(self)}"
        self.grad_W = f"{self.name}_W"
        self.grad_b = f"{self.name}_b"
        
    def forward(self, inputs):
        '''
        input: [batch_size, in_features]
        output: [batch_size, out_features] = inputs x W + b
        '''
        self.inputs = inputs
        outputs = inputs @ self.W + self.b
        return outputs
    
    def backward(self, grad_output):
        #print("------", grad_output.shape)
        #print("------", self.inputs.shape)
        #print(self.W.shape)
        self.dW = self.inputs.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        #print("------", self.dW.shape)
        
        
        # gradient trả về cho layer phía trước
        grad_input = grad_output @ self.W.T
        return grad_input
    
    def step(self):
        self.W = self.optimzer.update(self.W, self.dW, self.grad_W)
        self.b = self.optimzer.update(self.b, self.db, self.grad_b)
        
    
    def state_dict(self):
        return  {
            'W': self.W,
            'b': self.b
        }
        
    
    def load_state_dict(self, state):
        self.W = state['W']
        self.b = state['b']
        
        
    def output_shape(self):
        '''
        hàm lấy thông tin out_features phục vụ summary 
        '''
        return (None, self.out_features)
    

    def input_shape(self):
        '''
        hàm lấy thông tin in_features phục vụ summary
        '''
        return (None, self.in_features)
    
    
    def num_params(self):
        '''
        hàm tính tổng số trọng số qua lớp Dense
        '''
        return self.W.size + self.b.size
   
   
   
class Relu(Layer):
    '''
    layer activetion relu để kích hoạt phi tuyến sau layer linear
    Z = max(X, 0)
    '''
    
    def __init__(self, name=None):
        self.name = name or f"Relu_{id(self)}"
    
    def forward(self, inputs):
        self.inputs = inputs
        output = np.maximum(inputs, 0)
        return output
    
    def backward(self, grad_out):
        return grad_out * (self.inputs > 0)
    
    def step(self):
        pass
    
    def state_dict(self):
        pass
    
    def load_state_dict(self, state):
        pass
    
    
    
class Sigmoid(Layer):
    
    '''
    layer activertion sigmoid để kích hoạt phi tuyến sau layer linear
    Z = 1/ (1 + exp(-Z))
    trong mạng thì Z  = Dense()
    '''
    
    def __init__(self, name=None):
        self.name = name or f"Sigmoid_{id(self)}"
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-self.inputs))
        return self.outputs
    
    def backward(self, grad_out):
        return grad_out * self.outputs * (1 - self.outputs)
    
    def step(self):
        pass
        
    def state_dict(self):
        pass
    
    def load_state_dict(self, state):
        pass
      
        
class Dropout(Layer):
    
    '''
    layer dropout để tắt 1 cập nhập học 1 số param khi trainning
    Tránh overfiting khi trainning
    Khi suy luận sẽ không tắt
    '''
    
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        return super().forward(inputs)
    
    