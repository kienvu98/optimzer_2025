from optimzer_project.code.backend.backend import xp as np

STATE_DICT = "state_dict"

class Model:
    
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
        self._backup = {} # phục vụ backtracking linse search
        
    
    def forward(self, x):
        '''
        forward qua từng layer trong list
        '''
        for layer in self.layers:
            #print(layer.name)
            x = layer.forward(x)
            #print(x.shape)
        return x
    
    
    def backward(self, grad_out):
        '''
        backward qua từng layer
        grad_out: kết quả của đạo hàm ở tầng trước
        '''
        for layer in reversed(self.layers):
            #print(layer.name)
            grad_out = layer.backward(grad_out)
            #print(grad_out.shape)
        return grad_out
    
    
    def step(self):
        '''
        cập nhập tham số cho từng layer nếu có
        '''
        for layer in self.layers:
            layer.step()
            
    
    def summary(self):
        '''
        hàm show ra model
        '''
        print("Model Summary")
        print("-" * 90)
        print(f"{'Idx':<5}{'Layer':<15}{'Name':<20}{'Input':<20}{'Output':<20}{'#Params':>8}")
        print("-" * 90)
        
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_type = layer.__class__.__name__
            name = getattr(layer, "name", f"{layer_type}_{id}")
            in_shape = getattr(layer, "input_shape", lambda: "-")()
            out_shape = getattr(layer, "output_shape", lambda: "-")()
            num_params = getattr(layer, "num_params", lambda: "-")()
            if (isinstance(num_params, int)):
                total_params += num_params
            print(f"{i:<5}{layer_type:<15}{name:<20}{str(in_shape):<20}{str(out_shape):<20}{num_params:>8}")
        
        print("-" * 90)
        print(f"Total parameters: {total_params}")
        
        
    def state_dict(self):
        '''
        hàm dùng để lưu lại thông layer
        '''
        state = dict()
        for layer in self.layers:
            if hasattr(layer, STATE_DICT):
                state[layer.name] = layer.state_dict()
                
        return state
    
    
    def load_state_dict(self, state):
        '''
        set lại và lưu thông tin khi load model
        '''
        for layer in self.layers:
            if layer.name in state:
                layer.load_state_dict(state[layer.name])
                 
                
    def get_total_grad_norm(self):
        total_norm = 0.0
        for layer in self.layers:
            if hasattr(layer, 'dW'):
                total_norm += np.linalg.norm(layer.dW) ** 2
            if hasattr(layer, 'db'):
                total_norm += np.linalg.norm(layer.db) ** 2
        total_norm = np.sqrt(total_norm)
              
        return total_norm
    
    
    def get_params(self):
        '''
        get_params để line search
        duỗi các tham số để phục vụ line search
        '''
        flat_params = []
        for layer in self.layers:
            if hasattr(layer, "get_params"):
                for p in layer.get_params():
                    flat_params.append(p.flatten())
        return np.concatenate(flat_params)
    
    
    def set_params(self, flat_params):
        """
        Gán lại tham số từ vector phẳng
        """
        offset = 0
        for layer in self.layers:
            if hasattr(layer, "get_params") and hasattr(layer, "set_params"):
                params = layer.get_params()
                if params is not None:
                    new_params = []
                    for p in params:
                        size = p.size
                        new_p = flat_params[offset:offset+size].reshape(p.shape)
                        new_params.append(new_p)
                        offset += size
                    # gán lại cho layer
                    layer.set_params(new_params)
    
    
    def get_grads(self):
        '''
        get grads của line search
        Lấy gradient của toàn bộ model thành vector phẳng
        (sau khi đã backward)
        '''
        flat_grads = []
        for layer in self.layers:
            if hasattr(layer, 'dW') and layer.dW is not None:
                flat_grads.append(layer.dW.flatten())
            if hasattr(layer, 'db') and layer.db is not None:
                flat_grads.append(layer.db.flatten())
        return np.concatenate(flat_grads)
    
    
    def get_params_not_line_search(self):
        '''
        get_params để line search
        duỗi các tham số để phục vụ line search
        '''
        flat_params = []
        for layer in self.layers:
            if hasattr(layer, "get_params"):
                for p in layer.get_params():
                    flat_params.append(p.flatten())
        return np.concatenate(flat_params)