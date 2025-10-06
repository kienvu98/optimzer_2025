from optimzer_project.code.backend.backend import xp as np
from optimzer_project.code.backend.backend import is_gpu_enable
from optimzer_project.code.model.model import Layer
from optimzer_project.code.backend.utils_computing import UtilComputing
if is_gpu_enable():
    from cupy.cuda import cudnn


class Conv2D_Cpu(Layer):
    '''
    class triển khai convulotion 2D với cpu
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, name=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        # trọng số của mô hình conv 2D
        scale = np.sqrt(2.0 / (self.in_channels * np.prod(np.array(self.kernel_size))))
        self.W = np.random.rand(out_channels, in_channels, *self.kernel_size) * scale
        self.b = np.zeros(self.out_channels)

        # gradient
        self.dW = None
        self.db = None
        self.cache = None

        # lưu thông tin từng tâng để lấy được để tính đạo hàm và cập nhập trọng số
        self.name = name or f"Conv2D_{id(self)}"
        self.grad_W = f"{self.name}_W"
        self.grad_b = f"{self.name}_b"


    def forward(self, x):
        '''
        hàm triển khai forward của CNN
        '''
        self.x = x
        N, C, H, W = x.shape

        # tính im2cols duỗi data thành ma trận cỡ (N*out_H*out_W, C*kH*kW)
        cols, out_H, out_W  = UtilComputing.im2col(x=x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        # duỗi trọng số thành vector để thực hiện phép nhân với cols --> chính là phép convulution
        W_col = self.W.reshape(self.out_channels, -1)
        print(W_col.shape)
        print(cols.shape)

        out = cols.dot(W_col.T) + self.b

        # reshape trả lại kích cỡ out_put của data khi qua lớp tích chập
        out = out.reshape(N, out_H, out_W, self.out_channels).transpose(0, 3, 1, 2)

        # lưu lại cache để phục vụ backward
        self.cache = (cols, W_col, out_H, out_W)
        return out
    
    
    def backward(self, grad_out):
        '''
        hàm triển khai tính backward của CNN
        '''
        cols, W_col, out_H, out_W = self.cache
        #N, C, H, W = self.x.shape
        
        grad_out_reshape = grad_out.transpose(0,2,3,1).reshape(-1, self.out_channels)
        
        #gradient bias
        self.dW = grad_out_reshape.T.dot(cols).reshape(self.W.shape)
        
        # gradient input
        dcols = grad_out_reshape.dot(W_col)
        dx = UtilComputing.col2im(x=self.x, cols=cols, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        return dx
        
    
    def get_params(self): 
        '''
        trả về danh sách tham số
        '''
        return [self.W, self.b]
    
    
    def get_grads(self):
        '''
        trả về danh sách gradient tương ứng
        '''
        return [self.dW, self.db]
    
    
    def num_params(self):
        '''
        hàm tính tổng số trọng số qua lớp tích chập
        '''
        return self.W.size + self.b.size
    
    
    
class MaxPool2D_Cpu(Layer):
    '''
    class triển khai max pooling --> giảm kích cỡ ảnh lấy giá trị lớn nhất trong vị trí kernel trượt qua
    '''
    
    def __init__(self, kernel_size, stride=2, padding=0, name=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.name = name or f"MaxPooling2D_{id(self)}"
        self.cache = None
        
        
    def forward(self, x):
        '''
        hàm triển khai forward cho max pooling 2D
        '''
        self.x = x
        N, C, H, W = x.shape
        kH, kW = self.kernel_size
        
        # sử dụng im2col biến data dạng imag thành cols theo kernel
        cols, out_H, out_W = UtilComputing.im2col(x=x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        
        cols_original = cols.copy()
        # mỗi hàng của matrix col là 1 kernel --> chỉ cần lấy max trên mỗi hàng là được
        # lấy chỉ số của phần tử lớn nhất trong hàng
        cols = cols.reshape(N*out_H*out_W, C, kH*kW)
        self.argmax = np.argmax(cols, axis=2)
        
        # lọc phần tử max và thu nhỏ matrix
        out = np.max(cols, axis=2)
        
        # reshape lại kích cỡ đúng
        out = out.reshape(N, out_H, out_W, C).transpose(0, 3, 1, 2)
        self.cache = (cols_original, out_H, out_W)
        return out
    
    
    def backward(self, grad_out):
        '''
        hàm triển khai backward cho max poolng 2D
        '''
        
        cols, out_H, out_W = self.cache
        
        # chuyển grad_out về thành vector 1 chiều
        grad_out_flatten = grad_out.transpose(0, 2, 3, 1).ravel()
        
        # tạo ma trận cùng cỡ với cols
        dcols = np.zeros_like(cols)
        
        # gán gradient vào các vị trí max 
        dcols[np.arange(cols.shape[0]), self.argmax] = grad_out_flatten
        
        # biến về kích cỡ của ảnh của dx
        dx = UtilComputing.col2im(x=self.x, cols=dcols, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        return dx
    
    
    
class AvgPool2D_Cpu(Layer):
    '''
    class triển khai average pooling 2D --> giảm chiều data lấy trung bình các giá trị mà kernek trượt qua
    '''
    
    def __init__(self, kernel_size, stride=2, padding=0, name=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.name = name or f"AvgPooling2D_{id(self)}"
        self.cache = None
    
    
    def forward(self, x):
        '''
        hàm triển khai forward cho avg pooling
        '''
        self.x = x
        N, C, H, W = x.shape
        kH, kW = self.kernel_size
        
        # sử dụng im2col biến data dạng imag thành cols theo kernel
        cols, out_H, out_W = UtilComputing.im2col(x=x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        
        # mỗi hàng của matrix col là 1 kernel --> lấy trung bình trên từng hàng
        cols = cols.reshape(N*out_H*out_W, C, kH*kW) # reshape về dạng (N, matrix_path_kernel)
        out = np.mean(cols, axis=2)
        
        # reshape lại kích cỡ đúng
        out = out.reshape(N, out_H, out_W, C).transpose(0, 3, 1, 2)
        
        self.cache = (out_H, out_W)
        return out
         
         
    def backward(self, grad_out):
        '''
        Hàm triển khai backward cho avg pooling
        '''
        out_H, out_W = self.cache
        kH, kW = self.kernel_size
        
        # chuyển grad_out về thành vector 1 chiều
        grad_out_flatten = grad_out.transpose(0, 2, 3, 1).ravel()
        
        # phân bổ đêu các giá trị về các patch trong kernel
        dcols = np.repeat(grad_out_flatten[:, None], kH * kW,  axis=1) /  (kH * kW)
        
        # biến về kích cỡ của ảnh của dx
        dx = UtilComputing.col2im(x=self.x, cols=dcols, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        return dx
        
        
        
class Flatten(Layer):
    '''
    class triển khai flatten duỗi dữ liệu nhiều chiều thành vector
    '''
    
    def __init__(self, name=None):
        super().__init__()
        self.input_shape = None # để lưu chiều của để backward
        self.name = name or f"Flatten_{id(self)}"
        
    
    def forward(self, x):
        '''
        hàm triển khai forward
        '''
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    
    def backward(self, grad_out):
        '''
        hàm triển khai backward
        '''
        return grad_out.reshape(self.input_shape) # đạo hàm trả lại chiều