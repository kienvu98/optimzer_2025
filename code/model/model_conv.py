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
        return self.W.size  + self.b.size
    