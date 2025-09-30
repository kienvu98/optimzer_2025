from optimzer_project.code.backend.backend import xp as np
from optimzer_project.code.backend.backend import is_gpu_enable
from optimzer_project.code.model.model import Layer
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
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # trọng số của mô hình conv 2D
        scale = np.sqrt(2.0 / (self.in_channels * np.prod(self.kernel_size)))
        self.W = np.random.rand(out_channels, in_channels, *self.kernel_size) * scale
        self.d = np.zeros(self.out_channels)

        # gradient
        self.dW = None
        self.db = None
        self.cache = None

        # lưu thông tin từng tâng để lấy được để tính đạo hàm và cập nhập trọng số
        self.name = name or f"Conv2D_{id(self)}"
        self.grad_W = f"{self.name}_W"
        self.grad_b = f"{self.name}_b"



    def forward(self, inputs):
        '''
        hàm triển khai forward của CNN
        '''
        return super().forward(inputs)
    
    
    def backward(self, grad_out):
        return super().backward(grad_out)

