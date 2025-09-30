from optimzer_project.code.backend.backend import xp as np
from cupy.cuda import cudnn
from optimzer_project.code.model.model import Layer

class Conv2D_Cpu(Layer):
    '''
    class triển khai convulotion 2D bằng cpu
    '''
    
    def __init__(self):
        super().__init__()