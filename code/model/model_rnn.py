from optimzer_project.code.backend.backend import xp as np
from optimzer_project.code.backend.backend import is_gpu_enable
from optimzer_project.code.model.model import Layer

class RNN_Cpu(Layer):
    '''
    class triển khai RNN với Cpu
    '''

    def __init__(self):
        super().__init__()

