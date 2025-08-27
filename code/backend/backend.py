import numpy as _numpy

# kiểm tra xem máy có cudd_gpu hay không
try:
    import cupy as _cupy
    _gpu_available = _cupy.cuda.runtime.getDeviceCount() > 0
except:
    _cupy = None
    _gpu_available = False
    
    
# mạc định là GPU nếu có không thì dùng cpu
xp = _cupy if _gpu_available else _numpy

def set_backend(use_gpu: bool = False, verbose: bool = True):
    '''
    người dùng chọn gpu hoặc cpu cho ptoject
    '''
    
    global xp
    
    if use_gpu:
        if _cupy and _gpu_available:
            if verbose:
                print(" project run with gpu")
        
        else:
            xp = _numpy
            if verbose:
                print("gpu not available")
                
    else:
        xp = _numpy
        if verbose:
            print("project run with numpy")
            

def is_gpu_enable() -> bool:
    return xp.__name__=="cupy"
