from optimzer_project.code.backend.backend import xp as np

class UtilComputing:
    '''
    class này lưu các hàm chung phụ vụ tính toán có thể dùng đi dùng lại được
    '''
    
    @classmethod
    def im2col(cls, x, kernel_size, stride, padding):
        N, C, H, W = x.shape # lấy thông tin chiều của dữ liệu
        kH, kW = kernel_size
        
        # lấy thông tin số lượng cửa sổ trượt
        out_H = (H + 2 * padding - kH) // stride + 1
        out_W = (W + 2 * padding - kW) // stride + 1
        
        # padding theo các chiều của dữ liệu
        x_padded = np.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)))
        
        cols = np.zeros((N, C, kH, kW, out_H, out_W))
        for i in range(kH):
            i_end = i + stride * out_H
            for j in range(kW):
                j_end = j + stride * out_W
                cols[:, :, i, j, :, :] = x_padded[:, :, i:i_end:stride, j:j_end:stride]
                
        # reshape thành (N*out_H*out_W, C*kH*kW)
        cols = cols.transpose(0,4,5,1,2,3).reshape(N*out_H*out_W, -1)
        return cols, out_H, out_W