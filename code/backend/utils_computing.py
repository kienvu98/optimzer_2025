from optimzer_project.code.backend.backend import xp as np

class UtilComputing:
    '''
    class này lưu các hàm chung phụ vụ tính toán có thể dùng đi dùng lại được
    '''
    
    @classmethod
    def im2col(cls, x, kernel_size, stride, padding):
        '''
        hàm triển khai biến data dạng image thành matrix giúp convulotion thành phép dot-product --> tăng tốc độ
        '''
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
        cols = cols.transpose(0,4,5,1,2,3).reshape(N*out_H*out_W, C*kH*kW)
        return cols, out_H, out_W
    

    @classmethod
    def col2im(cls, x, cols, kernel_size, stride, padding):
        '''
        hàm triển khai biến col thành image để phục vụ backward
        '''
        N, C, H, W = x.shape
        kH, kW= kernel_size

        # trả lại đúng chiều và data từ cols --> image (data sau khi qua convulotion)
        cols_reshape = cols.reshape(N, kH, kW, C, H, W).transpose(0,3,4,5,1,2)
        dx_padded = np.zeros((N, C, H + padding * 2, W + padding * 2))

         # lấy thông tin số lượng cửa sổ trượt
        out_H = (H + 2 * padding - kH) // stride + 1
        out_W = (W + 2 * padding - kW) // stride + 1

        # đưa từng patch gradient về vị trí đúng trong dx_padded
        for i in range(kH):
            i_end = i + stride * out_H
            for j in range(kW):
                j_end = j + stride * out_W
                dx_padded[:, :, i:i_end:stride, j:j_end:stride] += cols_reshape[:, :, i, j, :, :]
                
        # cắt bỏ padding để cũng cỡ với data gốc
        dx = dx_padded[:, :, padding:H + padding, padding:W + padding]
        return dx

