'''
file triển khai bài toán với pytorch phục vụ so sánh với code tự triển khai
'''
import torch.nn as nn

class Pytorch(nn.Module):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dense = nn.Linear(self.input_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()
    
    

