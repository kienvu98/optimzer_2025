'''
file triển khai bài toán với pytorch phục vụ so sánh với code tự triển khai
logistic đơn giản
'''
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class Logistic(nn.Module):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dense = nn.Linear(self.input_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.dense()
        return self.sigmoid(x)
    
def main(X_train, y_train):

    batch_size = 40000

    # loader data chuẩn bị train
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # model
    list_lr = [0.1, 0.2, 0.3, 0.4, 0.5]
    for lr in list_lr:
        model = Logistic(768,1)
        loss = nn.BCELoss()
        optimzer = optim.SGD(model.parameters(),lr=lr)
        a = optim.
