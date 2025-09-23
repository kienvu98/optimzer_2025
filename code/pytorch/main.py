'''
file triển khai bài toán với pytorch phục vụ so sánh với code tự triển khai
logistic đơn giản
'''
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import time

class Logistic(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Logistic, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dense = nn.Linear(self.input_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.dense(x)
        return self.sigmoid(x)
    
def main(X_file, y_file):

    X_train = np.array(np.load(X_file))
    y_train = np.array(np.load(y_file))

    X_train = torch.tensor(X_train, dtype=torch.float32)  
    y_train = torch.tensor(y_train, dtype=torch.float32)

    #y_train = y_train.view(1, -1)

    print(X_train.shape)
    print(y_train.shape)

    batch_size = 40000

    # loader data chuẩn bị train
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # model
    list_lr = [0.01, 0.02, 0.03, 0.04, 0.05]
    for lr in list_lr:
        model = Logistic(768,1)
        criterion = nn.BCELoss()
        optimzer = optim.SGD(model.parameters(),lr=lr)
        
        for epoch in range(1000):
            start_time = time.time()
            total_loss = 0.0
            for xb, yb in dataloader:
                # Forward
                preds = model(xb)
                loss = criterion(preds, yb)

                # Backward
                optimzer.zero_grad()
                loss.backward()
                optimzer.step()

                total_loss += loss.item() * xb.size(0)

            avg_loss = total_loss / len(dataloader.dataset)
            time_train = time.time() - start_time
            print(f"Epoch {epoch+1}/{1000}, Loss: {avg_loss:.4f}, Time: {time_train}")


if __name__ == "__main__":
    X_file = r'C:\Users\Vu Trung Kien\Desktop\optimzer\data\imdb_encoded_X.npy'
    y_file = r'C:\Users\Vu Trung Kien\Desktop\optimzer\data\imdb_encoded_y.npy'
    main(X_file, y_file)