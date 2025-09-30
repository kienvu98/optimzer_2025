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
import os
import json
from scipy.optimize import line_search

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available()) 

class Logistic(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Logistic, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dense = nn.Linear(self.input_dim, self.output_dim)
        #self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.dense(x)
        #return self.sigmoid(x)
        return x
    
def main_train_sgd_gd(X_file, y_file, folder, patience=20):

    X_train = np.array(np.load(X_file))
    y_train = np.array(np.load(y_file))

    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    #y_train = y_train.view(1, -1)

    print(X_train.device)
    print(y_train.shape)

    batch_size = 40000

    # loader data chuẩn bị train
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    name = "GD"

    # model
    list_lr = [0.2]
    for lr in list_lr:
        model = Logistic(768,1).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimzer = optim.SGD(model.parameters(),lr=lr)
        
        best_val_loss = float('inf')
        dict_time_loss = {}
        epoch_num = 0
        wait = 0
        
        time_start_train = time.time()
        for epoch in range(3000):
            start_time = time.time()
            total_loss = 0.0
            for xb, yb in dataloader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                # Forward
                preds = model(xb)
                loss = criterion(preds, yb)

                # Backward
                optimzer.zero_grad()
                loss.backward()
                optimzer.step()

                total_loss += loss.item() * xb.size(0)

            avg_loss = total_loss / len(dataloader.dataset)
            
            
            min_delta = 1e-3
            epoch_num += 1
            if epoch > 100:
                if best_val_loss - avg_loss > min_delta:
                    best_val_loss = avg_loss
                    wait = 0
                else:
                    wait += 1
                    print(f"Train loss không cải thiện ({wait}/{patience})")
                
                if wait >= patience:
                    print(f"Dừng sớm tại epoch {epoch+1} do train loss không cải thiện sau {patience} epoch.")
                    break
            
            time_train = time.time() - start_time
            print(f"Epoch {epoch+1}/{1000}, Loss: {avg_loss:.4f}, Time: {time_train}")
            
        dict_time_loss.setdefault(lr, {})['time'] = time.time() - time_start_train
        dict_time_loss.setdefault(lr, {})['loss_min'] = best_val_loss
        dict_time_loss.setdefault(lr, {})['iteration'] = epoch_num
        
        
        file_path= os.path.join(folder, name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dict_time_loss, f, ensure_ascii=False, indent=4)
            
            
            
def get_params(model):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])

def set_params(model, theta):
    idx = 0
    for p in model.parameters():
        size = p.numel()
        new_val = torch.from_numpy(theta[idx:idx+size].reshape(p.shape)).float()
        p.data.copy_(new_val)
        idx += size
        
        
# Hàm loss + grad cho SciPy
def make_loss_and_grad(model, loss_fn, x_batch, y_batch):
    def f(theta):
        set_params(model, theta)
        with torch.no_grad():
            loss = loss_fn(model(x_batch), y_batch).item()
        return loss

    def fprime(theta):
        set_params(model, theta)
        model.zero_grad()
        loss = loss_fn(model(x_batch), y_batch)
        loss.backward()
        grad = np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in model.parameters()])
        return grad

    return f, fprime
        

# Training loop với line search
def train_with_linesearch(model, loss_fn, x, y, epochs=50):
    for epoch in range(epochs):
        theta = get_params(model)
        f, fprime = make_loss_and_grad(model, loss_fn, x, y)

        grad = fprime(theta)
        pk = -grad  # hướng đi = -gradient

        # SciPy line search (Armijo + Wolfe)
        ls = line_search(f, fprime, theta, pk, grad)
        alpha = ls[0] if ls[0] is not None else 1e-3  # fallback nếu LS fail

        # Update tham số
        new_theta = theta + alpha * pk
        set_params(model, new_theta)

        # In loss
        loss_val = f(new_theta)
        print(f"Epoch {epoch+1}: Loss = {loss_val:.6f}, Step size = {alpha:.6f}")
        
        
        
def main_train_lbfgs(X_file, y_file, folder, patience=20):
    X_train = np.array(np.load(X_file))
    y_train = np.array(np.load(y_file))

    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    print(X_train.device)
    print(y_train.shape)

    batch_size = 40000
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    name = "LBFGS"
    list_lr = [1.0]   # thường LBFGS nên để lr=1.0

    for lr in list_lr:
        model = Logistic(768, 1).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.LBFGS(
            model.parameters(),
            lr=lr,
            max_iter=20,        # số lần lặp nội bộ mỗi step
            history_size=10,    # số vector lưu cho quasi-Newton
            line_search_fn="strong_wolfe"  # dùng line search Wolfe
        )

        best_val_loss = float("inf")
        dict_time_loss = {}
        epoch_num = 0
        wait = 0

        time_start_train = time.time()

        for epoch in range(3000):
            start_time = time.time()
            total_loss = 0.0

            # train theo batch (LBFGS thường dùng full batch, nhưng bạn vẫn có thể giữ DataLoader)
            for xb, yb in dataloader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                def closure():
                    optimizer.zero_grad()
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
                total_loss += loss.item() * xb.size(0)

            avg_loss = total_loss / len(dataloader.dataset)

            min_delta = 1e-3
            epoch_num += 1
            if epoch > 100:
                if best_val_loss - avg_loss > min_delta:
                    best_val_loss = avg_loss
                    wait = 0
                else:
                    wait += 1
                    print(f"Train loss không cải thiện ({wait}/{patience})")

                if wait >= patience:
                    print(f"Dừng sớm tại epoch {epoch+1} do train loss không cải thiện sau {patience} epoch.")
                    break

            time_train = time.time() - start_time
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Time: {time_train:.4f}")

        dict_time_loss.setdefault(lr, {})["time"] = time.time() - time_start_train
        dict_time_loss.setdefault(lr, {})["loss_min"] = best_val_loss
        dict_time_loss.setdefault(lr, {})["iteration"] = epoch_num

        file_path = os.path.join(folder, name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dict_time_loss, f, ensure_ascii=False, indent=4)
            

if __name__ == "__main__":
    #X_file = r'C:\Users\Vu Trung Kien\Desktop\optimzer\data\imdb_encoded_X.npy'
    #y_file = r'C:\Users\Vu Trung Kien\Desktop\optimzer\data\imdb_encoded_y.npy'
    X_file = '/workspace/data/data_optimzer_project_train/imdb_encoded_X.npy'
    y_file = '/workspace/data/data_optimzer_project_train/imdb_encoded_y.npy'
    folder = '/workspace/optimzer_project/pytorch_train'
    main_train_sgd_gd(X_file, y_file, folder)