from optimzer_project.code.backend.backend import xp as np
from tqdm import tqdm
import time

class Trainer:
    '''
    class triển khai training
    '''
    def __init__(self, model, train_loader, val_loader, loss, predict_fn, accuracy_fn, epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss = loss
        self.predict_fn = predict_fn
        self.accuracy_fn = accuracy_fn
        self.epochs = epochs
        
    
    def train_epoch(self):
        '''
        hàm trên từng epoch với tập train
        '''
        total_loss = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        
        # tung batch trong 1 epoch
        for x_batch, y_batch in tqdm(self.train_loader, desc="🔄 Training", leave=False):
            # forward qua model
            out_put = self.model.forward(x_batch)
            
            # tính loss 
            loss_value = self.loss.forward(out_put, y_batch)
            total_loss += loss_value * y_batch.shape[0]
            total_samples += y_batch.shape[0]
            
            # dự đoán nhãn và tính toán accuracy
            predicts = self.predict_fn(out_put)
            all_preds.append(predicts)
            all_targets.append(y_batch)
            
            # backward
            grad = self.loss.backward()
            self.model.backward(grad)
            # cập nhập lại trọng số
            self.model.step()
        
        avg_loss = total_loss / total_samples
        y_true = np.concatenate(all_targets)
        y_predicts = np.concatenate(all_preds)
        acc = self.accuracy_fn(y_true, y_predicts)
        return avg_loss, acc
    
    
    def validate(self):
        '''
        hàm trên tạp validation 
        sẽ không có quá trình backward và cập nhập trọng số mô hình
        '''
        total_loss = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        
        # tung batch trong 1 epoch
        for x_batch, y_batch in tqdm(self.val_loader, desc="🔄 Training", leave=False):
            # forward qua model
            out_put = self.model.forward(x_batch)
            
            # tính loss 
            loss_value = self.loss.forward(out_put, y_batch)
            total_loss += loss_value * y_batch.shape[0]
            total_samples += y_batch.shape[0]
            
            # dự đoán nhãn và tính toán accuracy
            predicts = self.predict_fn(out_put)
            all_preds.append(predicts)
            all_targets.append(y_batch)
            
        avg_loss = total_loss / total_samples
        y_true = np.concatenate(all_targets)
        y_predicts = np.concatenate(all_preds)
        acc = self.accuracy_fn(y_true, y_predicts)
        return avg_loss, acc
    
    
    def fit(self, patience=10, min_delta=1e-4):
        '''
        hàm gọi để thực hiện trainning model
        - patience: số epoch cho phép không cải thiện
        - min_delta: mức cải thiện tối thiểu để được tính là "giảm"
        '''
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []
        
        best_val_loss = float('inf')
        wait = 0  # số epoch không cải thiện
        for epoch in range(self.epochs):
            print(f"\n📘 Epoch {epoch+1}/{self.epochs}")
            start_time = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            elapsed = time.time() - start_time
            print(f"📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | ⏱️ Time: {elapsed:.2f}s")
            
            self.train_loss_list.append(train_loss)
            self.val_loss_list.append(val_loss)
            self.train_acc_list.append(train_acc)
            self.val_acc_list.append(val_acc)

            # Kiểm tra điều kiện early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                wait = 0  # reset bộ đếm
            else:
                wait += 1
                print(f"Val loss không cải thiện ({wait}/{patience})")

            if wait >= patience:
                print(f"\n Dừng sớm tại epoch {epoch+1} do val loss không cải thiện sau {patience} epoch.")
                break
        