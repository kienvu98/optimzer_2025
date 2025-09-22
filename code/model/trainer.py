from optimzer_project.code.backend.backend import xp as np
from tqdm import tqdm
import time

class Trainer:
    '''
    class triển khai training
    '''
    def __init__(self, model, optimzer, train_loader, val_loader, loss, predict_fn, accuracy_fn, epochs=10):
        self.model = model
        self.optimzer = optimzer
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
            self.loss.set_batch(x_batch, y_batch)
            #out_put = self.model.forward(x_batch)
            
            # tính loss 
            loss_value = self.loss()
            total_loss += loss_value * y_batch.shape[0]
            total_samples += y_batch.shape[0]
            
            # dự đoán nhãn và tính toán accuracy
            predicts = self.predict_fn(self.loss.predicts)
            all_preds.append(predicts)
            all_targets.append(y_batch)
            
            # backward
            grad = self.loss.backward()
            self.model.backward(grad)
            # cập nhập lại trọng số
            self.optimzer.step(self.model)
        
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
            self.loss.set_batch(x_batch, y_batch)
            #out_put = self.model.forward(x_batch)
            
            # tính loss 
            loss_value = self.loss()
            total_loss += loss_value * y_batch.shape[0]
            total_samples += y_batch.shape[0]
            
            # dự đoán nhãn và tính toán accuracy
            predicts = self.predict_fn(self.loss.predicts)
            all_preds.append(predicts)
            all_targets.append(y_batch)
            
        avg_loss = total_loss / total_samples
        y_true = np.concatenate(all_targets)
        y_predicts = np.concatenate(all_preds)
        acc = self.accuracy_fn(y_true, y_predicts)
        return avg_loss, acc
    
    
    def fit(self, patience=10, min_delta=1e-4, grad_threshold=1e-3, start_epoch=20):
        '''
        hàm gọi để thực hiện trainning model
        - patience: số epoch cho phép không cải thiện
        - min_delta: mức cải thiện tối thiểu để được tính là "giảm"
        '''
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []
        self.epoch_num = 0;
        
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
            #if val_loss < best_val_loss - min_delta:
            #    best_val_loss = val_loss
            #    wait = 0  # reset bộ đếm
            #else:
             #   wait += 1
              #  print(f"Val loss không cải thiện ({wait}/{patience})")

            #if wait >= patience:
            #    print(f"\n Dừng sớm tại epoch {epoch+1} do val loss không cải thiện sau {patience} epoch.")
            #    break
            if epoch > start_epoch:
                if min_delta < abs(best_val_loss - train_loss) :
                    best_val_loss = train_loss
                    wait = 0
                else:
                    wait += 1
                    print(f"Train loss không cải thiện ({wait}/{patience})")
                    
                grad_norm = self.model.get_total_grad_norm()
                if grad_norm < grad_threshold:
                    print(f"Dừng sớm tại epoch {epoch+1} do gradient quá nhỏ: {grad_norm:.2e}")
                    break

                if wait >= patience:
                    print(f"Dừng sớm tại epoch {epoch+1} do val loss không cải thiện sau {patience} epoch.")
                    break
            
            
            
    def fit_line_search(self, patience=20, min_delta=1e-3, grad_threshold=1e-3, start_epoch=20):
        """
        Train loop với optimizer LineSearch
        - patience: số epoch cho phép không cải thiện
        - min_delta: mức cải thiện tối thiểu để tính là cải thiện
        - grad_threshold: dừng sớm nếu chuẩn gradient quá nhỏ
        """
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []
        self.epoch_num = 0

        best_val_loss = float('inf')
        wait = 0

        for epoch in range(self.epochs):
            print(f"\n📘 Epoch {epoch+1}/{self.epochs}")
            start_time = time.time()

            # --- Train ---
            train_loss, train_acc = self.train_epoch()

            # --- Validation ---
            val_loss, val_acc = self.validate()

            elapsed = time.time() - start_time
            print(f"📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | ⏱️ Time: {elapsed:.2f}s")
        
            # Lưu lại history
            self.train_loss_list.append(train_loss)
            self.val_loss_list.append(val_loss)
            self.train_acc_list.append(train_acc)
            self.val_acc_list.append(val_acc)
            self.epoch_num += 1

            # --- Early stopping ---
            if epoch > start_epoch:
                if best_val_loss - train_loss > min_delta:
                    best_val_loss = train_loss
                    wait = 0
                else:
                    wait += 1
                    print(f"Train loss không cải thiện ({wait}/{patience})")

                # Kiểm tra gradient norm (model cần có get_total_grad_norm)
                grad_norm = self.model.get_total_grad_norm()
                if grad_norm < grad_threshold:
                    print(f"Dừng sớm tại epoch {epoch+1} do gradient quá nhỏ: {grad_norm:.2e}")
                    break

                if wait >= patience:
                    print(f"Dừng sớm tại epoch {epoch+1} do train loss không cải thiện sau {patience} epoch.")
                    break
        