import matplotlib.pyplot as plt
from optimzer_project.code.model.sequential import Model
from optimzer_project.code.model.model import Dense
from optimzer_project.code.model.model import Relu
from optimzer_project.code.model.model import Sigmoid
from optimzer_project.code.optimzer.optimzer import Momentum, GD, Adam, LineSearch, Newton
from optimzer_project.code.backend.backend import is_gpu_enable
from optimzer_project.code.backend.utils import split_arrays, sigmoid_to_label, accuracy_score, plot_metrics_optmzer
from optimzer_project.code.backend.backend import xp as np
from optimzer_project.code.backend.data_loader import DataLoader
from optimzer_project.code.loss.loss import BinaryCrossEntropy, LossWrapper
from optimzer_project.code.model.trainer import Trainer
import time
import json
import os


def main(file_name_X, file_name_y, folder):
    
    #momentum = Momentum()
    
    #model = Model([Dense(768, 256, momentum, 'Dense_1'), 
    #               Relu('Relu_1'),
    #               Dense(256, 1, momentum, 'Dense_2'),
    #               Sigmoid('Sigmoid_1')])

    #model.summary()
    
    # load dữ liệu
    X = np.array(np.load(file_name_X))
    y= np.array(np.load(file_name_y))
    
    # chia tập dữ liệu thành train, test
    X_train, y_train, X_test, y_test = split_arrays(X, y, ratio=0.8)
    
    
    # khởi tạo loss 
   

    #list_lr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    #list_lr = [0.2]#, 0.3]
    #list_lr = [0.1, 0.2, 0.3, 0.4, 0.5] #, 0.6, 0.7, 0.8]
    list_lr = [1]
    # danh sách các thuật toán tối ưu
    optimizers = {
        #"Adam": Adam(),
        #"GD": GD()
        #"SGD": SGD(),
        #"Momentum": Momentum(),
        #"GD_LineSearch": GD_LineSearch(
        #   lr=1.0,
        #  rho=0.5,
        #  c=1e-4,
        #  loss_fn=lambda model: loss(model.forward(X_train), y_train),
        #  model=None # sẽ gắn sau
        #)
        "lineSearch": LineSearch(model=None, loss_fn=None,
                                direction="newton", lr=1, rho=0.5, c=1e-4, min_alpha=1e-8, 
                                max_iter=50, reuse_lr=True, cg_tol=1e-4, cg_maxiter=50)
        #"new_ton": Newton(model=None, loss_fn=None, lr=0.1, cg_tol=1e-4, cg_maxiter=50)
    }
    for name, optimizer in optimizers.items():
        dict_lr = {}
        dict_time_loss = {}
        for lr in list_lr :
            # tạo DataLoader
            if name == "GD" or name == "lineSearch" or name == "new_ton": 
                batch_size = 40000
            else:
                batch_size = 4096
                
            train_loader = DataLoader.from_arrays(X_train, y_train, batch_size=batch_size)
            val_loader = DataLoader.from_arrays(X_test, y_test, batch_size=batch_size, shuffle=False)
            optimizer.lr = lr    
            
            # Khởi tạo model mới cho mỗi optimizer
            model = Model([
                # Dense(768, 256, optimizer, name='Dense_1'),
                # Relu('Relu_1'),
                Dense(768, 1, optimizer, name='Dense_1'),
                Sigmoid('Sigmoid_1')
            ])
            
            model.summary()
            loss = BinaryCrossEntropy()
            loss_warpper = LossWrapper(loss, model)

            # Gán model vào optimizer nếu cần (cho GD_Backtracking)
            if hasattr(optimizer, 'model') and optimizer.model is None:
                optimizer.loss_fn = loss_warpper
                optimizer.model = model

            # Khởi tạo trainer
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                optimzer=optimizer,
                val_loader=val_loader,
                loss=loss_warpper,
                predict_fn=sigmoid_to_label,
                accuracy_fn=accuracy_score,
                epochs=2000
            )

            # Huấn luyện
            start_time = time.time()
            trainer.fit_line_search()
            
            time_train = time.time() - start_time
            dict_time_loss.setdefault(lr, {})['time'] = time_train
            dict_lr[lr] = trainer.train_loss_list
            loss_min = min(trainer.train_loss_list)
            dict_time_loss.setdefault(lr, {})['loss_min'] = loss_min.tolist()
            dict_time_loss.setdefault(lr, {})['iteration'] = trainer.epoch_num
        
        file_path= os.path.join(folder, name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dict_time_loss, f, ensure_ascii=False, indent=4)
            
        plot_metrics_optmzer(dict_lr, file_path)
    
    #traine = Trainer(model=model, train_loader=train_loader,
     #                val_loader=val_loader, loss=loss,  
     #                predict_fn=sigmoid_to_label, 
     #                accuracy_fn=accuracy_score, epochs=300)
    
    #traine.fit()
    #plot_metrics(traine)

if __name__ == "__main__":
    if is_gpu_enable():
        print('project run with gpu')
    else:
        print('project run with cpu')
    #X_file = '/workspace/data/data_optimzer_project_train/imdb_encoded_X.npy'
    #y_file = '/workspace/data/data_optimzer_project_train/imdb_encoded_y.npy'
    file_path = r'C:\Users\Vu Trung Kien\Desktop\optimzer\optimzer_project\code\folder_image_file_train'
    X_file = r'C:\Users\Vu Trung Kien\Desktop\optimzer\data\imdb_encoded_X.npy'
    y_file = r'C:\Users\Vu Trung Kien\Desktop\optimzer\data\imdb_encoded_y.npy'
    main(X_file, y_file, file_path)
    #X = np.array(np.load(X_file))
    #y= np.array(np.load(y_file))
    #print(X.shape)
    #X_train, y_train, X_test, y_test = split_arrays(X, y, ratio=0.8)
    #print(X_train.shape)