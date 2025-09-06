import matplotlib.pyplot as plt
from optimzer_project.code.model.sequential import Model
from optimzer_project.code.model.model import Dense
from optimzer_project.code.model.model import Relu
from optimzer_project.code.model.model import Sigmoid
from optimzer_project.code.optimzer.optimzer import Momentum
from optimzer_project.code.backend.backend import is_gpu_enable
from optimzer_project.code.backend.utils import split_arrays, sigmoid_to_label, accuracy_score, plot_metrics
from optimzer_project.code.backend.backend import xp as np
from optimzer_project.code.backend.data_loader import DataLoader
from optimzer_project.code.loss.loss import BinaryCrossEntropy
from optimzer_project.code.model.trainer import Trainer

def main(file_name_X, file_name_y):
    momentum = Momentum()
    
    
    model = Model([Dense(768, 256, momentum, 'Dense_1'), 
                   Relu('Relu_1'),
                   Dense(256, 1, momentum, 'Dense_2'),
                   Sigmoid('Sigmoid_1')])

    model.summary()
    
    # load dữ liệu
    X = np.array(np.load(file_name_X))
    y= np.array(np.load(file_name_y))
    
    # chia tập dữ liệu thành train, test
    X_train, y_train, X_test, y_test = split_arrays(X, y, ratio=0.8)
    
    # tạo DataLoader
    train_loader = DataLoader.from_arrays(X_train, y_train, batch_size=2048)
    val_loader = DataLoader.from_arrays(X_test, y_test, batch_size=2048, shuffle=False)
    
    # khởi tạo loss 
    loss = BinaryCrossEntropy()
    
    traine = Trainer(model=model, train_loader=train_loader,
                     val_loader=val_loader, loss=loss, 
                     predict_fn=sigmoid_to_label, 
                     accuracy_fn=accuracy_score, epochs=300)
    
    traine.fit()
    plot_metrics(traine)

if __name__ == "__main__":
    if is_gpu_enable():
        print('project run with gpu')
    else:
        print('project run with cpu')
    X_file = '/workspace/data/data_optimzer_project_train/imdb_encoded_X.npy'
    y_file = '/workspace/data/data_optimzer_project_train/imdb_encoded_y.npy'
    main(X_file, y_file)
    #X = np.array(np.load(X_file))
    #y= np.array(np.load(y_file))
    #print(X.shape)
    #X_train, y_train, X_test, y_test = split_arrays(X, y, ratio=0.8)
    #print(X_train.shape)