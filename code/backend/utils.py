from optimzer_project.code.backend.backend import xp as np
from matplotlib import pyplot as plt

def split_arrays(X, y, ratio=0.8, shuffe=True):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    
    if shuffe:
        indices = np.random.permutation(indices)
        
    split_point = int(num_samples * ratio)
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    return X_train, y_train, X_test, y_test


def sigmoid_to_label(probabilities, threshold=0.5):
    return (probabilities > threshold).astype(int)

def sofmax_labels(probabilities):
    return np.argmax(probabilities, axis=1)


def accuracy_score(y_true, y_predict):
    return np.mean(y_true == y_predict).astype(np.float32)

def plot_metrics(trainer, save_path):
    epochs = range(1, len(trainer.train_loss_list) + 1)

    # Chuyển dữ liệu về dạng float
    train_loss_list = [float(loss) for loss in trainer.train_loss_list]
    val_loss_list = [float(loss) for loss in trainer.val_loss_list]
    train_acc_list = [float(acc) for acc in trainer.train_acc_list]
    val_acc_list = [float(acc) for acc in trainer.val_acc_list]

    # Tạo figure với 2 hàng, 2 cột
    plt.figure(figsize=(12, 8))

    # Ô 1: Train Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss_list, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()

    # Ô 2: Validation Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_loss_list, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.grid(True)
    plt.legend()

    # Ô 3: Train Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_acc_list, label='Train Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.grid(True)
    plt.legend()

    # Ô 4: Validation Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_acc_list, label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"📁 Biểu đồ đã được lưu tại: {save_path}")
    
    
def plot_metrics_optmzer(dict_lr, save_path):
    '''
    vẽ biểu đồ loss của mỗi lr
    '''
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.tab10.colors  # có 10 màu phân biệt rõ
    for i, (label, values) in enumerate(dict_lr.items()):  # dùng enumerate
        epochs = range(1, len(values) + 1)
        train_loss_list = [float(loss) for loss in values]

        plt.plot(epochs, train_loss_list, label=f"lr={label}", color=colors[i % len(colors)])  # mỗi đường có màu tự động khác nhau

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Optimizer Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Biểu đồ đã được lưu tại: {save_path}")