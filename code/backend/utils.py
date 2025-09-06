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


def accuracy_score(y_true, y_predict):
    return np.mean(y_true == y_predict).astype(np.float32)

def plot_metrics(trainer, save_path="training_metrics.png"):
    epochs = range(1, len(trainer.train_loss_list) + 1)
    plt.figure(figsize=(10, 5))
    
    train_loss_list =  [float(loss) for loss in trainer.train_loss_list]
    val_loss_list = [float(loss) for loss in trainer.val_loss_list]

    # Biểu đồ Train Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs,train_loss_list, label='Train Loss', color='blue')
    plt.ylabel('Epoch')
    plt.xlabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()

    # Biểu đồ Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_loss_list, label='Validation Accuracy', color='green')
    plt.ylabel('Epoch')
    plt.xlabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)  # Lưu biểu đồ
    plt.show()
    print(f"📁 Biểu đồ đã được lưu tại: {save_path}")