from optimzer_project.code.backend.backend import xp as np

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