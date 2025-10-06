import os
import gzip
import urllib.request
from optimzer_project.code.backend.backend import xp as np
from typing import Tuple, Optional, Callable, List
import random


# download dữ liệu minst từ trang chủ của yan-lecun

def download_mnist(data_dir: str="./data"):
    '''
    hàm tải bộ dữ liệu minst
    '''
    os.makedirs(data_dir, exist_ok= True)
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    
    for fname in files.values():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(base_url + fname, path)
    print("✅ MNIST downloaded.")
    
    
def load_mnist_images(path: str) -> np.ndarray:
    '''
    load dữ liệu ảnh lên
    '''
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28).astype(np.float32) / 255.0
    return data


def load_mnist_labels(path: str) -> np.ndarray:
    '''
    load dữ liệu nhãn của ảnh minst
    '''
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data.astype(np.int64)



class MNIST_Dataset:
    '''
    class triển khai download và load data set minst
    '''
    def __init__(self, root: str = "./data", train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = True):
        if download:
            download_mnist(root)
            
        if train:
            images_path = os.path.join(root, "train-images-idx3-ubyte.gz")
            labels_path = os.path.join(root, "train-labels-idx1-ubyte.gz")
        else:
            images_path = os.path.join(root, "t10k-images-idx3-ubyte.gz")
            labels_path = os.path.join(root, "t10k-labels-idx1-ubyte.gz")
            
        self.images = load_mnist_images(images_path)
        self.labels = load_mnist_labels(labels_path)  
        self.transform = transform
        self.target_transform = target_transform
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform[y]
        return x, y
    
    

class DataLoader:
    '''
    class triển khai data loader
    '''
    
    def __init__(self, dataset, batch_size=32, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
    def __iter__(self):
        idxs = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(idxs)
        
        for i in range(0, len(idxs), self.batch_size):
            batch_idx = idxs[i:i+self.batch_size]
            if len(batch_idx) < self.batch_size and self.drop_last:
                continue
            
            batch = [self.dataset[j] for j in batch_idx]
            xs, ys = zip(*batch)
            xs = np.stack(xs, axis=0) # (batch, channel, 28, 28)
            ys = np.array(ys, dtype=np.int64)
            
            yield xs, ys
            
    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size    
    
        
        
