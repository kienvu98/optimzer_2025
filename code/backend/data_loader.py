from optimzer_project.code.backend.backend import xp as np

class DataLoader:
    '''
    class triển khai data loader, load dữ liệu chia batch để trainning
    '''
    
    def __init__(self, X, y, batch_size=64, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = self.X.shape[0]
        self.indices = np.arange(self.num_samples)
    
    def __iter__(self):
        if self.shuffle:
            self.indices = np.random.permutation(self.num_samples)
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration
        end = min(self.current_idx + self.batch_size, self.num_samples)
        batch_idx = self.indices[self.current_idx:end]
        batch_X = self.X[batch_idx]
        batch_y = self.y[batch_idx]
        self.current_idx = end
        return batch_X, batch_y
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    

    @classmethod
    def from_arrays(cls, X, y, batch_size=64, shuffle=True):
        return cls(X, y, batch_size=batch_size, shuffle=shuffle)
    
    