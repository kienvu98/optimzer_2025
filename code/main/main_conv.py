from optimzer_project.code.backend.backend import xp as np
from optimzer_project.code.model.sequential import Model
from optimzer_project.code.model.model_conv import Conv2D_Cpu, MaxPool2D_Cpu, AvgPool2D_Cpu, Flatten
from optimzer_project.code.minst_dataset.mnist_dataset import MNIST_Dataset, DataLoader
import matplotlib.pyplot as plt


def show_mnist_sample(dataset, idx=0):
    """Hiển thị 1 ảnh trong dataset MNIST."""
    image, label = dataset[idx]
    if hasattr(image, 'get'):
        image = image.get()
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()
    plt.savefig("mnist_sample.png")
    
    
def main():
    
    #model = Model([
    #    Conv2D_Cpu(in_channels=1, out_channels=32, kernel_size=3)
    #])
    #model.summary()
    '''
    a = Conv2D_Cpu(in_channels=3, out_channels=32, kernel_size=3)
    b = AvgPool2D_Cpu(kernel_size=2)
    c = Flatten()
    x = np.random.rand(2, 3, 7, 7)
    x = a.forward(x)
    print(x.shape)
    x = b.forward(x)
    print(x.shape)
    x = c.forward(x)
    print(x.shape)
    '''
    # Dataset & DataLoader
    train_dataset = MNIST_Dataset(train=True)
    test_dataset = MNIST_Dataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Lấy 1 batch để test forward CNN
    for X_batch, y_batch in train_loader:
        # reshape: (B, 1, 28, 28) để dùng cho Conv2D
        X_batch = X_batch[:, None, :, :]  # thêm channel dimension
        print("Batch shape:", X_batch.shape)
        print("Label shape:", y_batch.shape)
        break 
    
    show_mnist_sample(train_dataset, idx=123)

main()