from optimzer_project.code.backend.backend import xp as np
from optimzer_project.code.model.sequential import Model
from optimzer_project.code.model.model_conv import Conv2D_Cpu, MaxPool2D_Cpu, AvgPool2D_Cpu, Flatten
from optimzer_project.code.model.model import Relu, Dense
from optimzer_project.code.minst_dataset.mnist_dataset import MNIST_Dataset, DataLoader
from optimzer_project.code.loss.loss import CrossEntropy, LossWrapper
from optimzer_project.code.optimzer.optimzer import Adam
from optimzer_project.code.model.trainer import Trainer
from optimzer_project.code.backend.utils import accuracy_score, sofmax_labels
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
    
    # model test CNN
    model = Model([
        Conv2D_Cpu(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, name="Conv2D_1"),
        Relu(name="Relu_1"),
        MaxPool2D_Cpu(kernel_size=2, stride=2, name="MaxPool2D_1"),
        Conv2D_Cpu(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, name="Conv2D_2"),
        Relu(name="Relu_2"),
        MaxPool2D_Cpu(kernel_size=2, stride=2, name="MaxPool2D_2"),
        Flatten(name="Flatten"),
        Dense(in_features=64*7*7, out_features=128, name="Desne_1"),
        Relu(name="Relu_3"),
        Dense(in_features=128, out_features=10, name="Desne_2"),    
    ])
    
    model.summary()
    loss = CrossEntropy()
    loss_warpper = LossWrapper(loss, model)
    optimzer = Adam()
    
    # Dataset & DataLoader
    train_dataset = MNIST_Dataset(train=True)
    test_dataset = MNIST_Dataset(train=False)

    #train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    #val_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    train_loader = DataLoader.from_arrays(train_dataset, batch_size=256)
    val_loader = DataLoader.from_arrays(test_dataset, batch_size=256, shuffle=False)
    
    X, y = train_dataset[0]
    
    print(y.shape)
    
    trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimzer=optimzer,
            val_loader=val_loader,
            loss=loss_warpper,
            predict_fn=sofmax_labels,
            accuracy_fn=accuracy_score,
            epochs=3000,
        )
    
    trainer.fit()
    
main()