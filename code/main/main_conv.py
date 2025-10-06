from optimzer_project.code.backend.backend import xp as np
from optimzer_project.code.model.sequential import Model
from optimzer_project.code.model.model_conv import Conv2D_Cpu, MaxPool2D_Cpu, AvgPool2D_Cpu, Flatten


def main():
    
    #model = Model([
    #    Conv2D_Cpu(in_channels=1, out_channels=32, kernel_size=3)
    #])
    #model.summary()
    a = Conv2D_Cpu(in_channels=1, out_channels=32, kernel_size=3)
    b = AvgPool2D_Cpu(kernel_size=2)
    c = Flatten()
    x = np.random.rand(2, 1, 7, 7)
    x = a.forward(x)
    print(x.shape)
    x = b.forward(x)
    print(x.shape)
    x = c.forward(x)
    print(x.shape)

main()