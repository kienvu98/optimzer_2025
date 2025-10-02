from optimzer_project.code.backend.backend import xp as np
from optimzer_project.code.model.sequential import Model
from optimzer_project.code.model.model_conv import Conv2D_Cpu


def main():
    
    #model = Model([
    #    Conv2D_Cpu(in_channels=1, out_channels=32, kernel_size=3)
    #])
    #model.summary()
    a = Conv2D_Cpu(in_channels=1, out_channels=32, kernel_size=3)
    a.num_params()
    x = np.random.rand(1, 1, 7, 7)
    print(a.forward(x).shape)

main()