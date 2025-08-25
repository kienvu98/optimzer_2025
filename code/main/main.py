import matplotlib.pyplot as plt
from optimzer_project.code.model.sequential import Model
from optimzer_project.code.model.model import Dense
from optimzer_project.code.model.model import Relu
from optimzer_project.code.model.model import Sigmoid
from optimzer_project.code.optimzer.optimzer import Momentum

def main():
    momentum = Momentum()
    
    model = Model([Dense(768, 256, momentum, 'Dense_1'), 
                   Relu('Relu_1'),
                   Dense(256, 1, momentum, 'Dense_2'),
                   Sigmoid('Sigmoid_1')])

    model.summary()

if __name__ == "__main__":
    main()
    