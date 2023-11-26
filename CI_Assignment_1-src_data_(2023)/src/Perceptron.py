import numpy as np

class Perceptron:
    def __init__(self, bias, size):
        self.bias = bias
        self.weights = np.random.randn(size)
        self.der = None
        self.output = None
        self.z = None
        self.bias_der = None

    def calculate_output(self, input):
        self.z = np.dot(input, self.weights) + self.bias
        # self.output = self.relu(self.z)
        self.output = self.z
        return self.output

    def calculate_output_single_perceptron(self, input):
        self.z = np.dot(input, self.weights) + self.bias
        self.output = self.stepFunction(self.z)
        return self.output

    def relu(self, z):
        return np.maximum(0,z)

    def relu_derivative(self, x):
        # return x > 0
        return 1

    def stepFunction(self, x):
        if x>0:
            return 1
        else:
            return 0

    def toString(self):
        print("Weights: " + str(self.weights))
        print("Bias: " + str(self.bias))


