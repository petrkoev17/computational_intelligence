# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np
import pandas as pd

import Perceptron as pc


def sig_derivative(z):
    return (1 - z) * z

def apply_softmax(layer, input):
    zets = []
    for perceptron in layer:
        perceptron.z = np.dot(perceptron.weights, input) + perceptron.bias
        zets.append(perceptron.z)

    softmax = np.exp(zets)/np.sum(np.exp(zets), 0)
    for index, result in enumerate(softmax):
        layer[index].output = result
    return softmax

class ANN:
    def __init__(self, sizes):
        self.sizes = sizes
        self.nn = []

        for index, layer_size in enumerate(sizes):
            layer = []
            if index == 0:
                continue
            for perceptrons in range(0, layer_size):
                layer.append(pc.Perceptron(bias = 0.1, size = sizes[index-1]))
            self.nn.append(layer)

    def forward_propagation(self, input):
        for layer_index, layer in enumerate(self.nn):
            layer_results = []
            if layer_index != len(self.nn) - 1:
                for perceptron in layer:
                    current_perceptron_output = perceptron.calculate_output(input)
                    layer_results.append(current_perceptron_output)
            else:
                layer_results = apply_softmax(layer, input)
            input = layer_results

        return layer_results

    def backward_propagation(self, input, y_true, y_pred, learning_rate):

        target = np.zeros(7)
        target[y_true-1] = 1

        error = y_pred - target
        previous_derivatives = error
        previous_weights = []

        for index, layer in reversed(list(enumerate(self.nn))):
            if index == 0:
                prev_layer_res = input
            else:
                prev_layer_res = []
                for perceptron in self.nn[index-1]:
                    prev_layer_res.append(perceptron.output)
                    previous_weights.append(perceptron.weights)
                previous_weights = previous_weights[0:7]

            temp_prev_derivatives = []
            temp_prev_weights = []
            for perceptron_index, perceptron in enumerate(layer):
                temp_prev_weights.append(perceptron.weights)
                if index == len(self.nn) - 1:
                    perceptron.der = np.array(prev_layer_res)*np.sum(error)
                    perceptron.bias_der = np.sum(error)
                    temp_prev_derivatives = error
                else:
                    current_previous_weights = np.array(list(map(lambda x: x[perceptron_index], previous_weights)))
                    temp_prev_derivatives.append(perceptron.relu_derivative(perceptron.z) * np.dot(current_previous_weights, previous_derivatives))
                    perceptron.der = np.dot(temp_prev_derivatives[-1], prev_layer_res)
                    perceptron.bias_der = np.sum(temp_prev_derivatives[-1])
                perceptron.weights = perceptron.weights - learning_rate * perceptron.der
                perceptron.bias = perceptron.bias - learning_rate * perceptron.bias
            previous_derivatives = temp_prev_derivatives
            previous_weights = temp_prev_weights

    def fit_nn(self, X, Y, epochs, learning_rate):
        for epoch in range(0, epochs):
            for index, training_sample in enumerate(X):
                y_true = int(Y[index])
                y_pred = self.forward_propagation(training_sample)
                self.backward_propagation(training_sample, y_true, y_pred, learning_rate)

    def fit_nn_detailed(self, X, Y, epochs, learning_rate):
        accuracies = []
        for epoch in range(0, epochs):
            counter = 0
            for index, training_sample in enumerate(X):
                y_true = int(Y[index])
                y_pred = self.forward_propagation(training_sample)
                self.backward_propagation(training_sample, y_true, y_pred, learning_rate)
                y_pred = np.argmax(y_pred) + 1
                if y_pred == y_true:
                    counter = counter + 1
            accuracies.append(counter/len(X))
        return accuracies

    def toString(self):
        for index, layer in enumerate(self.nn):
            index = index + 1
            print("Layer " + str(index) + ": ")
            for perceptron in layer:
                perceptron.toString()

    def categorical_cross_entropy(self, y_pred, y_true):
        loss = 0
        for i in range(0, len(y_pred)):
            loss = loss + y_true[i] * np.log(y_pred[i])
        loss = loss * -1
        return loss

    def default_loss(self, y_pred, y_true):
        return np.sum(y_true - y_pred)

    def cross_entropy_derivative(self, y_pred, y_true):
        derivative = 0
        for i in range(0, len(y_pred)):
            loss = loss + y_true[i] / y_pred[i]
        loss = loss * -1
        return loss

    
