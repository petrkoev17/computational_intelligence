import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sb

import Perceptron as pc
import NeuralNetwork as nn

def singlePerceptronAnd():
    x_train = [[0,0],
               [0,1],
               [1,0],
               [1,1]]
    y_train = [0,0,0,1]
    single_perceptron = pc.Perceptron(bias = 0, size = 2)
    lReluAlpha = 0.01
    alpha = 0.01
    losses = []
    for epoch in range(0, 1500):
        # print("")
        loss_for_append = 0
        for index, sample in enumerate(x_train):
            input = sample
            y_true = y_train[index]
            single_perceptron.calculate_output_single_perceptron(input, alpha)
            update_weights_single_perceptron(single_perceptron, single_perceptron.output, y_true, alpha, lReluAlpha,
                                             input)
            # print("Epoch: " + str(epoch))
            # print("Input: " + str(sample))
            # print("Output: " + str(single_perceptron.output))
            # print("Bias: " + str(single_perceptron.bias))
            # print("weights: " + str(single_perceptron.weights))
            loss_for_append = loss_for_append + np.abs(y_true - single_perceptron.output)
        losses.append(loss_for_append / 4)
    plt.title("AND")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.show()

def singlePerceptronOr():
    x_train = [[0,0],
               [0,1],
               [1,0],
               [1,1]]
    y_train = [0,1,1,1]
    single_perceptron = pc.Perceptron(bias = 0, size = 2)
    lReluAlpha = 0.01
    alpha = 0.01
    losses = []
    for epoch in range(0, 1500):
        # print("")
        loss_for_append = 0
        for index, sample in enumerate(x_train):
            input = sample
            y_true = y_train[index]
            single_perceptron.calculate_output_single_perceptron(input, alpha)
            update_weights_single_perceptron(single_perceptron, single_perceptron.output, y_true, alpha, lReluAlpha,
                                             input)
            # print("Epoch: " + str(epoch))
            # print("Input: " + str(sample))
            # print("Output: " + str(single_perceptron.output))
            # print("Bias: " + str(single_perceptron.bias))
            # print("weights: " + str(single_perceptron.weights))
            loss_for_append = loss_for_append + np.abs(y_true - single_perceptron.output)
        losses.append(loss_for_append / 4)
    plt.title("OR")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.show()


def singlePerceptronXOR():
    x_train = [[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]]
    y_train = [0, 1, 1, 0]
    single_perceptron = pc.Perceptron(bias=0, size=2)
    lReluAlpha = 0.01
    alpha = 0.01
    losses = []
    for epoch in range(0, 1500):
        # print("")
        loss_for_append = 0
        for index, sample in enumerate(x_train):
            input = sample
            y_true = y_train[index]
            single_perceptron.calculate_output_single_perceptron(input, alpha)
            update_weights_single_perceptron(single_perceptron, single_perceptron.output, y_true, alpha, lReluAlpha,
                                             input)
            # print("Epoch: " + str(epoch))
            # print("Input: " + str(sample))
            # print("Output: " + str(single_perceptron.output))
            # print("Bias: " + str(single_perceptron.bias))
            # print("weights: " + str(single_perceptron.weights))
            loss_for_append= loss_for_append + np.abs(y_true - single_perceptron.output)
        losses.append(loss_for_append/4)
    plt.title("XOR")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.show()


def update_weights_single_perceptron(perceptron, y_pred, y_true, alpha, lReluAlpha, x_i):
    # loss = categorical_cross_entropy_single(y_pred, y_true)
    # derivative_loss = cross_entropy_derivative(y_pred, y_true)
    # derivative_activation = lRelu_derivative(perceptron.z, lReluAlpha)
    # perceptron.weights = perceptron.weights - alpha * derivative_loss * derivative_activation * np.array(x_i)
    # perceptron.bias = perceptron.bias - alpha * derivative_loss * derivative_activation
    loss = y_true - y_pred
    perceptron.weights = perceptron.weights + alpha*loss*np.array(x_i)
    perceptron.bias = perceptron.bias + alpha*loss


def plot_confusion_matrix(true, predicted):
    cm = confusion_matrix(true, predicted)
    heatmap = sb.heatmap(cm, cmap='Greens', fmt='g', annot=True)

    heatmap.set_title('Confusion Matrix for Test Set')
    heatmap.set_xlabel('True')
    heatmap.set_ylabel('Predicted')
    heatmap.xaxis.set_ticklabels(['1', '2', '3', '4', '5', '6', '7'])
    heatmap.yaxis.set_ticklabels(['1', '2', '3', '4', '5', '6', '7'])

    plt.show()

def findOptimalNeurons():
    optimal_Neurons = [7, 10, 15, 20, 30]
    
    input = np.genfromtxt("../data/features.txt", delimiter=",")
    Y = np.genfromtxt("../data/targets.txt")

    X_train, X_test, y_train, y_test = train_test_split(input, Y, test_size=0.2, random_state=42)


    accuracies = []

    for i in optimal_Neurons:
        neural_network = nn.ANN([10, i, 7])
        neural_network.fit_nn(X_train, y_train, epochs=20, learning_rate=0.01)
        predictions = []
        counter = 0
        for sample_index, sample in enumerate(X_test):
            curr_results = neural_network.forward_propagation(sample)
            y_pred = np.argmax(curr_results) + 1
            predictions.append(y_pred)
            if y_pred == y_test[sample_index]:
                counter = counter + 1
        accuracies.append(counter/len(X_test))

    fig, ax = plt.subplots()
    ax.plot(optimal_Neurons, accuracies)
    plt.xlabel("Number of neurons")
    plt.ylabel("Accuracy")
    plt.show()

def exercise_12():
    input = np.genfromtxt("../data/features.txt", delimiter=",")
    Y = np.genfromtxt("../data/targets.txt")

    X_train, X_test, y_train, y_test = train_test_split(input, Y, test_size=0.2, random_state=42)
    neural_network = nn.ANN([10, 15, 7])
    results = neural_network.fit_nn_detailed(X_train, y_train, 20, 0.01)
    epochs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    fig, ax = plt.subplots()
    ax.plot(epochs, results)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Performance over validation set")
    plt.show()

def crossValidate():
    optimal_Neurons = [7,10,15,20, 25]
    input = np.genfromtxt("../data/features.txt", delimiter=",")
    Y = np.genfromtxt("../data/targets.txt")

    X_train, X_test, y_train, y_test = train_test_split(input, Y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    scores = []
    n = len(X_train)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // 5
    for neurons in optimal_Neurons:
        neural_network = nn.ANN([10, neurons, 7])
        for i in range(5):
            neural_network.fit_nn(X_train, y_train, 5, 0.01)
            counter = 0
            for training_sample_index, training_sample in enumerate(X_val):
                curr_results = neural_network.forward_propagation(training_sample)
                y_pred = np.argmax(curr_results) + 1
                if y_pred == y_val[training_sample_index]:
                    counter = counter + 1
            print("Number of Neurons: " + str(neurons))
            print("Cross-Validation: " + str(i+1))
            print("Accuracy: " + str(counter/len(y_val)))





if __name__ == "__main__":
    # singlePerceptronOr()
    # singlePerceptronAnd()
    # singlePerceptronXOR()
    # findOptimalNeurons()
    # findOptimalNeurons()
    # exercise_12()
    crossValidate()

    # neural_network = nn.ANN([10, 16, 7])
    # input = np.genfromtxt("../data/features.txt", delimiter=",")
    # Y = np.genfromtxt("../data/targets.txt")
    # X_train, X_test, y_train, y_test = train_test_split(input, Y, test_size = 0.15, random_state = 42)
    # neural_network.fit_nn(X_train, y_train, epochs=100, learning_rate=0.01)
    # predictions = []
    # counter = 0
    # for sample_index, sample in enumerate(X_test):
    #     curr_results = neural_network.forward_propagation(sample)
    #     y_pred = np.argmax(curr_results) + 1
    #     predictions.append(y_pred)
    #     if y_pred == y_test[sample_index]:
    #         counter = counter + 1
    # print(counter / len(y_test))
