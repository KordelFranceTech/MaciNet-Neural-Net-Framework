
from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions
from MaciNet.deep_learning import NeuralNetwork
from MaciNet.utils import train_test_split, to_categorical, normalize, Plot
from MaciNet.utils import get_random_subsets, shuffle_data, accuracy_score
from MaciNet.deep_learning.optimizers import StochasticGradientDescent, Adam, RMSprop, Adagrad, Adadelta
from MaciNet.deep_learning.loss_functions import CrossEntropy, SquareLoss
from MaciNet.utils.misc import bar_widgets
from MaciNet.deep_learning.layers import Dense, Dropout, Activation
from MaciNet import data_processing
import pandas as pd


def main():

    optimizer = Adam()

    # NEURAL NETWORK
    ###############################################################################################
    ###############################################################################################

    filename: str = 'VOCDataset.csv'

    # read each value, check for missing / erroneous data, and import the data into a pandas dataframe
    dataframe: pd.DataFrame = pd.read_csv(filename, header=None)

    # we need to cast the x- and y-vals as a new type for nn training
    # do it here
    x_vals_nn = dataframe.drop([16], 1)
    # x_vals_nn = dataframe.iloc[:, : 16]
    print(f'\n\tx vals: {x_vals_nn}')
    # for i in range(0, len(x_vals_nn.columns)):
    #     if i != 16:
    #         x_vals_nn.iloc[i].values[:] = 0

    y_vals_nn = dataframe[16]
    # y_vals_nn = dataframe.iloc[:, 16]
    print(f'\n\ty vals: {y_vals_nn}')
    # for i in range(0, len(y_vals_nn.columns)):
    #     if i == 16:
    #         y_vals_nn.iloc[i].values[:] = 0

    # now declare these vals as np array
    # this ensures they are all of identical data type and math-operable
    x_data_train = np.array(x_vals_nn)
    y_data_train = np.array(y_vals_nn)

    # data = datasets.load_digits()
    # X = data.data
    # y = data.target
    X = x_data_train
    y = y_data_train
    print(f'x.shape: {X.shape}\ny.shape: {y.shape}')

    # Convert to one-hot encoding
    y = to_categorical(y.astype("int"))

    n_samples, n_features = X.shape
    n_hidden = n_features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, seed=9)

    clf = NeuralNetwork(optimizer=optimizer,
                        loss=CrossEntropy,
                        validation_data=(X_test, y_test))

    clf.add(Dense(n_hidden, input_shape=(n_features,)))
    clf.add(Activation('relu'))
    clf.add(Dense(n_hidden, input_shape=(n_features,)))
    clf.add(Activation('relu'))
    # clf.add(Dropout(0.25))
    clf.add(Dense(n_hidden, input_shape=(n_features,)))
    clf.add(Activation('relu'))
    # clf.add(Dropout(0.25))
    clf.add(Dense(n_hidden, input_shape=(n_features,)))
    clf.add(Activation('relu'))
    # clf.add(Dropout(0.25))
    clf.add(Dense(y.shape[1], input_shape=(n_features,)))
    clf.add(Activation('softmax'))
    #
    # clf = NeuralNetwork(optimizer=optimizer, loss=CrossEntropy, validation_data=(X_test, y_test))
    # clf.add(Dense(n_hidden, input_shape=(n_features,)))
    # clf.add(Activation('sigmoid'))                        # not really a layer
    # clf.add(Dense(n_hidden, input_shape=(n_features,)))
    # clf.add(Activation('sigmoid'))                        # not really a layer
    # clf.add(Dense(n_hidden, input_shape=(n_features,)))
    # clf.add(Activation('sigmoid'))                        # not really a layer
    # clf.add(Dense(2, input_shape=(n_features,)))

    print ()
    clf.summary(name="MLP")
    
    train_err, val_err = clf.fit(X_train, y_train, n_epochs=500, batch_size=10)
    
    # Training and validation error plot
    n = len(train_err)
    training, = plt.plot(range(n), train_err, label="Training Error")
    validation, = plt.plot(range(n), val_err, label="Validation Error")
    plt.legend(handles=[training, validation])
    plt.title("Error Plot")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()

    _, accuracy = clf.test_on_batch(X_test, y_test)
    print ("Accuracy:", accuracy)

    # Reduce dimension to 2D using PCA and plot the results
    y_pred = np.argmax(clf.predict(X_test), axis=1)
    Plot().plot_in_2d(X_test, y_pred, title="Neural Network", accuracy=accuracy, legend_labels=range(y.shape[1]))


if __name__ == "__main__":
    main()