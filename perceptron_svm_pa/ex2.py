import io
from random import shuffle
from random import seed

import numpy as np
import sys

gender_to_int = dict()
gender_set = set()


def create_weights(X_train, Y_train):
    """
    Create the weights matrix to use in the perceptron algorithm.
    The number of rows is the amount of classifications.
    The number of columns is the amount of features.
    :param X_train: the content of the x train set.
    :return: a matrix (rows * columns) of zeros.
    """
    # len(X_train[0]) is the amount of features
    # 3 is the amount of classifications
    class_num = len(set(y for y in Y_train))
    w = np.zeros([class_num, len(X_train[0]) - 1])
    w = np.c_[w, np.ones(class_num)]
    return w


def add_bias(X_train, X_test):
    """
    Adds bias to the data
    :param X_train: train set
    :param X_test: test set
    :return: train set and test set with bias
    """
    X_train = np.c_[X_train, np.ones(len(X_train))]
    X_test = np.c_[X_test, np.ones(len(X_test))]
    return X_train, X_test


def one_hot_encoder(gender):
    """
    Create a one hot encoder format value, using the gender to int dictionary.
    :param gender:
    :return: a one hot encoder format value (example: [0, 0, 1])
    """

    letter = [0 for _ in range(len(gender_set))]
    letter[gender_to_int[gender]] = 1
    return letter


def file_parse(file):
    """
    Read the file and convert each number to float.
    :param file: a file content.
    :return: the file content after conversion to float.
    """
    # read the file
    file_content = file.readlines()
    # go over the file content and split the lines
    for row in range(len(file_content)):
        file_content[row] = file_content[row].split(",")
        # cast every float from string to float
        for r in range(len(file_content[row])):
            v = file_content[row][r]
            if isfloat(v):
                file_content[row][r] = float(v)
    return file_content


def isfloat(value):
    """
    Check if a string is float convertible.
    :param value: a string.
    :return: True is convertible, o.w. False.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def set_dicts(gender_set):
    """
    Set the dictionaries for the one hot encode.
    Map each nominal value to an int (index).
    Both from value to int and from int to value.
    :param gender_set: The first column in the train file as nominal  values.
    :return: None.
    """
    for i, g in enumerate(gender_set):
        gender_to_int.update({g: i})


def classify(w, X_test):
    """
    Test the trained model on a test set, and compare it to the results of the test set.
    :param w: The weights matrix after train.
    :param X_test: features of test set.
    :return: A list of classifications according to the trained model, and the test set.
    """
    results = list()
    for x in X_test:
        # calculate y_hat - predict
        y_hat = np.argmax(np.dot(w, x))
        results.append(int(y_hat))

    return results


def z_score_norm_data(X_train, X_test):
    """
    Normalize the data using z-score.
    :param X_set: The data set.
    :return: The data set normalized.
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    for i in range(len(X_train[0])):
        mean = np.mean(X_train[:, i], axis=0)
        std = np.std(X_train[:, i], axis=0)
        if std == 0 or std == 0.0:
            continue
        X_train[:, i] = (np.asarray(X_train)[:, i] - mean) / std
        X_test[:, i] = (np.asarray(X_test)[:, i] - mean) / std
    return X_train, X_test


def min_max_norm_data(X_train, X_test):
    """
    Normalize the data using min - max.
    :param X_set: The data set.
    :return: The data set normalized.
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    rows, cols = X_train.shape
    for i in range(cols - 1):
        min_x = np.min(X_train[:, i])
        max_x = np.max(X_train[:, i])
        new_min = 0
        new_max = 1
        for j in range(rows - 1):
            if (max_x - min_x) == 0 or (max_x - min_x) == 0.0:
                continue
            X_train[j][i] = ((X_train[j][i] - min_x) / (max_x - min_x)) * (new_max - new_min) + new_min
            if j < len(X_test):
                X_test[j][i] = ((X_test[j][i] - min_x) / (max_x - min_x)) * (new_max - new_min) + new_min
    return X_train, X_test


def svm_predict(w, X_train, Y_train):
    """
    Trains the weights matrix according to SVM algorithm.
    :param w: The weights matrix.
    :param X_train: The features array.
    :param Y_train: The classifications array.
    :return: The weights matrix after train.
    """
    epochs = 25
    eta = 0.41
    lamda = 0.46

    for e in range(epochs):
        zip_files = list(zip(X_train, Y_train))
        seed(1)
        shuffle(zip_files)
        X_train, Y_train = zip(*zip_files)
        for x, y in zip(X_train, Y_train):
            y = int(y)
            y_hat = np.argmax(np.dot(w[y], x))
            if y_hat != y:
                # update w[y] and w[y_hat]
                y_hat = int(y_hat)
                w[y, :] = (1 - (eta * lamda)) * w[y] + (eta * np.asarray(x))
                w[y_hat, :] = (1 - (eta * lamda)) * w[y_hat] - (eta * np.asarray(x))
                # update all other weights
                for i in range(len(w)):
                    if (int(i) != y) and (int(i) != y_hat):
                        w[i, :] = (1 - (eta * lamda)) * w[i]
            else:
                # update all other weights
                for i in range(len(w)):
                    if (int(i) != y) and (int(i) != y_hat):
                        w[i, :] = (1 - (eta * lamda)) * w[i]

    return w


def pa_loss(w, x, y, y_hat):
    """
    The loss of PA is L(w, x, y) = max{0, 1- w[y_t] * x + w[y_hat_t] * x}
    :param w: The weights matrix in the current iteration.
    :param x: The example.
    :param y: The wanted classification.
    :param y_hat: The classification of the current iteration.
    :return: loss = max{0, 1- w[y_t] * x + w[y_hat_t] * x}
    """
    return max((0, 1 - np.dot(w[y], x) + np.dot(w[y_hat], x)))


def pa_predict(w, X_train, Y_train):
    """
    Trains the weights matrix according to PA algorithm.
    :param w: The weights matrix.
    :param X_train: The features array.
    :param Y_train: The classifications array.
    :return: The weights matrix after train.
    """
    epochs = 39

    for e in range(epochs):
        zip_files = list(zip(X_train, Y_train))
        seed(2)
        shuffle(zip_files)
        X_train, Y_train = zip(*zip_files)
        for x, y in zip(X_train, Y_train):
            y = int(y)
            # calculate y_hat - predict
            y_hat = np.argmax(np.dot(w, x))
            # calculate tau
            denominator = (2 * (np.linalg.norm(x) ** 2))
            if denominator == 0 or denominator == 0.0:
                tau = 0
            else:
                tau = (pa_loss(w, x, y, y_hat) / denominator)
            # update weights
            if y_hat != y:
                w[y, :] = w[y, :] + tau * np.asarray(x)
                w[y_hat, :] = w[y_hat, :] - tau * np.asarray(x)
    return w


def perceptron_predict(w, X_train, Y_train):
    """
    The perceptron algorithm, train the weights matrix according to the train set.
    :param w: the weights matrix.
    :param X_train: features train set.
    :param Y_train: classification set.
    :return: the trained weight matrix.
    """
    epochs = 5
    eta = 0.01

    for e in range(epochs):
        zip_files = list(zip(X_train, Y_train))
        seed(1)
        shuffle(zip_files)
        X_train, Y_train = zip(*zip_files)
        for x, y in zip(X_train, Y_train):
            y = int(y)
            # calculate y_hat - predict
            y_hat = np.argmax(np.dot(w, x))
            # update weights
            if y_hat != y:
                w[y, :] = w[y, :] + eta * np.asarray(x)
                w[y_hat, :] = w[y_hat, :] - eta * np.asarray(x)
    return w


def run(train_x_file, train_y_file, test_x_file):
    """
    Creates the perceptron model, trains the model on the train set and predicts
    classifications of the test set.
    :return: The classifications of the test set.
    """
    # read the train file for svm
    with open(train_x_file) as X_train_file:
        X_train_svm = file_parse(X_train_file)
        # set the dictionaries for one hot encode
        gender_set.update(x[0][0] for x in X_train_svm)
        set_dicts(gender_set)
        for i in range(len(X_train_svm)):
            new_row = one_hot_encoder(X_train_svm[i][0])
            new_row += (X_train_svm[i][1:])
            X_train_svm[i] = new_row

    # read the train file for other algorithms
    with open(train_x_file) as X_train_file:
        f = X_train_file.read()
        f = f.replace("F", '-1')
        f = f.replace("I", '0')
        f = f.replace('M', '1')
        # read the x train file
        f = io.StringIO(f)
        X_train = np.loadtxt(f, delimiter=',')

    # read classifications train file
    with open(train_y_file) as Y_train_file:
        # read the classification file
        Y_train = np.loadtxt(Y_train_file, delimiter='\n')

    # read test file for svm
    with open(test_x_file) as X_test_file:
        X_test_svm = file_parse(X_test_file)
        for i in range(len(X_test_svm)):
            new_row = one_hot_encoder(X_test_svm[i][0])
            new_row += (X_test_svm[i][1:])
            X_test_svm[i] = new_row

    # read test file for other algorithms
    with open(test_x_file) as X_test_file:
        f = X_test_file.read()
        f = f.replace("F", '-1')
        f = f.replace("I", '0')
        f = f.replace('M', '1')
        # read the x train file
        f = io.StringIO(f)
        X_test = np.loadtxt(f, delimiter=',')

    # add bias to perceptron, svm and pa
    X_train_perc, X_test_perc = add_bias(X_train, X_test)
    X_train_svm, X_test_svm = add_bias(X_train_svm, X_test_svm)
    X_train_pa, X_test_pa = add_bias(X_train, X_test)

    # normalize the data
    X_train_svm, X_test_svm = z_score_norm_data(X_train_svm, X_test_svm)
    X_train_pa, X_test_pa = z_score_norm_data(X_train_pa, X_test_pa)

    # create the weights
    w1 = create_weights(X_train_perc, Y_train)
    w2 = create_weights(X_train_svm, Y_train)
    w3 = create_weights(X_train_pa, Y_train)

    # predict for each algorithm
    w1 = perceptron_predict(w1, X_train_perc, Y_train)
    perc_list = classify(w1, X_test_perc)

    w2 = svm_predict(w2, X_train_svm, Y_train)
    svm_list = classify(w2, X_test_svm)

    w3 = pa_predict(w3, X_train_pa, Y_train)
    pa_list = classify(w3, X_test_pa)

    return perc_list, svm_list, pa_list


def main():
    train_x, train_y, test_x = sys.argv[1:]
    perc_list, svm_list, pa_list = run(train_x, train_y, test_x)

    for i in range(len(perc_list)):
        print("perceptron: %d, svm: %d, pa: %d" % (perc_list[i], svm_list[i], pa_list[i]))


main()
