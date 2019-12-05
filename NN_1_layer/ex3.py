import numpy as np


def relu(x):
    """
    Returns the value of ReLU function
    :param x: a vector
    :return: max between x and 0
    """
    return np.maximum(x, 0)


def d_relu(x):
    """
    Returns the derivative of ReLU.
    For each element: if > 0: x = 1, o.w.: x = 0
    :param x: input to the function
    :return: derivative of ReLU over x
    """
    x[x > 0] = 1
    x[x < 0] = 0
    return x


def softmax(w, x, b):
    """
    Returns a vector after applying the softmax function
    :param w: weights matrix
    :param x: output layer vector
    :param b: bias vector
    :return: a vector of probabilities for each classification
    """
    z = np.dot(w, x) + b
    s = np.exp(z - np.max(z)) / (np.exp(z - np.max(z))).sum()
    return s


def one_hot_encoder(y):
    """
    Create a one hot encoder format value.
    :param y:
    :return: a one hot encoder format value
    """

    letter = np.zeros((10, 1))
    letter[int(y)] = 1
    return letter


def initialize_weights(rows, cols):
    """
    Initialize a weights matrix of size (rows X cols)
    :param rows: rows number
    :param cols: columns number
    :return: weights matrix
    """
    w = np.random.uniform(low=-0.08, high=0.08, size=(rows, cols))
    return w


def loss(h):
    """
    Returns the loss rate of a specific example
    :param h: Vector loss in index of y
    :return: -log(h)
    """
    if (h == 0 or h == 0.0):
        print("zero in loss!")
    return -np.log(h)


def forward(x, params):
    """

    :param x: the x set
    :param params: weights and bias
    :return: updated params with vector of probabilities
    """
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    # to first layer
    z1 = np.dot(w1, x) + b1
    h1 = relu(z1)
    # to output layer
    z2 = np.dot(w2, h1) + b2
    h2 = softmax(w2, h1, b2)

    # build return dictionary
    ret = {'x': x, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
    return ret


def back_prop(params, ret, x, y):
    y_vec = one_hot_encoder(y)
    h2 = ret['h2']
    h1 = ret['h1']
    z2 = ret['z2']
    z1 = ret['z1']
    w2 = params['w2']
    w1 = params['w1']
    dL_z2 = (h2 - y_vec)
    dL_w2 = np.outer(dL_z2, h1.T)
    dL_b2 = dL_z2
    dL_z1 = np.dot(w2.T, dL_z2) * d_relu(z1)
    dL_w1 = np.dot(dL_z1, x.T)
    dL_b1 = dL_z1
    return {'dL_w2': dL_w2, 'dL_b2': dL_b2, 'dL_w1': dL_w1, 'dL_b1': dL_b1}


def update_weights(params, gradients, lr):
    w2_new = params['w2'] - lr * gradients['dL_w2']
    b2_new = params['b2'] - lr * gradients['dL_b2']
    w1_new = params['w1'] - lr * gradients['dL_w1']
    b1_new = params['b1'] - lr * gradients['dL_b1']
    params['w2'] = w2_new
    params['b2'] = b2_new
    params['w1'] = w1_new
    params['b1'] = b1_new


def predict_on_validation(params, val_x, val_y):
    """
    The function goes over the validation set and predicts classifcations.
    For each epoch we calculate the avg loss and accuracy
    :param params: weights and bias
    :param val_x: validation set x
    :param val_y: validation set y
    :return: average loss and accuracy
    """
    sum_loss = 0
    accuracy = 0

    # go over the validation set
    for x, y in zip(val_x, val_y):
        x = x.reshape(x.shape[0], 1)
        ret = forward(x, params)
        # ret['h2'] is the predication probabilities vector from the output layer
        l = loss(ret['h2'][int(y)])
        sum_loss += l
        if y == ret['h2'].argmax():
            accuracy += 1

    # average the results
    avg_accuracy = accuracy / len(val_x)
    avg_loss = sum_loss / len(val_x)
    return avg_accuracy, avg_loss


def shuffle(train_x, train_y):
    i = np.arange(len(train_x))
    np.random.shuffle(i)
    return train_x[i], train_y[i]


def train_network(train_x, train_y, params):  # , val_x, val_y):
    """
    Train the network on the train set.
    :param train_x: train set x - set of features.
    :param train_y: train set y - set of classifications.
    :param params: weights and bias.
    :return: updated weights and bias
    """
    global lr
    global epochs

    # print("epoch, train loss, validation loss, validation accuracy")
    for e in range(epochs):
        sum_loss = 0

        train_x, train_y = shuffle(train_x, train_y)

        for x, y in zip(train_x, train_y):
            x = x.reshape(x.shape[0], 1)
            ret = forward(x, params)
            l = loss(ret['h2'][int(y)])
            sum_loss += l

            gradients = back_prop(params, ret, x, y)

            update_weights(params, gradients, lr)

        # validation_accuracy, validation_loss = predict_on_validation(params, val_x, val_y)
        # print(e + 1, sum_loss / len(train_x), validation_loss, "{}%".format(validation_accuracy * 100))
    return params


def predict_on_test(params):
    """
    Apply the prediction on hte test set.
    Writes into 'test_y' file.
    :param params: weights and bias.
    :return: none
    """
    with open('test_x', 'r') as test_x_file:
        test_x = np.loadtxt(test_x_file)
        # normalize the data
        test_x /= 255
        with open('test_y', 'w') as test_y_file:
            for x in test_x:
                x = x.reshape(x.shape[0], 1)
                ret = forward(x, params)
                y_hat = ret['h2'].argmax()
                test_y_file.write(str(y_hat)+" \n")


if __name__ == '__main__':

    first_hidden_layer_size = 397
    lr = 0.01
    epochs = 10
    #print("hidden layer size: %d, learning rate: %f" % (first_hidden_layer_size, lr))
    with open('train_x') as train_x_file:
        train_x = np.loadtxt(train_x_file)
        # normalize the data
        train_x /= 255
    with open('train_y') as train_y_file:
        train_y = np.loadtxt(train_y_file)

    # shuffle the data
    train_x, train_y = shuffle(train_x, train_y)

    # assign 20% for validation set and 80% for train set
    # validation_set_size = int(len(train_x) * 0.2)
    # validation_set_x, validation_set_y = train_x[-validation_set_size:], train_y[-validation_set_size:]
    # train_x, train_y = train_x[: -validation_set_size], train_y[: -validation_set_size]

    # initialize the weights
    w1 = initialize_weights(first_hidden_layer_size, 28 * 28)
    b1 = np.random.uniform(-0.08, 0.08, (first_hidden_layer_size, 1))
    w2 = initialize_weights(10, first_hidden_layer_size)
    b2 = np.random.uniform(-0.08, 0.08, (10, 1))
    params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    train_network(train_x, train_y, params) #, validation_set_x, validation_set_y)

    predict_on_test(params)
