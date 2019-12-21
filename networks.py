#!/usr/bin/python3

"""networks.py Contains all networks components for simple split learning

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def client_network(inputs):
    """Constructor of the partial network trained at the client(s)
       up to the split layer

    Args:
        inputs: Input tensor

    Returns:
        outputs: Output tensor
    """
    # First 3x3 convolutional layer
    x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu",
            input_shape=(28, 28, 1), name="client_conv_1")(inputs)

    # Max pooling layer
    outputs = MaxPooling2D(pool_size=(2, 2), name="client_mp_1")(x)

    return outputs


def server_network(inputs):
    """Constructor of the partial network trained at the server
       from the split layer up to the loss function

    Args:
        inputs: Input tensor

    Returns:
        ???
    """
    # Second 3x3 convolutional layer
    x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu",
        input_shape=(13, 13, 32), name="server_conv_2")(inputs)

    # Second max pooling layer
    x = MaxPooling2D(pool_size=(2, 2), name="server_mp_2")(x)

    # Third 3x3 convolutional layer
    x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu",
        name="server_conv_3")(x)

    # Flatten the layer
    x = Flatten()(x)

    # First fully-connected layer
    x = Dense(64, activation="relu", name="server_fc_1")(x)

    # Second fully-connected layer with softmax output
    outputs = Dense(10, activation="softmax", name="server_fc_2")(x)

    return outputs


def complete_network(inputs):
    x = client_network(inputs)
    outputs = server_network(x)

    return outputs

