"""
network_architecture.py
------------------------
Task 2: Deep Learning Feed-Forward Neural Network

Switched this over to MNIST digit classification per the actual task
PDF (the iris version was a placeholder before I had confirmation on
the dataset). Still a plain feed-forward net, not a CNN -- images get
flattened to a 784-length vector before hitting the input layer, no
conv/pooling layers here since the task specifically calls out
Input/Hidden/Output dense layers.

Architecture:
    Input(784) -> Dense(256, relu) -> Dropout -> Dense(128, relu)
                -> Dropout -> Dense(64, relu) -> Dense(10, softmax)

Went a bit bigger than the iris version obviously, 784 raw pixel
inputs need more capacity than 4 tabular features did. Added a second
dropout layer too since a plain feed-forward net on MNIST overfits
pretty fast if you don't regularize it at all.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_dim=784, num_classes=10, learning_rate=0.001):
    model = keras.Sequential(name="mnist_feedforward_net")

    # input layer -- expects already-flattened 28x28 images (784,)
    model.add(keras.Input(shape=(input_dim,), name="input_layer"))

    # hidden layer 1
    model.add(layers.Dense(256, activation="relu", name="hidden_layer_1"))
    model.add(layers.Dropout(0.3))

    # hidden layer 2
    model.add(layers.Dense(128, activation="relu", name="hidden_layer_2"))
    model.add(layers.Dropout(0.3))

    # hidden layer 3 -- narrowing it down before the output layer
    model.add(layers.Dense(64, activation="relu", name="hidden_layer_3"))

    # output layer -- 10 digit classes, softmax
    model.add(layers.Dense(num_classes, activation="softmax", name="output_layer"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    # quick sanity check that the model builds + shapes line up
    m = build_model()
    m.summary()
