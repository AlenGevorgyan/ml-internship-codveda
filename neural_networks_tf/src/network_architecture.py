from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_dim=784, num_classes=10, learning_rate=0.001):
    model = keras.Sequential(name="mnist_feedforward_net")
    model.add(keras.Input(shape=(input_dim,), name="input_layer"))

    model.add(layers.Dense(256, activation="relu", name="hidden_layer_1"))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(128, activation="relu", name="hidden_layer_2"))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64, activation="relu", name="hidden_layer_3"))

    model.add(layers.Dense(num_classes, activation="softmax", name="output_layer"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
