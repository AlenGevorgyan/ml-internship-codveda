"""
train_nn.py
-----------
Task 2: MNIST digit classification with a feed-forward net.

Loads MNIST (via keras.datasets, falls back to a local Kaggle-style
CSV if there's no internet on the machine this runs on), normalizes
pixel values, flattens each 28x28 image into a 784-length vector,
trains the model from network_architecture.py, and plots loss/accuracy
curves.

Run from inside neural_networks_tf/src/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist

from network_architecture import build_model

# fallback path if keras can't pull the dataset (no internet / firewall
# on the machine) -- point this at Kaggle's digit-recognizer train.csv
# (columns: label, pixel0, pixel1, ..., pixel783) if you need it
FALLBACK_CSV_PATH = "../data/mnist_train.csv"

OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
IMG_SIZE = 28
NUM_PIXELS = IMG_SIZE * IMG_SIZE


def load_mnist():
    """
    Tries the standard keras.datasets.mnist loader first (downloads +
    caches automatically, comes pre-split into train/test). Falls back
    to a local CSV export if that fails for whatever reason.
    """
    try:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        print("loaded MNIST via keras.datasets "
              f"(train: {X_train.shape}, test: {X_test.shape})")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"[train_nn] keras.datasets.mnist failed ({e}), "
              f"falling back to local CSV at {FALLBACK_CSV_PATH}")
        return _load_mnist_from_csv(FALLBACK_CSV_PATH)


def _load_mnist_from_csv(path):
    df = pd.read_csv(path)
    y = df["label"].values
    X = df.drop(columns=["label"]).values.reshape(-1, IMG_SIZE, IMG_SIZE)

    # csv fallback usually only gives one file, so carve out a test
    # split manually to match the shape the rest of the script expects
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, y_train, X_test, y_test


def prepare_data(X_train, y_train, X_test, y_test):
    # flatten 28x28 -> 784, and scale pixel values from 0-255 down to
    # 0-1 (plain feed-forward net trains way more reliably on
    # normalized inputs than raw pixel intensities)
    X_train = X_train.reshape(-1, NUM_PIXELS).astype("float32") / 255.0
    X_test = X_test.reshape(-1, NUM_PIXELS).astype("float32") / 255.0

    y_train = y_train.astype("int64")
    y_test = y_test.astype("int64")

    return X_train, X_test, y_train, y_test


def plot_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train acc")
    axes[1].plot(history.history["val_accuracy"], label="val acc")
    axes[1].set_title("Accuracy over epochs")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "loss_accuracy_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("saved training curves ->", path)


def main():
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_mnist()
    X_train, X_test, y_train, y_test = prepare_data(
        X_train_raw, y_train_raw, X_test_raw, y_test_raw
    )

    model = build_model(input_dim=X_train.shape[1], num_classes=10)
    model.summary()

    # held out 10% of the training data as validation, early stopping
    # so it doesn't just keep training past the point it stopped
    # actually improving (MNIST is big enough that epochs get slow,
    # no reason to run all of them if val loss plateaus early)
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.10,
        epochs=30,
        batch_size=128,
        callbacks=[early_stop],
        verbose=1,
    )

    plot_curves(history)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\ntest loss: {test_loss:.4f}  |  test accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nclassification report:")
    print(classification_report(y_test, y_pred))
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, history


if __name__ == "__main__":
    main()
