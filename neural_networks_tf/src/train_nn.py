import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist

from network_architecture import build_model

FALLBACK_CSV_PATH = "../data/mnist_train.csv"

OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
IMG_SIZE = 28
NUM_PIXELS = IMG_SIZE * IMG_SIZE


def load_mnist():
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

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, y_train, X_test, y_test


def prepare_data(X_train, y_train, X_test, y_test):
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
