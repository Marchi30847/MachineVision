import os
import ssl
from dataclasses import dataclass
from pathlib import Path

import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


def create_certifi_ssl_context(
        protocol: int = ssl.PROTOCOL_TLS_CLIENT,
        *,
        cafile: str | None = None,
        capath: str | None = None,
        cadata: str | bytes | None = None,
) -> ssl.SSLContext:
    """Create SSL context using certifi certificates by default."""
    del cafile, capath, cadata

    return ssl.create_default_context(
        purpose=ssl.Purpose.SERVER_AUTH,
        cafile=certifi.where(),
    )


ssl._create_default_https_context = create_certifi_ssl_context

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import NDArray
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

type FloatArray = NDArray[np.float32]
type IntArray = NDArray[np.int64]


@dataclass(slots=True, kw_only=True)
class TrainingConfig:
    validation_size: float = 0.15
    random_state: int = 14

    batch_size: int = 64
    epochs: int = 60

    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    model_path: Path = Path("cifar10_cnn.keras")
    report_path: Path = Path("report.pdf")


CLASS_NAMES: list[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_cifar10_data(
        config: TrainingConfig,
) -> tuple[FloatArray, FloatArray, IntArray, IntArray, FloatArray, IntArray]:
    """Load CIFAR-10 and prepare train, validation and test sets."""
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train_full = x_train_full.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    y_train_full = y_train_full.reshape(-1).astype(np.int64)
    y_test = y_test.reshape(-1).astype(np.int64)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=y_train_full,
    )

    return x_train, x_val, y_train, y_val, x_test, y_test


def build_model(config: TrainingConfig) -> keras.Model:
    """Build a CNN model for CIFAR-10 classification."""
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.08, 0.08),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=(32, 32, 3), name="input_image")

    x = data_augmentation(inputs)

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(10, activation="softmax", name="class_probabilities")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cifar10_cnn",
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(
        model: keras.Model,
        x_train: FloatArray,
        y_train: IntArray,
        x_val: FloatArray,
        y_val: IntArray,
        config: TrainingConfig,
) -> keras.callbacks.History:
    """Train the model and return the training history."""
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        ),
    ]

    return model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=config.batch_size,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )


def plot_training_history(history: keras.callbacks.History) -> plt.Figure:
    """Create a training history figure."""
    history_data: dict[str, list[float]] = history.history

    epochs = range(1, len(history_data["loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, history_data["loss"], label="train loss")
    ax.plot(epochs, history_data["val_loss"], label="validation loss")
    ax.plot(epochs, history_data["accuracy"], label="train accuracy")
    ax.plot(epochs, history_data["val_accuracy"], label="validation accuracy")

    ax.set_title("Training history")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    return fig


def plot_confusion_matrix(
        y_true: IntArray,
        y_pred: IntArray,
) -> plt.Figure:
    """Create a confusion matrix figure."""
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(9, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        values_format="d",
    )

    ax.set_title("Confusion matrix")

    fig.tight_layout()

    return fig


def get_model_summary_text(model: keras.Model) -> str:
    """Return model.summary() as a string."""
    lines: list[str] = []
    model.summary(print_fn=lines.append)

    return "\n".join(lines)


def create_summary_page(
        model_summary: str,
        test_accuracy: float,
) -> plt.Figure:
    """Create a text page with model architecture and final accuracy."""
    fig = plt.figure(figsize=(8.27, 11.69))

    text = (
        "Project 4: Convolutional Neural Network for CIFAR-10\n\n"
        f"Final test accuracy: {test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%)\n\n"
        "Model architecture summary:\n\n"
        f"{model_summary}"
    )

    fig.text(
        0.05,
        0.95,
        text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=7,
    )

    return fig


def save_pdf_report(
        report_path: Path,
        summary_figure: plt.Figure,
        history_figure: plt.Figure,
        confusion_matrix_figure: plt.Figure,
) -> None:
    """Save all report figures into one PDF file."""
    with PdfPages(report_path) as pdf:
        pdf.savefig(summary_figure)
        pdf.savefig(history_figure)
        pdf.savefig(confusion_matrix_figure)

    plt.close(summary_figure)
    plt.close(history_figure)
    plt.close(confusion_matrix_figure)


def main() -> None:
    config = TrainingConfig()

    tf.keras.utils.set_random_seed(config.random_state)

    x_train, x_val, y_train, y_val, x_test, y_test = load_cifar10_data(config)

    model = build_model(config)

    history = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        config=config,
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    probabilities = model.predict(
        x_test,
        batch_size=config.batch_size,
        verbose=1,
    )

    y_pred = np.argmax(probabilities, axis=1).astype(np.int64)

    model.save(config.model_path)

    model_summary = get_model_summary_text(model)

    summary_figure = create_summary_page(
        model_summary=model_summary,
        test_accuracy=float(test_accuracy),
    )

    history_figure = plot_training_history(history)

    confusion_matrix_figure = plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
    )

    save_pdf_report(
        report_path=config.report_path,
        summary_figure=summary_figure,
        history_figure=history_figure,
        confusion_matrix_figure=confusion_matrix_figure,
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Saved model to: {config.model_path}")
    print(f"Saved report to: {config.report_path}")


if __name__ == "__main__":
    main()
