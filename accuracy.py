"""Accuracy reporting and a standalone accuracy tester for saved MNIST models.

Both Network3.py and DropConnect.py train on center-cropped, mean-subtracted
MNIST digits rather than the raw 28x28 images, so this tester inspects each
saved model's expected input size and preprocesses the MNIST test set to
match before evaluating it.

Run directly to evaluate one or more saved .h5/.keras models against the
MNIST test set:

    python accuracy.py models_random_elastic_2/modelo1.h5
    python accuracy.py models_random_elastic_2/*.h5
"""
import argparse
import glob
import sys

import numpy as np


def print_accuracy_report(model, X_test, y_test, Y_test):
    """Evaluate a trained model and print loss, accuracy and misclassified samples."""
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)

    preds = model.predict(X_test, batch_size=None, verbose=0)
    preds = np.argmax(preds, 1)

    mistakes = np.where(preds != y_test)[0]

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}% ({len(X_test) - len(mistakes)}/{len(X_test)})")
    print(f"Misclassified samples: {len(mistakes)}")
    if len(mistakes):
        print(f"Misclassified indices: {mistakes.tolist()}")

    return loss, accuracy


def _center_crop(img, size):
    n = img.shape[0] - size
    x_0 = n // 2
    y_0 = n // 2
    return img[x_0:x_0 + size, y_0:y_0 + size]


def _prepare_test_set(crop_size):
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical

    (_, _), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

    if crop_size != 28:
        X_test = np.array([_center_crop(img, crop_size) for img in X_test])
    X_test = np.array([img - np.mean(img) for img in X_test])

    Y_test = to_categorical(y_test, 10)
    return X_test, y_test, Y_test


def evaluate_saved_model(model_path):
    """Load a saved model from disk, preprocess the MNIST test set to match its
    expected input size, and report its accuracy."""
    from tensorflow.keras.models import load_model

    print(f"\n=== {model_path} ===")
    model = load_model(model_path)

    crop_size = model.input_shape[1]
    X_test, y_test, Y_test = _prepare_test_set(crop_size)

    return print_accuracy_report(model, X_test, y_test, Y_test)


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved MNIST-NET10 models on the MNIST test set.")
    parser.add_argument('models', nargs='+', help="Path(s) or glob pattern(s) to saved .h5/.keras model files")
    args = parser.parse_args()

    model_paths = []
    for pattern in args.models:
        matches = sorted(glob.glob(pattern))
        model_paths.extend(matches if matches else [pattern])

    if not model_paths:
        print("No model files found.")
        sys.exit(1)

    for model_path in model_paths:
        evaluate_saved_model(model_path)


if __name__ == '__main__':
    main()
