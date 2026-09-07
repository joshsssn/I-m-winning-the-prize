"""GPU-optimised version of other_model.py (the DropConnect-style CNN on 24x24 crops).

Same pipeline and same model as other_model.py: random 24x24 crop, per-image mean
subtraction, random rotation/shear/shift, elastic deformation (4 variants + original),
then SGD training with Nesterov momentum and the original inverse-time learning-rate decay.

Differences are purely about speed:
  * elastic deformation runs on the GPU in batches (shared with model_gpu.py, verified
    against the SciPy version to ~1e-6);
  * crop / mean subtraction are vectorised (identical values, identical random draws);
  * training steps are batched (steps_per_execution) and XLA-compiled, both in float32;
  * GPU memory is allocated on demand so the script can coexist with other jobs.
"""
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import mnist

from image_utils import (
    random_rotation, random_shear, random_shift,
    transform_matrix_offset_center, apply_transform,
)
from accuracy import print_accuracy_report
from timing import EpochTimer, RunEstimator, timed_stage
# GPU elastic deformation (batched Gaussian blur + bilinear sampling); importing
# model_gpu also enables on-demand GPU memory growth.
from model_gpu import getElastics, elastic_transform, STEPS_PER_EXECUTION, USE_XLA

# Smoke-test mode: set to False to run the full experiment as originally intended.
SMOKE_TEST = False
SMOKE_TRAIN_SIZE = 200
SMOKE_TEST_SIZE = 100
SMOKE_EPOCHS = 1

# Models are saved in their own directory (model_gpu.py uses ./models_random_elastic_2)
# so both scripts can run and their checkpoints compared side by side:
#   python accuracy.py models_random_elastic_2/*.h5 models_other_random_elastic_2/*.h5
MODEL_DIR = './models_other_random_elastic_2'


def augment_data(dataset, dataset_labels, augmentation_factor=1, use_random_rotation=True,
                  use_random_shear=True, use_random_shift=True, use_random_zoom=True):
    """Apply random rotations, shears and shifts to each image in the dataset."""
    augmented_image = []
    augmented_image_labels = []

    for num in range(0, dataset.shape[0]):
        augmented_image.append(dataset[num])
        augmented_image_labels.append(dataset_labels[num])

        for i in range(0, augmentation_factor):

            if use_random_rotation:
                augmented_image.append(random_rotation(dataset[num], 20, row_axis=0, col_axis=1, channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])

            if use_random_shear:
                augmented_image.append(random_shear(dataset[num], 0.2, row_axis=0, col_axis=1, channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])

            if use_random_shift:
                augmented_image.append(random_shift(dataset[num], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])

    s = np.arange(len(augmented_image))
    return np.array(augmented_image)[s], np.array(augmented_image_labels)[s]


def crop(img, size, isRand=True):
    """Crop a window of the given size, at a random position unless isRand is False."""
    n = len(img) - size
    x_0 = 2
    y_0 = 2
    if isRand:
        x_0 = random.randint(0, n)
        y_0 = random.randint(0, n)

    return img[x_0:x_0 + size, y_0:y_0 + size]


def crop_batch(images, size, isRand=True):
    """Vectorised equivalent of `[crop(img, size, isRand) for img in images]`.

    Draws the random offsets with the same `random.randint` calls in the same order as
    the per-image loop, so the crops are identical for a given random seed.
    """
    count, height = images.shape[0], images.shape[1]
    n = height - size
    if isRand:
        offsets = np.array([(random.randint(0, n), random.randint(0, n)) for _ in range(count)],
                           dtype=np.intp).reshape(count, 2)
        x_0, y_0 = offsets[:, 0], offsets[:, 1]
    else:
        x_0 = np.full(count, 2, dtype=np.intp)
        y_0 = np.full(count, 2, dtype=np.intp)

    idx = np.arange(size)
    rows = (x_0[:, None] + idx)[:, :, None]      # (N, size, 1)
    cols = (y_0[:, None] + idx)[:, None, :]      # (N, 1, size)
    batch = np.arange(count)[:, None, None]
    return np.ascontiguousarray(images[batch, rows, cols])


def rotation_2(x, theta, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.):
    """Rotate an image by theta radians, based on Keras' former random_rotation."""
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def rotations(dataset, dataset_labels, angles):
    """Add a rotated copy of every image in the dataset for each angle given."""
    augmented_image = []
    augmented_image_labels = []

    for num in range(0, dataset.shape[0]):
        augmented_image.append(dataset[num])
        augmented_image_labels.append(dataset_labels[num])

        for theta in angles:
            augmented_image.append(rotation_2(dataset[num], theta, row_axis=0, col_axis=1, channel_axis=2))
            augmented_image_labels.append(dataset_labels[num])

    return np.array(augmented_image), np.array(augmented_image_labels)


def augmentate(images, labels, alpha, sigma):
    """Elastic-deform the dataset, then add rotated copies at fixed angles."""
    images = images.reshape(images.shape[0], 1, 24, 24)
    deformated, new_labels = getElastics(images, labels, alpha, sigma)
    deformated = deformated.reshape(deformated.shape[0], 24, 24, 1)
    augmented = rotations(deformated, new_labels, [-16, -8, 8, 16])
    return augmented[0].reshape(augmented[0].shape[0], 24, 24, 1), augmented[1]


def augmentate_2(images, labels, alpha, sigma):
    """Elastic-deform the dataset without adding rotated copies."""
    images = images.reshape(images.shape[0], 1, 24, 24)
    deformated, new_labels = getElastics(images, labels, alpha, sigma)
    deformated = deformated.reshape(deformated.shape[0], 24, 24, 1)
    return deformated, new_labels


def experiment(X_train, Y_train, X_test, Y_test, y_test, file, epochs=50, model_label=""):
    """Build, train and evaluate the DropConnect-style CNN, then save it to `file`."""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # `decay` was removed from the Keras optimizer; InverseTimeDecay with decay_steps=1
    # reproduces the old schedule exactly: lr0 / (1 + decay * step).
    lr = optimizers.schedules.InverseTimeDecay(0.01, decay_steps=1, decay_rate=1e-6)
    sgd = optimizers.SGD(learning_rate=lr, momentum=0.95, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'],
                  steps_per_execution=STEPS_PER_EXECUTION, jit_compile=USE_XLA)

    os.makedirs(os.path.dirname(file) or '.', exist_ok=True)
    checkpoint = ModelCheckpoint(file, monitor='val_accuracy', mode='max',
                                  save_best_only=True, verbose=1)

    print("Training...")
    model.fit(X_train, Y_train, batch_size=128, epochs=epochs, verbose=1,
              validation_data=(X_test, Y_test),
              callbacks=[EpochTimer(epochs, model_label), checkpoint])

    print("Evaluating best checkpoint...")
    best_model = load_model(file)
    return print_accuracy_report(best_model, X_test, y_test, Y_test)


def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if SMOKE_TEST:
        X_train = X_train[:SMOKE_TRAIN_SIZE]
        y_train = y_train[:SMOKE_TRAIN_SIZE]
        X_test = X_test[:SMOKE_TEST_SIZE]
        y_test = y_test[:SMOKE_TEST_SIZE]
    else:
        X_train = X_train[0:60000]
        y_train = y_train[0:60000]

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Vectorised crop (random position for train, fixed for test) + per-image mean
    # subtraction; same values as the per-image loops.
    X_train = crop_batch(X_train, 24)
    X_test = crop_batch(X_test, 24, isRand=False)

    X_train -= X_train.mean(axis=(1, 2, 3), keepdims=True)
    X_test -= X_test.mean(axis=(1, 2, 3), keepdims=True)

    with timed_stage("random augmentation"):
        X_train_random, y_train_random = augment_data(
            X_train, y_train, augmentation_factor=1,
            use_random_rotation=True, use_random_shear=True, use_random_shift=True)

    with timed_stage("elastic deformation"):
        X_train_2, y_train_2 = augmentate_2(X_train_random, y_train_random, 6, 4)
    Y_train_2 = to_categorical(y_train_2, 10)

    Y_test = to_categorical(y_test, 10)

    epochs = SMOKE_EPOCHS if SMOKE_TEST else 50
    n_models = 1 if SMOKE_TEST else 5

    print(f"Training set size after augmentation: {len(X_train_2)} images")
    print("Training: random augmentation + elastic deformations")

    run = RunEstimator(n_models)
    results = []
    for i in range(1, n_models + 1):
        with run.track_model(i):
            loss, acc = experiment(X_train_2, Y_train_2, X_test, Y_test, y_test,
                                   f'{MODEL_DIR}/modelo{i}.h5', epochs=epochs,
                                   model_label=f"model {i}/{n_models}")
        results.append((i, loss, acc))

    print(f"\n=== Summary: best-val_accuracy checkpoints in {MODEL_DIR} ===")
    for i, loss, acc in results:
        print(f"modelo{i}.h5: test accuracy {acc * 100:.2f}%  loss {loss:.4f}")


if __name__ == '__main__':
    main()
