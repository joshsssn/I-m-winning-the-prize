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

# --- GPU setup ---------------------------------------------------------------
# Allocate GPU memory on demand instead of reserving the whole card up front, so the
# script can share the GPU with other processes (it makes no difference to speed).
for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass  # already initialised

# Number of batches Keras runs inside a single tf.function call. The model is tiny, so
# at batch size 128 training is bound by per-step Python/launch overhead rather than by
# GPU compute; batching steps together removes most of it. Batch size, optimizer, data
# order and all per-step maths are unchanged, so training dynamics are identical.
STEPS_PER_EXECUTION = 200

# Compile the train/test step with XLA: fuses the small conv/dense/activation kernels
# into a few large ones. Same float32 arithmetic, no precision reduction.
USE_XLA = True

# Smoke-test mode: set to False to run the full experiment as originally intended.
SMOKE_TEST = False
SMOKE_TRAIN_SIZE = 200
SMOKE_TEST_SIZE = 100
SMOKE_EPOCHS = 1


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


def crop(img, size):
    """Crop a fixed-size window starting at (4, 4) from an image (H, W, C) or a batch (N, H, W, C)."""
    x_0 = 4
    y_0 = 4
    return img[..., x_0:x_0 + size, y_0:y_0 + size, :]


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


# --- GPU elastic deformation -------------------------------------------------
# Drop-in replacement for the SciPy version: the random displacement fields are
# built, Gaussian-smoothed and bilinearly sampled on the GPU, in batches.
# Numerically equivalent to gaussian_filter(mode='constant') + map_coordinates(order=1).

ELASTIC_CHUNK = 8192  # images processed per GPU batch (each yields 4 deformations)


def _gaussian_kernel1d(sigma, truncate=4.0):
    """1-D Gaussian kernel identical to the one scipy.ndimage builds."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k.astype(np.float32)


def _gaussian_blur(field, kernel):
    """Separable Gaussian blur on (B, H, W, 1). 'SAME' zero-padding == mode='constant', cval=0."""
    ky = tf.reshape(kernel, [-1, 1, 1, 1])
    kx = tf.reshape(kernel, [1, -1, 1, 1])
    field = tf.nn.conv2d(field, ky, strides=1, padding='SAME')
    field = tf.nn.conv2d(field, kx, strides=1, padding='SAME')
    return field


def _bilinear_sample(images, row, col):
    """Bilinear sampling matching map_coordinates(order=1, mode='constant', cval=0).

    Coordinates outside [0, n-1] yield 0, exactly as scipy's legacy 'constant' mode does.
    images: (B, H, W); row/col: (B, H, W) float32.
    """
    shape = tf.shape(images)
    b, h, w = shape[0], shape[1], shape[2]
    fh = tf.cast(h - 1, tf.float32)
    fw = tf.cast(w - 1, tf.float32)

    inside = ((row >= 0.0) & (row <= fh) & (col >= 0.0) & (col <= fw))

    r = tf.clip_by_value(row, 0.0, fh)
    c = tf.clip_by_value(col, 0.0, fw)
    r0 = tf.floor(r)
    c0 = tf.floor(c)
    wr = r - r0
    wc = c - c0

    r0i = tf.cast(r0, tf.int32)
    c0i = tf.cast(c0, tf.int32)
    r1i = tf.minimum(r0i + 1, h - 1)
    c1i = tf.minimum(c0i + 1, w - 1)

    flat = tf.reshape(images, [b, h * w])

    def gather(ri, ci):
        return tf.gather(flat, tf.reshape(ri * w + ci, [b, -1]), batch_dims=1)

    v00 = gather(r0i, c0i)
    v01 = gather(r0i, c1i)
    v10 = gather(r1i, c0i)
    v11 = gather(r1i, c1i)

    wr = tf.reshape(wr, [b, -1])
    wc = tf.reshape(wc, [b, -1])
    top = v00 * (1.0 - wc) + v01 * wc
    bottom = v10 * (1.0 - wc) + v11 * wc
    out = top * (1.0 - wr) + bottom * wr

    out = tf.reshape(out, [b, h, w])
    return tf.where(inside, out, tf.zeros_like(out))


def _elastic_from_fields(images, raw_dx, raw_dy, alpha, kernel):
    """Deform `images` (B, H, W) with the given raw uniform fields in [-1, 1].

    Split out from the random generation so the transform can be tested against SciPy
    on identical fields.
    """
    shape = tf.shape(images)
    h, w = shape[1], shape[2]

    dx = _gaussian_blur(raw_dx[..., None], kernel)[..., 0] * alpha
    dy = _gaussian_blur(raw_dy[..., None], kernel)[..., 0] * alpha

    rows = tf.cast(tf.range(h), tf.float32)[None, :, None]
    cols = tf.cast(tf.range(w), tf.float32)[None, None, :]

    return _bilinear_sample(images, rows + dy, cols + dx)


@tf.function(reduce_retracing=True)
def _elastic_batch(images, alpha, kernel, n_variants):
    """Return (B, n_variants + 1, H, W): n_variants deformations then the original."""
    shape = tf.shape(images)
    b, h, w = shape[0], shape[1], shape[2]

    repeated = tf.reshape(tf.tile(images[:, None], [1, n_variants, 1, 1]), [-1, h, w])
    raw_dx = tf.random.uniform(tf.shape(repeated), dtype=tf.float32) * 2.0 - 1.0
    raw_dy = tf.random.uniform(tf.shape(repeated), dtype=tf.float32) * 2.0 - 1.0

    deformed = _elastic_from_fields(repeated, raw_dx, raw_dy, alpha, kernel)
    deformed = tf.reshape(deformed, [b, n_variants, h, w])
    return tf.concat([deformed, images[:, None]], axis=1)


def elastic_transform(image, alpha, sigma, random_state=None):
    """Single-image elastic deformation, kept for compatibility (runs on the GPU)."""
    image = np.asarray(image, dtype=np.float32)
    kernel = tf.constant(_gaussian_kernel1d(sigma))
    if random_state is None:
        random_state = np.random.RandomState(None)
    raw_dx = (random_state.rand(*image.shape) * 2 - 1).astype(np.float32)
    raw_dy = (random_state.rand(*image.shape) * 2 - 1).astype(np.float32)
    out = _elastic_from_fields(image[None], raw_dx[None], raw_dy[None],
                               np.float32(alpha), kernel)
    return out.numpy()[0]


def getElastics(images, labels, alpha, sigma):
    """Apply elastic deformation to every image in the dataset (4 variants each).

    Same contract as the SciPy version: input (N, 1, H, W), output (5N, H, W) laid out
    as [4 deformations, original] per source image, plus each label repeated 5 times.
    """
    n_variants = 4
    images = np.asarray(images, dtype=np.float32)
    flat = images[:, 0]
    n, h, w = flat.shape

    kernel = tf.constant(_gaussian_kernel1d(sigma))
    alpha = np.float32(alpha)

    out = np.empty((n * (n_variants + 1), h, w), dtype=np.float32)
    for start in range(0, n, ELASTIC_CHUNK):
        chunk = flat[start:start + ELASTIC_CHUNK]
        result = _elastic_batch(tf.constant(chunk), alpha, kernel, n_variants)
        out[start * (n_variants + 1):(start + len(chunk)) * (n_variants + 1)] = \
            result.numpy().reshape(-1, h, w)

    return out, np.repeat(labels, n_variants + 1)


def augmentate(images, labels, alpha, sigma):
    """Elastic-deform the dataset, then add rotated copies at fixed angles."""
    images = images.reshape(images.shape[0], 1, 20, 20)
    deformated, new_labels = getElastics(images, labels, alpha, sigma)
    deformated = deformated.reshape(deformated.shape[0], 20, 20, 1)
    augmented = rotations(deformated, new_labels, [-16, -8, 8, 16])
    return augmented[0].reshape(augmented[0].shape[0], 20, 20, 1), augmented[1]


def augmentate_2(images, labels, alpha, sigma):
    """Elastic-deform the dataset without adding rotated copies."""
    images = images.reshape(images.shape[0], 1, 20, 20)
    deformated, new_labels = getElastics(images, labels, alpha, sigma)
    deformated = deformated.reshape(deformated.shape[0], 20, 20, 1)
    return deformated, new_labels


def experiment(X_train, Y_train, X_test, Y_test, y_test, file, epochs=50, model_label=""):
    """Build, train and evaluate the small CNN (3x3 convs), then save it to `file`."""
    model = Sequential()
    model.add(Conv2D(20, (3, 3), padding='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(40, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(640))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # `decay` was removed from the Keras 3 optimizer; InverseTimeDecay with
    # decay_steps=1 reproduces the old schedule exactly: lr0 / (1 + decay * step).
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
    print_accuracy_report(best_model, X_test, y_test, Y_test)


def experiment_2(X_train, Y_train, X_test_org, Y_test, y_test, file, epochs=50):
    """Variant of `experiment` using 5x5 convs, evaluated on the uncropped test set."""
    model = Sequential()
    model.add(Conv2D(20, (5, 5), padding='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(40, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(640))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # `decay` was removed from the Keras 3 optimizer; InverseTimeDecay with
    # decay_steps=1 reproduces the old schedule exactly: lr0 / (1 + decay * step).
    lr = optimizers.schedules.InverseTimeDecay(0.01, decay_steps=1, decay_rate=1e-6)
    sgd = optimizers.SGD(learning_rate=lr, momentum=0.95, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'],
                  steps_per_execution=STEPS_PER_EXECUTION, jit_compile=USE_XLA)

    print("Training...")
    model.fit(X_train, Y_train, batch_size=128, epochs=epochs, verbose=0)

    print("Evaluating...")
    print_accuracy_report(model, X_test_org, y_test, Y_test)

    model.save(file)


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

    # Vectorised crop + per-image mean subtraction (same result as the per-image loop).
    X_train = np.ascontiguousarray(crop(X_train, 20))
    X_test = np.ascontiguousarray(crop(X_test, 20))

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
    for i in range(1, n_models + 1):
        with run.track_model(i):
            experiment(X_train_2, Y_train_2, X_test, Y_test, y_test,
                       f'./models_random_elastic_2/20x20_{i}.h5', epochs=epochs,
                       model_label=f"model {i}/{n_models}")


if __name__ == '__main__':
    main()
