import random

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import mnist
from scipy.ndimage import gaussian_filter, map_coordinates

from image_utils import (
    random_rotation, random_shear, random_shift,
    transform_matrix_offset_center, apply_transform,
)
from accuracy import print_accuracy_report
from timing import EpochTimer, RunEstimator, timed_stage

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
    """Crop a fixed-size window from an image, starting at (4, 4)."""
    x_0 = 4
    y_0 = 4
    return img[x_0:x_0 + size, y_0:y_0 + size]


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


def elastic_transform(image, alpha, sigma, random_state=None):
    """Apply an elastic deformation to an image (chsasank/elastic_transform.py)."""
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


def getElastics(images, labels, alpha, sigma):
    """Apply elastic deformation to every image in the dataset (4 variants each)."""
    augmented = []
    augmented_labels = []
    for i in range(len(images)):
        news = np.array([elastic_transform(images[i][0], alpha, sigma) for j in range(4)])
        news = np.append(news, images[i][0])
        augmented.append(news)
        augmented_labels.append([labels[i], labels[i], labels[i], labels[i], labels[i]])

    return (np.reshape(augmented, newshape=(5 * len(images), images.shape[2], images.shape[3])),
            np.reshape(augmented_labels, 5 * len(images)))


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

    sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.95, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

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

    sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.95, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

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

    X_train = np.array([crop(X_train[i], 20) for i in range(len(X_train))])
    X_test = np.array([crop(X_test[i], 20) for i in range(len(X_test))])

    X_train = np.array([X_train[i] - np.mean(X_train[i]) for i in range(len(X_train))])
    X_test = np.array([X_test[i] - np.mean(X_test[i]) for i in range(len(X_test))])

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
                       f'./models_random_elastic_2/modelo{i}.h5', epochs=epochs,
                       model_label=f"model {i}/{n_models}")


if __name__ == '__main__':
    main()
