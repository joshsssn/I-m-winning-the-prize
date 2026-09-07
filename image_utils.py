"""Image transform helpers that used to live in keras.preprocessing.image
(random_rotation, random_shear, random_shift, transform_matrix_offset_center,
apply_transform). Those were removed from modern Keras, so they are
reimplemented here with scipy.
"""
import numpy as np
from scipy.ndimage import affine_transform


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    return np.dot(np.dot(offset_matrix, matrix), reset_matrix)


def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]

    channel_images = [
        affine_transform(
            x_channel, final_affine_matrix, final_offset,
            order=1, mode=fill_mode, cval=cval)
        for x_channel in x
    ]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                     fill_mode='nearest', cval=0.):
    theta = np.deg2rad(np.random.uniform(-rg, rg))
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    return apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)


def random_shear(x, intensity, row_axis=1, col_axis=2, channel_axis=0,
                  fill_mode='nearest', cval=0.):
    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                              [0, np.cos(shear), 0],
                              [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    return apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                  fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                    [0, 1, ty],
                                    [0, 0, 1]])
    transform_matrix = translation_matrix
    return apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
