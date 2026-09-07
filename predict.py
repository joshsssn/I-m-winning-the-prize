"""Produce a competition submission (id,target) from an ensemble of saved models.

Accepted test inputs:
  * CSV with a header. An `id` column (any case; `ImageId`/`ID` also work) is
    optional; the remaining 784 columns are the 28x28 pixels, row-major.
    Pixel values may be 0-255 or already scaled to 0-1.
  * .npy / .npz holding an array of shape (N, 28, 28) or (N, 784) or (N, 28, 28, 1).
    For .npz the array is read from key `x` if present, otherwise the first key.
  * A directory of PNG/JPG images named <id>.png (28x28 grayscale, white digit
    on black like MNIST). The id is taken from the file name.
  If no ids are found, ids are generated starting at --id-start (default 1001).

    python predict.py ../test -o submission.csv \\
        models_random_elastic_2/*.h5 models_other_random_elastic_2/*.h5

    python predict.py test.csv -o submission.csv \\
        models_random_elastic_2/*.h5 models_other_random_elastic_2/*.h5
"""
import argparse
import glob
import os
import sys

import numpy as np

from accuracy import _center_crop
from ensemble import fuse

ID_COLUMNS = {'id', 'imageid', 'image_id'}


def load_test(path, id_start):
    ids = None
    if os.path.isdir(path):
        from PIL import Image
        files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            raise ValueError(f"No image files found in {path}")
        stems = [os.path.splitext(f)[0] for f in files]
        if all(st.isdigit() for st in stems):
            order = sorted(range(len(files)), key=lambda i: int(stems[i]))
            ids = np.array([int(stems[i]) for i in order])
        else:
            order = sorted(range(len(files)), key=lambda i: files[i])
            ids = np.array([stems[i] for i in order])
        pixels = np.stack([
            np.array(Image.open(os.path.join(path, files[i])).convert('L').resize((28, 28)))
            for i in order])
    elif path.lower().endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(path)
        id_cols = [c for c in df.columns if c.strip().lower() in ID_COLUMNS]
        if id_cols:
            ids = df[id_cols[0]].to_numpy()
            df = df.drop(columns=id_cols)
        pixels = df.to_numpy(dtype='float32')
    elif path.lower().endswith('.npy'):
        pixels = np.load(path)
    elif path.lower().endswith('.npz'):
        data = np.load(path)
        pixels = data['x'] if 'x' in data else data[data.files[0]]
        if 'id' in data:
            ids = data['id']
    else:
        raise ValueError(f"Unsupported test file: {path}")

    pixels = np.asarray(pixels, dtype='float32')
    if pixels.ndim == 2 and pixels.shape[1] == 784 or pixels.ndim == 1:
        pixels = pixels.reshape(-1, 28, 28)
    if pixels.ndim == 4:
        pixels = pixels[..., 0]
    if pixels.shape[1:] != (28, 28):
        raise ValueError(f"Expected 28x28 images, got shape {pixels.shape}")
    if pixels.max() > 1.0:
        pixels /= 255.0
    if ids is None:
        ids = np.arange(id_start, id_start + len(pixels))
    return ids, pixels[..., np.newaxis]


def preprocess(images, crop_size):
    """Same preprocessing as accuracy.py: center crop, then subtract per-image mean."""
    if crop_size != 28:
        images = np.array([_center_crop(img, crop_size) for img in images])
    return images - images.mean(axis=(1, 2, 3), keepdims=True)


def ensemble_predict(model_paths, images, method='mean'):
    from tensorflow.keras.models import load_model

    cache, all_probs = {}, []
    for path in model_paths:
        model = load_model(path)
        crop_size = model.input_shape[1]
        if crop_size not in cache:
            cache[crop_size] = preprocess(images, crop_size)
        all_probs.append(model.predict(cache[crop_size], batch_size=256, verbose=0))
        print(f"loaded {path} ({crop_size}x{crop_size})")
    return fuse(np.stack(all_probs), method)


def main():
    parser = argparse.ArgumentParser(description="Write an id,target submission from an ensemble of saved models.")
    parser.add_argument('test_file', help="Competition test file (.csv, .npy or .npz)")
    parser.add_argument('models', nargs='+', help="Path(s) or glob pattern(s) to saved model files")
    parser.add_argument('-o', '--output', default='submission.csv')
    parser.add_argument('--method', choices=['mean', 'geo', 'vote', 'certainty'], default='mean')
    parser.add_argument('--id-start', type=int, default=1001, help="First id when the test file has no id column")
    args = parser.parse_args()

    model_paths = []
    for pattern in args.models:
        matches = sorted(glob.glob(pattern))
        model_paths.extend(matches if matches else [pattern])
    if not model_paths:
        print("No model files found.")
        sys.exit(1)

    ids, images = load_test(args.test_file, args.id_start)
    print(f"{len(images)} test images from {args.test_file}")
    fused = ensemble_predict(model_paths, images, args.method)
    targets = fused.argmax(1)

    with open(args.output, 'w') as f:
        f.write("id,target\n")
        for i, t in zip(ids, targets):
            f.write(f"{i},{t}\n")
    print(f"wrote {args.output} ({len(targets)} rows); predicted digit counts: {np.bincount(targets, minlength=10).tolist()}")


if __name__ == '__main__':
    main()
