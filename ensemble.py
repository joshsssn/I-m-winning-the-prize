"""Ensemble evaluation of saved MNIST models.

Each model may expect a different input size (Network3 uses 20x20 crops,
DropConnect uses 24x24), so the MNIST test set is preprocessed separately
for every model. The per-model softmax outputs are then fused and the
ensemble accuracy is compared to each individual model.

Fusion methods:
  mean      - average of the softmax probabilities (default)
  geo       - geometric mean (product of probabilities)
  vote      - majority vote over argmax predictions, ties broken by mean prob
  certainty - trust the single most confident model for each sample
              (the "degree of certainty" fusion from the MNIST-NET10 paper)
  wmean     - weighted mean, weights tuned by grid search (only with --select)

Method selection (--select): both training scripts use the MNIST test set as
their validation data, so there is no separate held-out set. --select splits
the test set into two halves, picks the best method on one half (the
val_accuracy) and reports it on the other half, then swaps the halves.

    python ensemble.py models_random_elastic_2/20x20.h5 models_other_random_elastic_2/24x24.h5
    python ensemble.py models_random_elastic_2/*.h5 models_other_random_elastic_2/*.h5 --method geo
    python ensemble.py ... --save-probs probs.npz
"""
import argparse
import glob
import sys

import numpy as np

from accuracy import _prepare_test_set


def fuse(probs, method, weights=None):
    """probs: array of shape (n_models, n_samples, n_classes)."""
    if method == 'mean':
        return probs.mean(axis=0)
    if method == 'wmean':
        w = np.asarray(weights, dtype='float64')
        return np.tensordot(w / w.sum(), probs, axes=1)
    if method == 'certainty':
        best = probs.max(axis=-1).argmax(axis=0)
        return probs[best, np.arange(probs.shape[1])]
    if method == 'geo':
        return np.exp(np.log(np.clip(probs, 1e-12, 1.0)).mean(axis=0))
    if method == 'vote':
        n_classes = probs.shape[-1]
        votes = np.stack([np.bincount(row, minlength=n_classes)
                          for row in probs.argmax(axis=-1).T])
        # Tie-break with the mean probability (tiny weight so it never overrides a vote).
        return votes + 1e-3 * probs.mean(axis=0)
    raise ValueError(f"Unknown fusion method: {method}")


def collect_predictions(model_paths):
    from tensorflow.keras.models import load_model

    test_sets = {}
    all_probs = []
    y_test = None
    for path in model_paths:
        model = load_model(path)
        crop_size = model.input_shape[1]
        if crop_size not in test_sets:
            test_sets[crop_size] = _prepare_test_set(crop_size)
        X_test, y_test, _ = test_sets[crop_size]
        probs = model.predict(X_test, batch_size=256, verbose=0)
        all_probs.append(probs)
    return np.stack(all_probs), y_test


def report_individual(model_paths, probs, y_test):
    n = len(y_test)
    individual_mistakes = []
    print("\n=== Individual models ===")
    for path, p in zip(model_paths, probs):
        mistakes = np.where(p.argmax(1) != y_test)[0]
        individual_mistakes.append(set(mistakes.tolist()))
        print(f"{path}: {(n - len(mistakes)) / n * 100:.2f}% ({len(mistakes)} errors)")
    return individual_mistakes


def report_ensemble(model_paths, probs, y_test, method, individual_mistakes):
    n = len(y_test)
    fused = fuse(probs, method)
    preds = fused.argmax(1)
    mistakes = np.where(preds != y_test)[0]
    shared = set.intersection(*individual_mistakes)
    union = set.union(*individual_mistakes)

    print(f"\n=== Ensemble ({method}, {len(model_paths)} models) ===")
    print(f"Test accuracy: {(n - len(mistakes)) / n * 100:.2f}% ({n - len(mistakes)}/{n})")
    print(f"Misclassified samples: {len(mistakes)}")
    print(f"Errors shared by every model: {len(shared)} (ensemble cannot fix these)")
    print(f"Errors made by at least one model: {len(union)}")
    if len(mistakes):
        print(f"Misclassified indices: {mistakes.tolist()}")
    return fused


def _accuracy(fused, y, idx):
    return (fused[idx].argmax(1) == y[idx]).mean() * 100


def candidate_methods(probs):
    """All fusion candidates as (name, fused_probs). Weighted means are only
    grid-searched for two models; more models fall back to the fixed methods."""
    cands = [(m, fuse(probs, m)) for m in ('mean', 'geo', 'vote', 'certainty')]
    if probs.shape[0] == 2:
        for w in np.round(np.arange(0.1, 0.95, 0.1), 1):
            cands.append((f'wmean(w={w:.1f})', fuse(probs, 'wmean', weights=[w, 1 - w])))
    return cands


def select_method(probs, y_test, seed=0):
    """2-fold selection on the test set: choose the method on one half by
    val_accuracy, score it on the other half, then swap."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y_test))
    halves = [perm[:len(perm) // 2], perm[len(perm) // 2:]]
    cands = candidate_methods(probs)

    print(f"\n=== Method selection (2-fold split of the test set, seed={seed}) ===")
    print(f"{'method':16s} {'val(A)':>7s} {'held(B)':>8s} {'val(B)':>8s} {'held(A)':>8s} {'full':>7s}")
    for name, fused in cands:
        a, b = _accuracy(fused, y_test, halves[0]), _accuracy(fused, y_test, halves[1])
        print(f"{name:16s} {a:7.2f} {b:8.2f} {b:8.2f} {a:8.2f} {_accuracy(fused, y_test, perm):7.2f}")

    held_correct = 0
    for val_idx, held_idx in (halves, halves[::-1]):
        best_name, best_fused = max(cands, key=lambda c: _accuracy(c[1], y_test, val_idx))
        val_acc, held_acc = _accuracy(best_fused, y_test, val_idx), _accuracy(best_fused, y_test, held_idx)
        held_correct += (best_fused[held_idx].argmax(1) == y_test[held_idx]).sum()
        print(f"Selected on val: {best_name} (val_accuracy {val_acc:.2f}%) -> held-out {held_acc:.2f}%")
    print(f"Cross-validated held-out accuracy of the selected method: {held_correct / len(y_test) * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Ensemble saved MNIST models on the MNIST test set.")
    parser.add_argument('models', nargs='+', help="Path(s) or glob pattern(s) to saved .h5/.keras model files")
    parser.add_argument('--method', choices=['mean', 'geo', 'vote', 'certainty'], default='mean')
    parser.add_argument('--all-methods', action='store_true', help="Report every fusion method")
    parser.add_argument('--select', action='store_true',
                        help="Pick the best fusion method on half of the test set and score it on the other half")
    parser.add_argument('--save-probs', help="Save per-model probabilities and labels to this .npz file")
    args = parser.parse_args()

    model_paths = []
    for pattern in args.models:
        matches = sorted(glob.glob(pattern))
        model_paths.extend(matches if matches else [pattern])
    if len(model_paths) < 2:
        print("Need at least two models to ensemble.")
        sys.exit(1)

    probs, y_test = collect_predictions(model_paths)
    if args.save_probs:
        np.savez(args.save_probs, probs=probs, y_test=y_test, models=np.array(model_paths))

    individual_mistakes = report_individual(model_paths, probs, y_test)
    methods = ['mean', 'geo', 'vote', 'certainty'] if args.all_methods else [args.method]
    for method in methods:
        report_ensemble(model_paths, probs, y_test, method, individual_mistakes)
    if args.select:
        select_method(probs, y_test)


if __name__ == '__main__':
    main()
