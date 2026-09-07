"""Timing helpers: stage timers and a Keras callback that estimates total
training time from the first completed epoch/model.
"""
import time
from contextlib import contextmanager

from tensorflow.keras.callbacks import Callback


def format_duration(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


@contextmanager
def timed_stage(name):
    """Print how long a block of code took to run."""
    start = time.time()
    print(f"[{name}] starting...")
    yield
    elapsed = time.time() - start
    print(f"[{name}] done in {format_duration(elapsed)}")


class EpochTimer(Callback):
    """Keras callback that times each epoch and prints a running ETA for the
    current model, based on the average epoch duration seen so far."""

    def __init__(self, total_epochs, model_label=""):
        super().__init__()
        self.total_epochs = total_epochs
        self.model_label = model_label
        self._epoch_start = None
        self._durations = []

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._epoch_start
        self._durations.append(elapsed)
        avg = sum(self._durations) / len(self._durations)
        remaining = self.total_epochs - (epoch + 1)
        eta = avg * remaining
        prefix = f"[{self.model_label}] " if self.model_label else ""
        print(f"{prefix}Epoch {epoch + 1}/{self.total_epochs} took {format_duration(elapsed)} "
              f"(avg {format_duration(avg)}/epoch, ETA for this model: {format_duration(eta)})")


class RunEstimator:
    """Tracks per-model durations across an entire run (all models in a
    script) and prints an ETA for the remaining models after each one."""

    def __init__(self, total_models):
        self.total_models = total_models
        self._durations = []

    @contextmanager
    def track_model(self, model_index):
        start = time.time()
        yield
        elapsed = time.time() - start
        self._durations.append(elapsed)
        avg = sum(self._durations) / len(self._durations)
        remaining = self.total_models - model_index
        eta = avg * remaining
        print(f"=== Model {model_index}/{self.total_models} finished in {format_duration(elapsed)} "
              f"(avg {format_duration(avg)}/model, ETA for remaining models: {format_duration(eta)}) ===")
