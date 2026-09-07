# MNIST-NET10

MNIST-NET10 is a heterogeneous fusion of deep networks aiming for a 0.1% error rate on MNIST digit classification. It combines two ensembles, FS1 and FS2, built from a mix of CNN architectures and data-augmentation strategies.

Full details: [MNIST-NET10: A heterogeneous deep networks fusion based on the degree of certainty to reach 0.1% error rate](https://www.researchgate.net/publication/340954880_MNIST-NET10_A_heterogeneous_deep_networks_fusion_based_on_the_degree_of_certainty_to_reach_01_error_rate_Ensembles_overview_and_proposal)

## Ensembles

**FS1** (CapsNet | MCDNN | DropConnect_2 | CapsNet | MCDNN | DropConnect_1 | DropConnect_2 | Network3 | DropConnect_2):
1. Pre-trained CapsNet — [Sarasra/models](https://github.com/Sarasra/models/tree/master/research/capsules)
2. MCDNN — [xanwerneck/ml_mnist](https://github.com/xanwerneck/ml_mnist)
3. Network3 with data augmentation — [Network3.py](Network3.py)
4. DropConnect with data augmentation — [DropConnect.py](DropConnect.py)

**FS2** (ECOC | PrE | MLP→LS | MLP):
1. CapsNet as a data transformer — [Sarasra/models](https://github.com/Sarasra/models/tree/master/research/capsules)
2. MATLAB code — [tsc.uc3m.es/~ralvear/Software.htm](http://www.tsc.uc3m.es/~ralvear/Software.htm)

## This repository

This repo contains the two Keras/TensorFlow training scripts for FS1's CNN components:

- [Network3.py](Network3.py) — a small CNN (two 3x3 conv blocks) trained on 20x20 center crops of MNIST digits.
- [DropConnect.py](DropConnect.py) — a deeper CNN (VGG-style, two 3x3 conv-conv-pool blocks) trained on 24x24 crops.
- [image_utils.py](image_utils.py) — random rotation/shear/shift helpers, reimplemented with scipy since these were removed from modern Keras.
- [accuracy.py](accuracy.py) — shared accuracy reporting used during training, and a standalone CLI to re-evaluate any saved model.
- [ensemble.py](ensemble.py) — fuses the predictions of several saved models and reports the ensemble accuracy.

Both training scripts share the same pipeline: crop the digit, subtract the per-image mean, apply random rotation/shear/shift augmentation, then apply elastic deformation, before training a CNN with SGD.

## Setup

The scripts require TensorFlow, which does not support Python 3.13. Use Python 3.10-3.12.

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

## Usage

### Training

Each script trains 5 models (by default) and saves them to `./models_random_elastic_2/modeloN.h5`.

```bash
python Network3.py
python DropConnect.py
```

By default both scripts run in **smoke-test mode** (`SMOKE_TEST = True`): a small data subset and a single epoch, just to verify everything runs end-to-end in a couple of minutes. To run the full training as originally intended (60,000 training images, 50 epochs, 5 models per script — expect this to take a long time on CPU), open the script and set:

```python
SMOKE_TEST = False
```

### Evaluating accuracy

`accuracy.py` loads a saved model, detects the input size it was trained on (20x20 or 24x24), applies the matching preprocessing to the MNIST test set, and reports loss, accuracy, and misclassified sample indices.

```bash
python accuracy.py models_random_elastic_2/modelo1.h5
python accuracy.py models_random_elastic_2/*.h5
```

### Ensembling

`ensemble.py` loads several saved models (each preprocessed to its own crop size), fuses their softmax outputs, and reports the ensemble accuracy next to each individual model. Fusion methods: `mean` (default), `geo` (geometric mean), `vote` (majority vote) and `certainty` (most confident model wins). Because the training scripts validate on the MNIST test set, `--select` splits the test set in two, picks the method by val_accuracy on one half and reports it on the other half.

```bash
python ensemble.py models_random_elastic_2/modelo1.h5 models_other_random_elastic_2/modelo1.h5
python ensemble.py models_random_elastic_2/*.h5 models_other_random_elastic_2/*.h5 --all-methods --select
```

## Notes

- No GPU support on native Windows with TensorFlow >= 2.11 — use WSL2 or the TensorFlow-DirectML plugin if you need GPU acceleration.
- Models are saved in the legacy HDF5 (`.h5`) format for consistency with the original scripts.
