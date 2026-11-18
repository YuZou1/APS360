# Early Sketch Recognition

An APS360 progress-report experiment on classifying unfinished sketches from
ten Quick, Draw! categories. The baseline sees a grayscale image. The second
model also reads the pen movement sequence through a GRU.

Both models were trained on 10,000 source drawings for six epochs and evaluated
on the same 2,000 validation drawings at each completion level. The pilot's
mean top-1 accuracy across the four levels was 46.41% for CNN and 49.89% for
CNN + GRU. The official test split was kept separate.

| Drawing completion | CNN top-1 | CNN + GRU top-1 |
| --- | ---: | ---: |
| 25% | 30.75% | 32.85% |
| 50% | 49.10% | 50.60% |
| 75% | 56.75% | 56.65% |
| 100% | 49.05% | 59.45% |

## Reproduce the pilot

With Python 3.12 in a virtual environment, install the CPU build of PyTorch
and the other dependencies:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python train_sketch_models.py --download --device cpu --output rerun
```

The training command fetches the ten official Sketch-RNN archives on first
run (about 124 MB). Use only trusted official archives: the NumPy object
arrays need pickle support. The included manifest records the source indices
within the official train and validation splits. The run uses the first
1,000 training and 200 validation drawings per class, batches of 64, seed 360,
cross-entropy and Adam with a learning rate of 0.001.

## Predict with the bundled weights

After downloading the official archives as above, run:

```bash
python predict_sketch.py --category airplane --index 60
python predict_sketch.py --category airplane --index 29
```

The two indices identify the validation drawings used in the progress report.
The command prints both models' top three guesses at each completion level.
If you only want inference, you can download just
[airplane.npz](https://storage.googleapis.com/quickdraw_dataset/sketchrnn/airplane.npz)
into `quickdraw/`.

## Files and sources

- `prepare_sketches.py`: sample selection and sketch rendering.
- `train_sketch_models.py`: model definitions, training, and evaluation.
- `predict_sketch.py`: inference using the included checkpoints.
- `build_experiment_evidence.py`: rebuilds the report's curves and examples
  from the recorded outputs. Run with
  `--data quickdraw --run experiment --output figures`.
- `experiment/`: configuration, checkpoints, metrics, example predictions,
  validation predictions, and training history.

Data: [Google Creative Lab, Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset),
CC BY 4.0. Stroke format: Ha and Eck,
[A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477).
