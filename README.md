# ECM-402 Lab 4 - CNNs for Multi-Dimensional Signals

- **Name / Roll number:** <fill in>
- **GitHub repository:** <paste link here - required for verification>
- Dataset: Fashion-MNIST (downloaded automatically by `torchvision`; `data/` is git-ignored)

## How to reproduce

```bash
pip install -r requirements.txt
python src/main.py            # full run: every figure, table, report.pdf and the hardware block below
python src/main.py --quick    # ~5 minute smoke test (writes to results_quick/, numbers are NOT meaningful)
```

Options: `--epochs 15` (custom CNN / ablations, >= 10), `--tl-epochs 6` (5-8), `--workers 2`, `--skip-bonus`.
Everything the report quotes comes from this one command: figures -> `results/figures/`, tables -> `results/tables/`,
`report.pdf` is generated at the repo root. No GPU is assumed (64x64 ResNet-18 inputs, 10k/2k stratified subset).

## Protocol

train -> validate/select -> freeze -> test once. All model selection (architecture, augmentation, fine-tuning strategy,
best epoch) uses the validation split only. The official test set is evaluated once in Task 5 (the MLP is evaluated on it
only to fill the efficiency table). Grad-CAM (Task 6) analyses the already-frozen ResNet-18.

## Layout

```
src/data.py        transforms, splits, loaders, classical kernels (Task 1)
src/models.py      custom CNN, MLP baseline, ResNet-18 setup, hand-computed layer table, receptive field
src/train.py       training / validation loops, metrics
src/ablations.py   Task 3 controlled experiments
src/transfer.py    Task 4 fine-tuning strategies + LR scheduler
src/gradcam.py     Task 6 Grad-CAM (hooks, from scratch)
src/denoise.py     bonus denoising autoencoder
src/viz.py         all figures
src/report.py      builds report.pdf, tables, README block
src/main.py        reproduces all results
```

## Hardware, seeds and runtimes

<!-- AUTO:START -->
(filled automatically at the end of `python src/main.py`)
<!-- AUTO:END -->
