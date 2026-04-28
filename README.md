
# RPPNet

## Quick Start

This repository contains RPPNet, a two-stage music generation pipeline. Below are the minimal steps to preprocess data, train both modules, and run inference to generate melodies.

### 1) Data preprocessing

1. Place your dataset under: `workspace/DataProcess`
2. Run the preprocessing script:

```bash
cd workspace/DataProcess
python data.py
```

Follow the interactive prompts to set the input path, output path, algorithm, and split configuration.

### 2) Training

From the repository root, run the training commands for each module:

```bash
# Train the RPP-level model
python manage.py -m train_RPP

# Train the Note-level model
python manage.py -m train_Note
```

Configuration for datasets, model hyperparameters, and other settings are located in each module's config file:

- `workspace/RPP_level/config/config.yaml`
- `workspace/Note_level/config/config.yaml`

Edit those files to change data paths, training parameters, or model hyperparameters.

### 3) Create a new inference workspace

Create a new experiment/inference folder (under `Exp_Record`) by running:

```bash
python manage.py -n
```

This command creates a timestamped directory under `Exp_Record` with subfolders for `midi`, `pkl`, `midi_inference`, etc.

### 4) Inference

Run RPP-level inference first, then Note-level inference. From the repository root:

```bash
# RPP-level inference (feature generation)
python manage.py -m inference_RPP

# Note-level inference (convert features to notes/midi)
python manage.py -m inference_Note
```

Outputs (generated MIDI and intermediate files) will be saved in the latest `Exp_Record/<timestamp>` folder created by the `-n` command.

### 5) Evaluation
1. The code to evaluate se is in: `se_metric`
2. The code to evaluate ppl is in: `workspace/Note_level/workspace/evaluate_ppl.py`