# Deepfake analysis for chest X-rays and radiology reports

This repository contains the scripts used to study generated chest X-ray images and reports. It covers three related tasks:

- checking whether an image matches its generated report;
- classifying reports as real or synthetic;
- comparing generated images and reports with real data using embedding-based metrics.

The code is research-oriented rather than packaged as a library. Most experiments are configured by editing the constants near the top of each script.

## Repository layout

```text
Coherence_analysis/
    image2report.py       Image-to-report retrieval with MedSigLIP
    matching.py           One-to-one image/report matching
    test_compare.py       Accuracy from saved round CSV files

Deepfake_detection/       Training and inference variants for real/fake detection
    CXRBert_deepfake_classification/
    Medseglip_deepfake_detection/
    RadDino_deepfake_classification/

image_report_fidelity_analysis/
    compare_deepfake_images.py       Image quality and distribution comparison
    compare_deepfake_reports.py      Report quality and distribution comparison
    deepfake_images_distance_plots.py Nearest-neighbour distance plots
```

Each detector folder contains `dataset.py`, `model.py`, `train.py`, and `inference.py`. The model name and training settings are currently defined in the training and inference scripts.

## Requirements

A recent Python 3 environment is recommended. The scripts use PyTorch, Transformers, pandas, NumPy, SciPy, scikit-learn, Pillow, Matplotlib, seaborn, tqdm, and OpenCLIP. The report comparison scripts also use NLTK, `textstat`, and `language_tool_python`.

Choose the PyTorch installation that matches the CUDA version on the machine. The models are downloaded from Hugging Face on first use, so network access is needed for the initial run. Some scripts can run on CPU, but the larger language and vision models are intended for a CUDA machine.

## Input data

The exact CSV schema depends on the script:

- `Coherence_analysis/matching.py` expects image and report columns. By default they are named `image_path` and `report`.
- The CXR-BERT dataset expects `report`, `label`, and `split`, where `split` is one of `train`, `val`, or `test`.
- The image comparison scripts read a CSV column containing image paths.
- The report comparison script reads a CSV column containing report text. It also supports a headerless CSV for one configured baseline.

Check that paths in the CSVs are readable from the machine running the script. Use `base_dir` in the comparison configs when the CSV stores relative paths.

## Image/report matching

`matching.py` samples rows from a metadata CSV, embeds the images and reports with MedSigLIP, and compares the Hungarian one-to-one assignment with independent per-image argmax predictions:

```bash
python Coherence_analysis/matching.py \
    --generated-csv /path/to/generated_samples.csv \
    --output-dir ./matching_output \
    --count 3 \
    --rounds 50 \
    --seed 42
```

The command writes per-round comparison panels and combined result CSVs. To calculate accuracy from those result files:

```bash
python Coherence_analysis/test_compare.py \
    --dir ./matching_output \
    --rounds all
```

`image2report.py` performs the smaller candidate-report retrieval experiment. Its command-line options are documented in the module docstring and `--help` output.

## Deepfake classifiers

Before training, update the `args` dictionary in the relevant `train.py` file. In particular, set the CSV path, output directory, backbone, and batch settings. The training CSV should include the columns expected by `dataset.py`.

For example:

```bash
cd Deepfake_detection/CXRBert_deepfake_classification
python train.py
python inference.py
```

The same pattern applies to `Medseglip_deepfake_detection` and `RadDino_deepfake_classification`. Training scripts save checkpoints, logs, and metric plots under the configured results directory. Inference scripts write per-example probabilities and predictions to the configured output CSV.

## Fidelity analysis

The two main comparison scripts have a `CONFIG` section near the top. Update the real-data CSV, generated-data CSVs, path columns, sample size, and output folder before running them:

```bash
python image_report_fidelity_analysis/compare_deepfake_images.py
python image_report_fidelity_analysis/compare_deepfake_reports.py
python image_report_fidelity_analysis/deepfake_images_distance_plots.py
```

The image suite compares RAD-DINO and BiomedCLIP features with FID, KID, and manifold-based precision/recall, and produces plots. The report suite includes embedding-based distribution metrics, lexical diversity, readability and grammar-related measures, plus an optional MedGemma judge. The MedGemma path requires access to the gated Hugging Face model and enough GPU memory; use a small sample first.

