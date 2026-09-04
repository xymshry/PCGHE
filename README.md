# PCGHE: Plaintext ECG Baseline for CKKS Inference

[![CI](https://github.com/xymshry/PCGHE/actions/workflows/ci.yml/badge.svg)](https://github.com/xymshry/PCGHE/actions/workflows/ci.yml)

PCGHE is a reproducible plaintext baseline for a privacy-preserving ECG
classification study. It converts annotated heartbeats from the MIT-BIH
Arrhythmia Database into a compact feature vector and trains a linear model
whose final computation can later be evaluated with CKKS homomorphic
encryption.

This repository implements and validates the plaintext half of the study. It
does **not** claim to provide a complete encrypted deployment yet. The exported
`ckks_interface.npz` and `ckks_test_vectors.npz` files define the exact
interface for a later CKKS implementation.

## Research question

Can a wavelet-PCA representation preserve useful ECG arrhythmia classification
performance while reducing the dimension of the vector that will be encrypted
for linear inference?

The main experiment is intentionally narrow:

```text
MIT-BIH ECG (MLII)
  -> 0.5-40 Hz zero-phase Butterworth band-pass filter
  -> reference R-peak segmentation from atr annotations
  -> 256-sample beat-wise normalization
  -> db4 level-4 discrete wavelet transform (256 coefficients)
  -> training-only standardization
  -> training-only PCA: 256 -> 32
  -> class-weighted logistic regression
  -> plaintext logit and N/V decision
```

The linear inference value is

```text
logit = dot(weight, pca_feature) + bias
```

The binary decision is `V` when `logit > 0` and `N` otherwise. A sigmoid is not
required for this decision and is deliberately omitted from the future CKKS
circuit.

## Current benchmark

The included benchmark was run on the MIT-BIH records downloaded on 2026-09-03
with Python 3.14.5 on Windows 10. The model was trained on all DS1 records and
evaluated once on the untouched DS2 records.

| Configuration | AUROC | AUPRC | F1 | Balanced accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Raw, no PCA | 0.932 | 0.568 | 0.491 | 0.849 | 0.836 | 0.862 |
| Raw, PCA-32 | 0.940 | 0.599 | 0.506 | 0.863 | 0.864 | 0.863 |
| DWT, no PCA | 0.931 | 0.539 | 0.495 | 0.855 | 0.849 | 0.860 |
| **DWT, PCA-32 (main)** | **0.943** | 0.431 | **0.551** | **0.910** | **0.953** | 0.867 |

The test set contains 39,647 selected beats: 36,428 `N` beats and 3,219 `V`
beats. Because the positive class is uncommon, AUROC must not be reported by
itself; the repository also reports AUPRC, F1, balanced accuracy, sensitivity,
and specificity.

The PCA dimension sweep gives the following test results:

| PCA dimension | AUROC | F1 | Balanced accuracy | Sensitivity | Specificity |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.815 | 0.297 | 0.729 | 0.750 | 0.708 |
| 16 | 0.944 | 0.539 | 0.903 | 0.944 | 0.862 |
| **32** | **0.943** | **0.551** | **0.910** | **0.953** | 0.867 |
| 64 | 0.914 | 0.498 | 0.854 | 0.845 | 0.863 |
| 128 | 0.931 | 0.494 | 0.854 | 0.849 | 0.860 |
| 256 | 0.931 | 0.494 | 0.854 | 0.849 | 0.860 |

These numbers are a reproducibility checkpoint, not a clinical performance
claim. Regenerate them after changing software, hardware, record selection, or
model parameters.

## Dataset

The experiment uses the **MIT-BIH Arrhythmia Database** distributed through
[PhysioNet](https://physionet.org/content/mitdb/1.0.0/). The database contains
two-channel ECG recordings sampled at 360 Hz and expert beat annotations.

Only these annotation symbols are used:

```text
N = normal beat                              -> label 0
V = ventricular ectopic beat (PVC)           -> label 1
```

The code uses the `MLII` lead when available and otherwise the first lead. R
peaks come from the database `atr` annotations. This isolates classification
and later encryption from automatic detector errors. Automatic Pan-Tompkins or
NeuroKit R-peak detection is left for a future robustness experiment.

### Patient-independent protocol

The split is by record, never by individual heartbeat. This prevents beats from
the same patient appearing in both training and test data.

| Split | Records | Use |
|---|---|---|
| DS1 | `101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230` | model fitting |
| DS2 | `100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234` | final test only |

The commonly excluded paced records (`102`, `104`, `107`, and `217`) are not
downloaded. There is no separate validation set in the main run because the
primary hyperparameters are fixed in advance. If tuning is added, use
record-grouped cross-validation inside DS1 and leave DS2 untouched.

## Signal and feature definition

### Filtering

Each selected lead is filtered with a fourth-order zero-phase Butterworth
band-pass filter:

```text
low cutoff:   0.5 Hz
high cutoff: 40.0 Hz
sampling rate: 360 Hz
```

`scipy.signal.sosfiltfilt` is used so the filter does not introduce a phase
shift around the annotated R peak.

### Beat segmentation

For an annotated R peak at sample `r`, the code extracts:

```text
start = r - 90
end   = r + 166
length = 256 samples
```

Beats that extend beyond the recording are skipped. Each beat is independently
z-normalized as `(beat - mean) / (std + 1e-8)`.

### Wavelet features

The default extractor calls PyWavelets with:

```python
pywt.wavedec(
    beat,
    wavelet="db4",
    level=4,
    mode="periodization",
)
```

All coefficient arrays are concatenated. With a 256-sample beat and
`periodization` mode, the feature vector has 256 values.

### Standardization, PCA, and logistic regression

The scaler and PCA are fitted on DS1 only. DS2 is transformed using the saved
DS1 parameters. The primary model is:

```python
StandardScaler()
PCA(n_components=32, svd_solver="full")
LogisticRegression(
    class_weight="balanced",
    solver="lbfgs",
    max_iter=2000,
)
```

`class_weight="balanced"` compensates for the normal/PVC imbalance during
training. The reported threshold is zero on the `decision_function` logit.

## Repository layout

```text
PCGHE/
├── README.md
├── THIRD_PARTY_SOURCES.md
├── LICENSE
├── pyproject.toml
├── docs/
│   └── ASSEMBLY_GUIDE_ZH.md
├── src/pcghe_ecg/
│   ├── config.py       # fixed records and experiment parameters
│   ├── dataset.py      # PhysioNet download and annotated beat extraction
│   ├── preprocess.py   # Butterworth filtering and normalization
│   ├── features.py     # raw and db4 DWT features
│   ├── model.py        # scaler, PCA, logistic model, CKKS export
│   ├── evaluation.py   # metrics and diagnostic plots
│   ├── experiment.py   # end-to-end experiment orchestration
│   └── cli.py          # pcghe-ecg command-line interface
└── tests/
    ├── test_config.py
    ├── test_dataset.py
    ├── test_features.py
    └── test_model.py
```

The local `references/` directory contains source archives used for code
review, but it is intentionally excluded from Git. The `data/`, `cache/`,
`results/`, `.venv/`, and generated package metadata directories are also
ignored.

## Platform support

The Python implementation is platform-independent and supports Linux and
Windows. It uses `pathlib` for paths, the non-interactive Matplotlib `Agg`
backend for plots, and a Python console entry point instead of an operating
system-specific launcher. Every push is tested on Ubuntu and Windows with
Python 3.10 and 3.13, plus a Python 3.11 Conda environment on Ubuntu, by GitHub
Actions.

No C/C++ compilation is normally required because the supported Python
versions have binary wheels for NumPy, SciPy, PyWavelets, pandas,
scikit-learn, Matplotlib, and WFDB. On a minimal Debian/Ubuntu installation,
install the virtual-environment package first if `python3 -m venv` is missing:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
```

If `apt` is unavailable, returns a mirror/proxy error, or the system Python is
3.8, use the Conda installation below. The project requires Python 3.10 or
newer, so installing `python3.8-venv` is not sufficient for this repository.

## Installation

Python 3.10 or newer is required. Python 3.14.5 was used for the included
benchmark.

### Linux with Conda (recommended for Ubuntu 20.04)

This method does not use the system Python, `ensurepip`, or the Ubuntu APT
mirror:

> If your Anaconda installation is in another user's/home partition and your
> own home directory is full, skip the named-environment commands in this
> section. Use [Use another location's Conda without making it global](#use-another-locations-conda-without-making-it-global)
> below. That procedure puts the environment and caches on the PCGHE disk.

```bash
git clone https://github.com/xymshry/PCGHE.git
cd PCGHE
conda env create -f environment.yml
conda activate pcghe
python --version
pcghe-ecg --help
```

If the shell prints `conda: command not found`, Conda is either not installed
or its shell hook has not been loaded. Check for an existing installation
first:

```bash
command -v conda || true
find "$HOME" /opt -type f -path "*/bin/conda" 2>/dev/null | head
```

If a path is found, load that installation for the current Bash session. For
example, if the result is `/opt/miniconda3/bin/conda`:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate
```

If no path is found, install Miniconda in your home directory. This does not
require `apt`, `sudo`, `ensurepip`, or a system Python package:

```bash
cd /tmp
curl -fL -o miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
conda --version
```

For an ARM64 machine, replace `Linux-x86_64` with `Linux-aarch64` in the
download URL. If `curl` is unavailable, use `wget -O miniconda.sh URL` instead.
If the download is blocked, configure the same HTTP/HTTPS proxy used by the
server before running `curl`, for example:

```bash
export http_proxy=http://192.168.104.9:7890
export https_proxy=http://192.168.104.9:7890
```

Then return to the project and create the PCGHE environment:

```bash
cd /home/C/xieyiming/PCGHE
conda env create -f environment.yml
conda activate pcghe
python --version
python -m pytest
```

### Keep Anaconda in the user directory and put the environment on the project disk

The Conda installation directory, project directory, and environment directory
can be different. This is useful when the user home directory is full. For
example, if Anaconda is installed at `/home/C/xieyiming/anaconda3` but the
project disk is `/home/C/xieyiming/PCGHE`, run:

```bash
CONDA_BASE_DIR=/home/C/xieyiming/anaconda3
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"

cd /home/C/xieyiming/PCGHE
export CONDA_PKGS_DIRS="$PWD/.conda-pkgs"
export PIP_CACHE_DIR="$PWD/.pip-cache"
ENV_PREFIX="$PWD/.conda-env"

conda create --prefix "$ENV_PREFIX" python=3.11 pip -y
conda activate "$ENV_PREFIX"
python -m pip install -e ".[dev]"
python -m pytest
pcghe-ecg --help
```

The `conda create --prefix` command puts the environment inside the project
directory instead of Conda's default `~/.conda/envs`. The two cache variables
also keep downloaded Conda and pip packages off the full home partition. The
directories `.conda-env/`, `.conda-pkgs/`, and `.pip-cache/` are ignored by Git.

For later sessions, load Anaconda and activate the same environment with its
absolute path:

```bash
source /home/C/xieyiming/anaconda3/etc/profile.d/conda.sh
cd /home/C/xieyiming/PCGHE
conda activate "$PWD/.conda-env"
```

If your Anaconda path is different, replace `/home/C/xieyiming/anaconda3` with
the directory that contains `bin/conda`. The project does not need to be moved
into the Anaconda directory.

### Use another location's Conda without making it global

If Conda is found at `/home/A/xieyiming/anaconda3/bin/conda` while PCGHE must
run under `/home/C/xieyiming/PCGHE`, do not run `conda init`, do not edit the
system `PATH`, and do not source Conda's shell hook. Invoke that executable by
its absolute path only to create a project-local environment:

```bash
cd /home/C/xieyiming/PCGHE

CONDA_EXE=/home/A/xieyiming/anaconda3/bin/conda
ENV_PREFIX="$PWD/.conda-env"
export CONDA_PKGS_DIRS="$PWD/.conda-pkgs"
export PIP_CACHE_DIR="$PWD/.pip-cache"

"$CONDA_EXE" --version
"$CONDA_EXE" create --prefix "$ENV_PREFIX" -c conda-forge python=3.11 pip -y
"$ENV_PREFIX/bin/python" -m pip install -e ".[dev]"
"$ENV_PREFIX/bin/python" -m pytest
"$ENV_PREFIX/bin/pcghe-ecg" --help
```

This does not register Conda for all users. The `/home/A` installation is used
only as the executable that creates and runs `/home/C/xieyiming/PCGHE/.conda-env`.
All project packages and caches remain under `/home/C/xieyiming/PCGHE`.

The commands deliberately use the environment's absolute Python path after
creation. They therefore do not require `conda activate`, `conda init`, or a
shell hook.

After creation, Conda is not required to run the installed program. Call the
environment executables directly:

```bash
cd /home/C/xieyiming/PCGHE
.conda-env/bin/python -m pytest
.conda-env/bin/pcghe-ecg --help
.conda-env/bin/pcghe-ecg download --data-dir data/mitdb
.conda-env/bin/pcghe-ecg run \
  --data-dir data/mitdb \
  --output-dir results/main \
  --feature dwt \
  --pca-components 32
```

Before using an Anaconda installation owned by another account, check that it
is readable and executable, but do not change its permissions or ownership:

```bash
test -x /home/A/xieyiming/anaconda3/bin/conda \
  && echo "Conda executable is available" \
  || echo "Conda executable is not usable by this account"
```

To make the command available automatically in future Bash sessions, run:

```bash
conda init bash
exec bash
conda activate pcghe
```

The environment file creates an isolated environment named `pcghe` with
Python 3.11 and installs this project in editable mode with its test
dependencies.

If `conda activate` reports that the shell is not initialized, initialize it
for the current Bash session:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pcghe
```

Alternatively, do not activate the environment and run every command through
Conda:

```bash
conda run -n pcghe python -m pytest
conda run -n pcghe pcghe-ecg download --data-dir data/mitdb
conda run -n pcghe pcghe-ecg run --data-dir data/mitdb --output-dir results/main --feature dwt --pca-components 32
```

To update an existing environment after pulling repository changes:

```bash
conda env update -n pcghe -f environment.yml --prune
```

### Linux with venv

```bash
git clone https://github.com/xymshry/PCGHE.git
cd PCGHE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Windows PowerShell

```powershell
git clone https://github.com/xymshry/PCGHE.git
cd PCGHE
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The main dependencies are:

```text
numpy          numerical arrays
scipy          digital filtering
PyWavelets     discrete wavelet transform
wfdb           PhysioNet/WFDB records and annotations
scikit-learn   scaling, PCA, logistic regression, metrics
pandas         CSV result tables
matplotlib     ROC, precision-recall, and confusion-matrix plots
```

### Ubuntu `ensurepip` / APT troubleshooting

The following error is caused by the operating-system Python installation, not
by PCGHE:

```text
The virtual environment was not created successfully because ensurepip is not
available.
```

On Ubuntu 20.04, the suggested `python3.8-venv` package still provides Python
3.8, while PCGHE requires Python 3.10 or newer. If the configured APT mirror
also returns HTTP 502, do not keep retrying that package solely for PCGHE. Use
the Conda method above to create a Python 3.11 environment independently of
APT.

## Quick start

The commands below are identical on Linux and Windows after activating the
virtual environment as shown above. Forward-slash paths are accepted on both
platforms.

### 1. Download MIT-BIH

The downloader is resumable: existing non-empty files are reused and failed
HTTP requests are retried.

```bash
pcghe-ecg download --data-dir data/mitdb
```

The download requires access to PhysioNet. ECG data are not included in this
repository and must not be committed to Git.

### 2. Run the main plaintext experiment

```bash
pcghe-ecg run --data-dir data/mitdb --output-dir results/main --feature dwt --pca-components 32
```

The default cache is `cache/mitdb_nv_beats.npz`. To rebuild it after changing
the signal protocol:

```bash
pcghe-ecg run --data-dir data/mitdb --cache-file cache/mitdb_nv_beats.npz --force-cache --output-dir results/main
```

To run raw beats rather than DWT features, use `--feature raw`. To omit PCA,
use `--no-pca`.

### 3. Run all baselines

```bash
pcghe-ecg baselines --data-dir data/mitdb --output-dir results/baselines
```

This runs `raw_no_pca`, `raw_pca32`, `dwt_no_pca`, and `dwt_pca32`.

### 4. Run the PCA dimension sweep

```bash
pcghe-ecg sweep --data-dir data/mitdb --output-dir results/dimension_sweep
```

To use a smaller custom sweep:

```bash
pcghe-ecg sweep --data-dir data/mitdb --output-dir results/dimension_sweep_small --dimensions 16 32 64
```

### 5. Run tests

The tests use synthetic signals and mocked WFDB records, so they do not need
the MIT-BIH download:

```bash
python -m pytest
```

## Output files

A main run creates:

```text
results/main/
├── config.json
├── class_counts.csv
├── metrics.json
├── metrics.csv
├── predictions_test.csv
├── roc_pr_curves.png
├── confusion_matrix.png
└── artifacts/
    ├── plaintext_pipeline.joblib
    ├── ckks_interface.npz
    ├── ckks_test_vectors.npz
    └── artifact_metadata.json
```

`config.json` captures the protocol, parameters, software versions, and the
maximum error between the manually computed and library logits.

`class_counts.csv`, `metrics.json`, and `metrics.csv` contain class counts and
AUROC, AUPRC, F1, balanced accuracy, sensitivity, specificity, and confusion
matrix counts.

`predictions_test.csv` contains the record ID, sample index, original symbol,
true label, plaintext logit, and predicted label for every selected DS2 beat.

### CKKS handoff artifacts

`ckks_interface.npz` contains:

```text
scaler_mean
scaler_scale
pca_mean
pca_components
pca_explained_variance_ratio
weight
bias
threshold
```

The expected future server operation is:

```text
encrypted_logit = dot(encrypted_pca_feature, weight) + bias
```

`ckks_test_vectors.npz` contains transformed DS2 PCA vectors, labels, record
IDs, sample indices, and plaintext logits. A CKKS implementation can use this
file to compare decrypted logits against the exact plaintext reference.

## Reproducibility checks

Before trusting a new result, verify:

1. DS1 and DS2 record lists are disjoint.
2. `StandardScaler.fit` and `PCA.fit` are called only on DS1.
3. DWT output has shape `(number_of_beats, 256)`.
4. The main exported weight has shape `(32,)`.
5. `manual_logit_max_abs_error` is close to zero.
6. No raw data, cache, model artifact, or result directory is staged for Git.

The `PlaintextModel.verify_manual_logit()` check enforces that exported linear
weights reproduce scikit-learn's decision logits.

## Open-source assembly

The project uses public package APIs rather than copying internal source code:

| Component | Project usage |
|---|---|
| [wfdb-python](https://github.com/MIT-LCP/wfdb-python) | `rdrecord`, `rdann`, and PhysioNet record download |
| [PyWavelets](https://github.com/PyWavelets/pywt) | `wavedec` with db4, level 4, periodization |
| [SciPy](https://github.com/scipy/scipy) | zero-phase Butterworth filtering |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | scaler, PCA, logistic regression, and metrics |
| [NeuroKit2](https://github.com/neuropsychology/NeuroKit) | reviewed as a future R-peak detector; no code copied |
| [Pan-Tompkins example](https://github.com/antimattercorrade/Pan_Tompkins_QRS_Detection) | algorithm reference only; no source copied because no license was declared |

The detailed adaptation notes and pinned upstream commits are in
[`THIRD_PARTY_SOURCES.md`](THIRD_PARTY_SOURCES.md) and
[`docs/ASSEMBLY_GUIDE_ZH.md`](docs/ASSEMBLY_GUIDE_ZH.md).

The CNN/LSTM code in the example repositories was not reused. Their random
beat-wise splits were also not reused because they can place beats from the
same patient in both training and test sets.

## Limitations

- The primary task is binary `N` versus `V`, not full AAMI five-class or
  clinical diagnosis.
- Reference R peaks are used; automatic detector robustness is not evaluated.
- DS2 is a single held-out test group, not multi-dataset external validation.
- Logistic regression is linear; it is selected because its inner product is
  practical for a first CKKS experiment.
- CKKS encryption, security estimation, key generation, ciphertext sizes, and
  encrypted latency are outside this plaintext-only release.
- This is a research prototype, not a medical device.

## Planned CKKS extension

The next stage should keep the plaintext pipeline unchanged and add a separate
encrypted inference module:

```text
client: ECG -> DWT -> scaler -> PCA -> z
client: CKKS.Encode(z) -> Encrypt(z)
server: MultiplyPlain(encrypted z, weight)
server: rotate-and-add slots for the inner product
server: add plaintext bias
client: Decrypt(logit) -> sign decision
```

The first encrypted experiment should report plaintext/decrypted logit MAE and
RMSE, label agreement, sign-flip rate near the boundary, key generation time,
encoding and encryption time, server evaluation latency, decryption time, and
ciphertext/key sizes.

Do not report a communication reduction from PCA unless the implementation
actually uses multiple ciphertexts or a batch-packing strategy where feature
dimension changes the number of transmitted ciphertexts.

## Citation

If this code contributes to a paper, cite the MIT-BIH database, the CKKS
scheme, and the upstream software listed in `THIRD_PARTY_SOURCES.md`. A concise
description of this baseline is:

> We evaluate a patient-independent N-versus-V ECG classifier on MIT-BIH. Each
> annotated beat is filtered, normalized, represented by level-4 db4 wavelet
> coefficients, standardized, and projected from 256 to 32 dimensions by PCA.
> A class-weighted logistic regression produces a plaintext logit that is
> exported as a CKKS-compatible inner-product interface.

## License and data notice

The original code in this repository is released under the MIT License in
[`LICENSE`](LICENSE). Third-party packages remain under their own licenses.
The MIT-BIH data are provided by PhysioNet under the database's terms and are
not redistributed by this repository.
