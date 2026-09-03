# Third-party sources and adaptation notes

The `references/` directory contains local source archives downloaded on
2026-09-03. It is intentionally excluded from Git. This repository imports the
official packages as dependencies; it does not vendor their source code.

| Repository | Commit | License | How it was used |
|---|---|---|---|
| [MIT-LCP/wfdb-python](https://github.com/MIT-LCP/wfdb-python) | `f627b5ff9dcfdb11d4c3150f9c1ebb47cfc3909d` | MIT | Public `rdrecord`, `rdann`, and `dl_database` APIs |
| [PyWavelets/pywt](https://github.com/PyWavelets/pywt) | `c55a14ecd062c19e4405ae44f0afb82f29d3b7e8` | MIT | Public `wavedec` API with db4 and periodization |
| [neuropsychology/NeuroKit](https://github.com/neuropsychology/NeuroKit) | `ff419d983568ef492eb8d229af643c0ef0100b32` | MIT | Optional reference for future automatic R-peak experiments; no code copied |
| [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | `757f5375119181b70ed2b60b7ea08a2b395def32` | BSD-3-Clause | Public scaler, PCA, logistic regression, and metric APIs |
| [antimattercorrade/Pan_Tompkins_QRS_Detection](https://github.com/antimattercorrade/Pan_Tompkins_QRS_Detection) | `5a76ed7b82b1bca8a40f58dc4125ab81cbfa95f1` | No license declared | Algorithm reference only; no source copied |
| [SergeyFilipov/ecg-arrhythmia-detection](https://github.com/SergeyFilipov/ecg-arrhythmia-detection) | `a689161dc89079811c85bd7fd9fd6ac5fb0aaa54` | MIT | WFDB/wavelet workflow reference; CNN and beat-wise random split not reused |
| [Tomoki-K-0409/ECG-LSTM-Reproduction](https://github.com/Tomoki-K-0409/ECG-LSTM-Reproduction) | `6cfb7408fd28671cfb6d9c89b15a30534c006ada` | MIT | db4 level-4 feature reference; LSTM code not reused |

## Material changes

1. Replaced beat-wise random train/test splitting with fixed, disjoint MIT-BIH
   patient record groups (DS1 training and DS2 testing).
2. Replaced CNN/LSTM classifiers with class-weighted logistic regression.
3. Added training-only standardization and PCA for a CKKS-friendly 32-value
   client feature.
4. Added exact manual-logit verification and a portable NumPy model export.
5. Added configuration capture, predictions, paper metrics, plots, and tests.
