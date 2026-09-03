# 开源代码组装与修改说明

本文档说明本项目如何把多个开源项目中可复用的部分组成适合论文的明文流程。

## 总体原则

上游科学计算库通过 Python 包接口调用，不复制或修改其内部实现。ECG 示例项目仅用于理解数据格式和算法流程。本项目重新实现实验组织代码，以保证：

- DS1 训练记录和 DS2 测试记录互不重叠；
- 所有标准化和 PCA 参数仅由训练集拟合；
- 输出固定维数的 PCA 特征和线性模型参数；
- 后续 CKKS 代码可以直接复算 `w^T z + b`；
- 每次实验记录配置、软件版本、预测和指标。

## 模块对应关系

### 1. WFDB 数据读取

来源：`MIT-LCP/wfdb-python`

使用接口：

```python
record = wfdb.rdrecord(record_path)
annotation = wfdb.rdann(record_path, "atr")
```

本项目位置：`src/pcghe_ecg/dataset.py`

修改内容：不再读取单条记录后随机拆分心拍，而是先按固定 DS1/DS2 记录表组成训练和测试集。当前主配置预先固定，不使用测试集调参；如需调参，应在 DS1 内按记录做交叉验证。

### 2. ECG 滤波和心拍切分

来源：MIT-BIH 标注格式和 Pan-Tompkins 处理流程。

本项目主实验直接使用 `atr` 中的 R 峰标注，不复制无许可证的 Pan-Tompkins Notebook。滤波由 SciPy 的四阶零相位 Butterworth 带通滤波完成。

本项目位置：

```text
src/pcghe_ecg/preprocess.py
src/pcghe_ecg/dataset.py
```

每个心拍截取 R 峰前 90 点和后 166 点，恰好得到 256 点，不需要尾部补零。

### 3. db4 小波特征

来源：`PyWavelets/pywt`，并参考两个 ECG 示例项目的 db4 调用方式。

使用接口：

```python
coeffs = pywt.wavedec(
    beat,
    wavelet="db4",
    level=4,
    mode="periodization",
)
feature = np.concatenate(coeffs)
```

本项目位置：`src/pcghe_ecg/features.py`

修改内容：显式使用 `periodization`，并检查 256 点输入必须产生 256 个小波系数。CNN/LSTM 示例中的截断和网络输入逻辑全部删除。

### 4. 标准化和 PCA

来源：`scikit-learn/scikit-learn`

使用接口：

```python
scaler.fit(X_train)
pca.fit(scaler.transform(X_train))
```

测试集仅调用 `transform`。主实验输出 32 维特征，消融实验使用 8、16、32、64、128、256 维。

本项目位置：`src/pcghe_ecg/model.py`

### 5. 逻辑回归

来源：scikit-learn 的 `LogisticRegression`。

替换内容：移除上游示例中的 CNN、LSTM、TensorFlow 和 PyTorch 训练过程，改成带类别权重的逻辑回归：

```python
LogisticRegression(
    class_weight="balanced",
    solver="lbfgs",
    max_iter=2000,
)
```

模型输出 logit，不需要 sigmoid。默认判定规则为 `logit > 0`。

### 6. CKKS 接口

`results/.../artifacts/ckks_interface.npz` 保存：

```text
scaler_mean
scaler_scale
pca_mean
pca_components
weight
bias
threshold
```

客户端明文处理到 PCA 特征 `z` 后加密；服务器读取 `weight` 和 `bias`，计算加密内积；客户端解密后判断符号。

## 明确没有复用的部分

- `ecg-arrhythmia-detection` 的 CNN 与心拍级随机拆分；
- `ECG-LSTM-Reproduction` 的 LSTM 和训练循环；
- 无许可证 Pan-Tompkins 仓库的具体源代码；
- NeuroKit 的自动 R 峰检测，主实验暂不需要；
- 上游仓库自带的模型文件和数据文件。

## 本地目录与 GitHub 内容

桌面 `PCGHE/references/` 保存上游源码归档及解压目录，方便人工核对。该目录被 `.gitignore` 排除。GitHub 仓库只包含本项目源代码、测试、许可证、来源说明和文档。
