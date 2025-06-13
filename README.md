# Master Thesis: INR Model Comparison

This repository contains scripts for comparing neural field models (SIREN, HashSIREN, HashReLU) on 2D image datasets using various loss functions and metrics.

## 📂 Key Files

- `compare_novel_learning.py` — Main script for training and comparing models with multiple loss functions.
- `train.py` — Simplified script that only measures for one loss (normal_loss) and uses single lists for metrics.
- `networks.py` — Model definitions (uses [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) Python bindings).
- `utils.py` — Utility functions for training, metrics, and visualization.

## 🚀 Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Tordsg/master_thesis.git
cd master_thesis
```

### 2. Create and activate the Conda environment

```bash
conda env create -f environment.yml
conda activate master-thesis
```

### 3. Install or Compile tiny-cuda-nn

#### Option A: Install via pip (if available)

```bash
pip install tinycudann
```

#### Option B: Build from source

```bash
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git
cd tiny-cuda-nn
cmake . -B build
cmake --build build --config RelWithDebInfo -j
cd bindings/pybind11
pip install .
cd ../../..
```

If you build from source, make sure the Python bindings are discoverable (add to `PYTHONPATH` if needed).

### 4. Prepare Data

Place your input image (e.g., `tokyo_crop.jpg`) in the repository root or update the path in the scripts.

### 5. Run the Comparison Script

```bash
python compare_novel_learning.py
```

This will train and compare the models, saving results and plots in the `novel/` directory.

### 6. Run the Simplified Training Script

```bash
python train.py
```

This script only measures for one loss (normal_loss) and uses single lists for metrics, saving results in the `train/` directory.

## 📝 Notes

- The scripts require a CUDA-capable GPU.
- You can adjust hyperparameters, models, and images at the top of the scripts.
- Results (metrics, images, plots) are saved in subfolders of `novel/` and `train/`.

## 📧 Contact

For questions or contributions, open an issue or contact [@Tordsg](https://github.com/Tordsg).

## 📚 References

- Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., & Wetzstein, G. (2020). "Implicit Neural Representations with Periodic Activation Functions (SIREN)." *NeurIPS 2020*. [arXiv:2006.09661](https://arxiv.org/abs/2006.09661)
- Müller, T., Evans, A., Schied, C., & Keller, A. (2022). "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding." *SIGGRAPH 2022*. [arXiv:2201.05989](https://arxiv.org/abs/2201.05989)  
  (HashReLU and HashSIREN are based on this hash encoding framework)
- tiny-cuda-nn: [https://github.com/NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) 