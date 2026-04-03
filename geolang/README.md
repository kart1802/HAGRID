# Geolang + VMamba Setup Guide

This guide explains how to set up the environment for Geolang with VMamba.

---

## 1. Create Conda Environment

```bash
conda create -n vmamba_geolang python=3.10
conda activate vmamba_geolang
```

---

## 2. Install VMamba (Core Dependency)

Follow the official VMamba installation instructions:

👉 https://github.com/MzeroMiko/VMamba?tab=readme-ov-file#getting-started

**Important:**
- Install VMamba with **GPU support enabled**.
- This is required because VMamba depends on **`selective-scan`**, which requires CUDA.
- A CPU-only installation will not work correctly for this project.

This step will install:
- PyTorch
- Triton
- Mamba (`mamba_ssm`)
- `selective-scan` and related CUDA dependencies

---

## 3. Install Additional Dependencies

Install the remaining Geolang dependencies using:

```bash
conda env update -f environment.yml
```

---

## 4. Training

To train Geolang, run:

```bash
python train_geolang.py --config config/OCID-VLG/geolang_main_config.yaml
```

### Distributed GPU Training with SLURM

To train on distributed GPUs, use the provided SLURM script:

```bash
sbatch train_geolang.slurm
```

Make sure the paths inside `train_geolang.slurm` are updated appropriately for your system, including:
- the project root directory
- the conda environment activation path
- the log/output directory
- any dataset or checkpoint paths if required

If needed, update the training command inside the SLURM file so that it points to the correct config file, for example:

```bash
python train_geolang.py --config config/OCID-VLG/geolang_main_config.yaml
```

---

## 5. Testing

```bash
python test_geolang.py --config config/OCID-VLG/geolang_main_config.yaml
```

---

## 6. Ablation Experiments

### ADCI Only

Set in config:

```yaml
use_adci: True
use_dggm: False
```

Run:

```bash
python test_geolang.py --config config/OCID-VLG/geolang_adci_config.yaml
```

---

### DGGM Only

Set in config:

```yaml
use_adci: False
use_dggm: True
normalize_after_reweight: True
```

Run:

```bash
python test_geolang.py --config config/OCID-VLG/geoland_dggm_config.yaml
```

---

### Full Model

Set in config:

```yaml
use_adci: True
use_dggm: True
normalize_after_reweight: False
```

Run:

```bash
python test_geolang.py --config config/OCID-VLG/geolang_main_config.yaml
```

---

## 7. Visualization

Enable in config:

```yaml
visualize: True
```

Then run:

```bash
python test_geolang.py --config config/OCID-VLG/geolang_main_config.yaml
```

---

## Notes

- Install VMamba first to ensure correct PyTorch and CUDA compatibility.
- Always update the `resume` path in the config before testing.
- Use the appropriate config files for ablation and full-model experiments.
- Ensure GPU and CUDA are properly configured before installing VMamba.
- For cluster training, verify all paths in `train_geolang.slurm` before submission.