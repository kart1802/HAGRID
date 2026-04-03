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
- CPU-only installation will not work correctly for this project.

This step will install:
- PyTorch (compatible version)
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

- Install VMamba first to ensure correct PyTorch + CUDA compatibility.
- Do not reinstall torch or triton after VMamba setup.
- Always update the `resume` path in config before testing.
- Use appropriate config files for ablation vs full model.
- Ensure GPU/CUDA is properly configured before installing VMamba.