# GeoLanG Reproduction Study

This project focuses on the **reproduction of the GeoLanG paper** for robotic grasping as part of the **Computer Vision Course (DSAIT4125)**.

---

## Overview

The goal of this project is to reproduce and analyze the methodology proposed in:

**GeoLanG: Geometry-Aware Language-Guided Grasping with Unified RGB-D Multimodal Learning**

GeoLanG addresses the problem of **language-guided robotic grasping**, where a robot must identify and grasp objects based on natural language instructions.

The key idea is to combine:
- **Visual features (RGB)**
- **Geometric information (Depth)**
- **Language understanding**

to predict **task-relevant grasp regions**.

This enables robots to perform **context-aware and task-specific grasping**, rather than generic object picking. :contentReference[oaicite:0]{index=0}

---

## Original Paper

**Title:** GeoLanG: Geometry-Aware Language-Guided Grasping with Unified RGB-D Multimodal Learning  
**Authors:** Rui Tang et al.  

Paper link:  
https://arxiv.org/pdf/2602.04231

---

## Project Objective

- Reproduce the GeoLanG pipeline
- Understand multimodal fusion (RGB + Depth + Language)
- Analyze grasp prediction performance
- Evaluate model behavior under different task conditions

---

## Contributors

**Group 18**

- Tejas Stanley  *
- Karthik Swaminathan *
- Yash Surange *
- Stan van de Berk *
* (equal contribution)

---

## Scope of Work

- Implementation and setup of GeoLanG framework
- Dataset preparation and preprocessing
- Training and evaluation
- Ablation analysis (DGGM, ADCI components)
- Visualization of grasp predictions

---

## Notes

- This project is a **reproduction study**, not a novel method.
- Any modifications or improvements are experimental and not part of the original paper.
