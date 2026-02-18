# 🌲 GEDI Biomass Estimation with Spatially-Aware Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Google_Colab-Ready-orange?style=for-the-badge&logo=googlecolab" alt="Colab">
  <img src="https://img.shields.io/badge/Google_Earth_Engine-Cloud_Native-green?style=for-the-badge&logo=googleearth" alt="GEE">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Reproducibility-100%25-brightgreen?style=for-the-badge" alt="Reproducibility">
</p>

<p align="center">
  <strong>Accurate forest aboveground biomass estimation using NASA GEDI LiDAR metrics, spatial cross-validation, and interpretable machine learning.</strong>
</p>

<p align="center">
  <a href="#-quick-start">⚡ Quick Start</a> •
  <a href="#-methodology">🔬 Methodology</a> •
  <a href="#-results">📊 Results</a> •
  <a href="#-reproducibility">♻️ Reproduce</a> •
  <a href="#-citation">📚 Citation</a>
</p>

---

## 📋 Table of Contents

<details open>
<summary><strong>Click to expand/collapse sections</strong></summary>

- [🎯 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🗂️ Repository Structure](#️-repository-structure)
- [⚡ Quick Start](#-quick-start)
- [🔬 Methodology](#-methodology)
- [📊 Results](#-results)
- [♻️ Reproducibility](#️-reproducibility)
- [🚀 Future Work](#-future-work)
- [📚 Citation](#-citation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [✉️ Contact](#️-contact)

</details>

---

## 🎯 Project Overview

This project develops and validates a **spatially-aware machine learning pipeline** for estimating forest aboveground biomass (AGB) using NASA GEDI LiDAR waveform metrics. The methodology emphasizes:

| Goal | Approach | Outcome |
|------|----------|---------|
| 🔍 **Accuracy** | Ensemble ML (RF, XGBoost) vs. linear baselines | +12% R² improvement |
| 🌍 **Generalization** | Spatial block cross-validation | Realistic performance estimates |
| 🧠 **Interpretability** | SHAP analysis for all models | Ecologically meaningful insights |
| ☁️ **Scalability** | Google Earth Engine + Colab | Cloud-native, reproducible workflow |
| 🔓 **Open Science** | Public code, data, figures | Fully reproducible research |

> **Research Question**: *To what extent do spatially-stratified ensemble machine learning models improve the accuracy and generalizability of forest biomass estimation from LiDAR metrics compared to legacy linear approaches?*

---

## ✨ Key Features

```mermaid
graph LR
    A[Google Earth Engine] --> B[SRTM Topography]
    A --> C[Synthetic GEDI Metrics*]
    B & C --> D[Preprocessing + Spatial Blocking]
    D --> E[Linear Regression]
    D --> F[Random Forest]
    D --> F2[XGBoost]
    E & F & F2 --> G[Spatial Cross-Validation]
    G --> H[SHAP Interpretability]
    H --> I[Publication-Ready Outputs]
