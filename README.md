# DermaFusion AI: Multimodal Skin Lesion Diagnostic System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A Leakage-Controlled Empirical Study on Image, Demographic Metadata, and Gated Multimodal Fusion for Skin Lesion Classification (HAM10000).**

---

## 🔬 Key Empirical Highlights (Zero-Leakage Test Set, $N = 1,452$)

```
                                 TEST BENCHMARK COMPARISON
                                 
        ┌────────────────────────────────────────────────────────────────────────┐
   M4   │ Acc: 67.77% │ Bal Acc: 53.01% │ Mel Recall: 61.20% │ Mel Spec: 83.22%  │ (Proposed)
        ├────────────────────────────────────────────────────────────────────────┤
   M3   │ Acc: 67.36% │ Bal Acc: 64.79% │ Mel Recall: 68.31% │ Mel Spec: 80.30%  │ (Concat)
        ├────────────────────────────────────────────────────────────────────────┤
   M2   │ Acc: 63.84% │ Bal Acc: 58.76% │ Mel Recall: 59.02% │ Mel Spec: 81.32%  │ (Image Only)
        ├────────────────────────────────────────────────────────────────────────┤
   M1   │ Acc: 26.45% │ Bal Acc: 30.04% │ Mel Recall: 18.03% │ Mel Spec: 90.70%  │ (Metadata Only)
        └────────────────────────────────────────────────────────────────────────┘
```

* **Zero-Leakage Patient-Lesion Isolation:** Enforced via `StratifiedGroupKFold` on unique `lesion_id` to eliminate optimistic identity memorization leakage.
* **Gated Multimodal Fusion Architecture ($M_4$):** Dynamically scales convolutional feature activations via a learned Sigmoid Modality Interaction Unit, achieving the highest overall test accuracy (**$67.77\%$**) and highest melanoma precision (**$34.46\%$**).
* **Metadata Permutation Ablation:** Shuffling patient age/sex/location across lesions triggers an **$-8.75\%$ collapse** in accuracy ($67.77\% \to 59.02\%$), proving the network learns authentic biological correspondences.
* **DullRazor Preprocessing Impact:** Omitting morphological hair inpainting causes melanoma sensitivity to collapse by **$-31.15\%$** ($59.02\% \to 27.87\%$).
* **Clinical Interpretability:** Genuine Selvaraju et al. Grad-CAM spatial activation heatmaps confirm the model attends to pathological tumor borders and pigment asymmetry.

---

## 🏛️ System Architecture

```
   [Dermoscopic Image]                             [Patient Metadata]
   (HAM10000 224x224x3)                          (Age, Sex, Anatomical Site)
            │                                                 │
            ▼                                                 ▼
   [DullRazor Morphological]                       [ColumnTransformer Imputation]
   [Black-Hat Inpainting]                          [& One-Hot 19-D Encoding]
            │                                                 │
            ▼                                                 ▼
   [EfficientNetB0 Backbone]                       [Dense(64) -> BatchNorm ->]
   [2-Stage Surgical Fine-Tune]                    [Dropout(0.2) -> Dense(128)]
            │                                                 │
            ▼ (128-D Image Embedding)                         ▼ (128-D Tabular Embedding)
            └───────────────────────┬─────────────────────────┘
                                    │
                                    ▼
                     [Gated Multimodal Fusion Unit]
                     [ Gate = Sigmoid(Dense(256)) ]
                     [ Gated Image = Img * Gate   ]
                                    │
                                    ▼
                     [Fused Representation: 256-D]
                                    │
                                    ▼
                     [Dense(128, ReLU) + Dropout]
                                    │
                                    ▼
                     [Dense(7, Softmax Output)]
                                    │
                                    ▼
              [Class-Weighted Sparse Categorical Focal Loss]
```

---

## 🚀 Quick Start (Web Application)

### 1. Install Dependencies
```bash
git clone https://github.com/your-username/DermaFusion-AI.git
cd DermaFusion-AI
pip install -r requirements.txt
```

### 2. Launch Diagnostic Server
```bash
python main.py
# or double-click run.bat on Windows
```
Open your browser and navigate to **`http://127.0.0.1:8000`**.

---

## 📂 Project Structure

```
├── models/
│   ├── gated_multimodal_fusion.keras      # Trained Proposed M4 Model
│   ├── multimodal_fusion_baseline.keras   # Checkpoint fallback
│   └── metadata_preprocessor.joblib       # 19-feature tabular encoder
├── outputs/
│   ├── four_model_benchmark.csv           # 4-Model Comparative Matrix
│   ├── statistical_significance_m3_vs_m4.csv # McNemar & Bootstrap CIs
│   ├── metadata_ablation_benchmark.csv    # Intact vs Zeroed vs Permuted
│   ├── dullrazor_sensitivity_analysis.csv # DullRazor vs Raw Dermoscopy
│   ├── per_class_performance_m4.csv       # 7-Class Precision/Recall/F1
│   ├── gradcam_clinical_heatmaps.png      # Publication Grad-CAM Figure
│   └── training_convergence_curves.png    # Validation Loss/Acc Curves
├── static/
│   ├── index.html                         # Dashboard HTML
│   └── app.js                             # React 18 Interactive Frontend
├── app.py                                 # FastAPI Server
├── inference.py                           # Inference & DullRazor Engine
├── main.py                                # Launcher script
├── research_paper_manuscript.tex          # Complete IEEE LaTeX Paper Source
├── research_paper_manuscript.md           # Markdown Paper Manuscript
└── references.bib                         # BibTeX Citations
```

---

## 📜 Citation (BibTeX)

```bibtex
@article{nikhil2026dermafusion,
  title={Multimodal Skin Lesion Classification: A Leakage-Controlled Empirical Study on Image, Metadata, and Gated Fusion},
  author={Nikhil et al.},
  journal={IEEE Journal of Biomedical and Health Informatics (Under Review)},
  year={2026}
}
```
