# Model Weights

The trained model weights are **not stored in this repository** due to file size constraints.

## Download Models

The trained `.keras` model files were trained on Kaggle using the `final_draft.ipynb` notebook.

### Option A: Re-train on Kaggle
1. Upload `final_draft.ipynb` to [Kaggle](https://www.kaggle.com/)
2. Run all cells with the HAM10000 dataset
3. Download the output models and place them in this `models/` folder

### Option B: Manual Download
If the author has shared the models, download and place these files here:

```
models/
├── gated_multimodal_fusion.keras       ← M4 (main model, recommended)
├── multimodal_fusion_baseline.keras    ← M4 fallback / older checkpoint
├── simple_concat_fusion.keras          ← M3
├── image_only_model.keras              ← M2
├── metadata_only_model.keras           ← M1
└── metadata_preprocessor.joblib       ← Required: tabular feature encoder (already in repo)
```

## Required File for Inference

`metadata_preprocessor.joblib` is already committed to the repo (it's small, ~4 KB).
The `.keras` model files must be downloaded separately.
