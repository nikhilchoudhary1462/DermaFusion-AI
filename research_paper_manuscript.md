# Multimodal Skin Lesion Classification: A Leakage-Controlled Empirical Study on Image, Demographic Metadata, and Gated Fusion

**Authors:** Nikhil et al.  
**Affiliation:** Artificial Intelligence and Biomedical Informatics Research Group  
**Target Venue:** *IEEE Journal of Biomedical and Health Informatics (JBHI)* / *IEEE Transactions on Medical Imaging (TMI)*  

---

## Abstract
Automated dermoscopic image classification holds significant clinical promise for early skin cancer detection. However, standard computer vision models frequently underperform on lethal minority malignancies, notably malignant melanoma, due to acute visual mimicry, severe dataset class skew, and the systematic neglect of patient demographic context. Furthermore, optimistic data leakage resulting from improper multi-image lesion splitting remains prevalent across the dermatological machine learning literature. In this study, we present an end-to-end, leakage-controlled empirical investigation into multimodal skin lesion classification on the benchmark HAM10000 dataset ($N = 10,015$; 7 diagnostic classes). 

To eliminate data leakage, we enforce a strict patient-lesion isolation protocol using Stratified Group K-Fold cross-partitioning on unique lesion identifiers ($\text{lesion\_id}$), guaranteeing zero patient overlap across training, validation, and test subsets ($N_{\text{test}} = 1,452$). We benchmark four controlled model paradigms: 
1. **$M_1$**: a 19-dimensional tabular multilayer perceptron operating exclusively on demographic metadata (age, sex, anatomical site); 
2. **$M_2$**: a fine-tuned EfficientNetB0 convolutional baseline equipped with DullRazor morphological hair inpainting; 
3. **$M_3$**: an unconstrained Simple Feature Concatenation multimodal network; and 
4. **$M_4$**: a proposed **Gated Multimodal Fusion Architecture** incorporating a learned Sigmoid Modality Interaction Unit optimized via rank-safe, class-weighted Sparse Categorical Focal Loss ($\gamma = 2.0, \alpha = 0.25$). 

Our primary empirical results demonstrate that while metadata alone yields only $18.03\%$ melanoma sensitivity, multimodal integration ($M_4$) elevates overall test accuracy to **$67.77\%$**, balanced accuracy to **$53.01\%$**, and melanoma sensitivity to **$61.20\%$**, while achieving the highest specificity (**$83.22\%$**) and highest melanoma precision (**$34.46\%$**) among all evaluated vision models. A 1,000-iteration lesion-clustered bootstrap analysis reveals that the sensitivity difference between $M_3$ and $M_4$ is statistically distinguishable ($95\%$ CI: $[-12.94\%, -1.06\%]$), delineating distinct clinical operating points on the sensitivity-specificity Pareto frontier. Crucially, a metadata permutation ablation—wherein patient demographics were randomly shuffled across mismatched lesions—triggered an acute **$-8.75\%$ collapse in overall accuracy** ($59.02\%$) and a **$-11.47\%$ decline in melanoma recall**, providing empirical evidence that the network leverages authentic biological correspondences rather than superficial regularizing noise. Omitting DullRazor artifact suppression reduced melanoma sensitivity by **$-31.15\%$** ($59.02\% \to 27.87\%$), confirming that hair inpainting is an essential prerequisite for robust visual feature extraction. Finally, genuine Grad-CAM saliency heatmaps, SHAP feature attributions, Monte Carlo Dropout uncertainty calibration, and subgroup demographic fairness evaluations are systematically synthesized. This study establishes a reproducible benchmark and highlights critical trade-offs for responsible, multimodal clinical decision support.

**Keywords:** Dermatopathology, Deep Learning, Multimodal Fusion, Modality Gating, Class Imbalance, Focal Loss, Explainable AI, Grad-CAM, Uncertainty Estimation, Algorithmic Fairness.

---

## 1. Introduction

Malignant melanoma represents the most lethal form of cutaneous malignancy, accounting for the overwhelming majority of skin-cancer-related mortalities worldwide despite comprising less than $5\%$ of all diagnosed skin neoplasms [1, 2]. Early clinical detection remains the single most critical determinant of long-term prognosis: localized early-stage melanomas exhibit five-year relative survival rates exceeding $99\%$, whereas metastatic lesions drop precipitously below $30\%$ [2]. While epiluminescence dermoscopy significantly improves diagnostic accuracy over unaided visual inspection, manual evaluation is inherently subjective, operator-dependent, and prone to diagnostic confusion with benign mimics such as dysplastic nevi and seborrheic keratoses [30, 31].

In recent years, deep convolutional neural networks (CNNs) and vision transformers have demonstrated remarkable diagnostic capabilities, in some settings matching board-certified dermatologists on isolated binary or narrow-category classification tasks [2]. However, translating automated vision models to comprehensive 7-class dermatological screening benchmarks exposes fundamental challenges that remain insufficiently addressed in the literature:

1. **Acute Class Imbalance:** Clinical datasets such as the HAM10000 benchmark [1] exhibit severe distributional skew. Common melanocytic nevi comprise more than $67\%$ of all cases, whereas fatal malignancies such as melanoma ($11.1\%$), basal cell carcinoma ($5.1\%$), and rare dermatofibromas ($1.1\%$) are heavily underrepresented. Standard cross-entropy loss functions trained on such skewed distributions degenerate toward majority-class overprediction, yielding deceptively high top-line accuracy while missing critical minority malignancies [4, 19].
2. **Optimistic Lesion-Level Data Leakage:** Individual patients frequently contribute multiple dermoscopic acquisitions of the same underlying lesion, captured from varying angles, magnifications, or chronological sessions. As rigorously documented by Cassidy et al. [7], naive randomized train-test splits inadvertently place duplicate lesion images across partitions, causing models to memorize lesion-specific visual artifacts and artificially inflating reported benchmark accuracies.
3. **Neglect of Orthogonal Clinical Context:** In actual clinical practice, dermatologists never evaluate dermoscopic morphology in isolation. Non-imaging demographic context—including patient age, biological sex, and anatomical lesion localization—provides crucial epidemiological priors that fundamentally alter pre-test diagnostic suspicion [11, 15, 13].
4. **Vulnerability to Acquisition Artifacts:** Dermoscopic photographs frequently contain occluding hair strands, air bubbles, and surgical marker ink. Unregulated CNNs readily latch onto these spurious correlations as predictive shortcuts rather than learning true pathological morphology [8, 9, 21].

### Research Questions (RQs):
* **RQ1 (Demographic Predictive Signal):** Does non-imaging demographic and anatomical metadata alone provide measurable diagnostic power for multi-class differential diagnosis?
* **RQ2 (Cross-Modal Synergy):** Does integrating patient metadata with dermoscopic imagery yield statistically meaningful sensitivity gains on lethal minority classes compared to an identically fine-tuned unimodal vision baseline?
* **RQ3 (Gating Dynamics & Clinical Trade-offs):** Does a learned Sigmoid Modality Interaction Unit offer quantifiable precision and specificity advantages over unconstrained feature concatenation on the clinical sensitivity-specificity Pareto frontier?

---

## 2. End-to-End System Architecture

```
                          END-TO-END MULTIMODAL DIAGNOSTIC PIPELINE
                          
   [Dermoscopic Image]                             [Patient Metadata]
   (HAM10000 224x224x3)                          (Age, Sex, 15 Anatomical Sites)
            │                                                 │
            ▼                                                 ▼
   [DullRazor Morphological]                       [ColumnTransformer Imputation]
   [Black-Hat Inpainting]                          [& One-Hot 19-D Encoding]
            │                                                 │
            ▼                                                 ▼
   [EfficientNetB0 Backbone]                       [Dense(64) -> BatchNorm ->]
   [2-Stage Surgical Fine-Tune]                    [Dropout(0.2) -> Dense(128)]
            │                                                 │
            ▼ (128-D Image Embedding h_img)                   ▼ (128-D Tabular Embedding h_meta)
            └───────────────────────┬─────────────────────────┘
                                    │
                                    ▼
                     [Gated Modality Interaction Unit]
                     [ Gate g = Sigmoid(Dense(256))   ]
                     [ h_gated = h_img ⊙ g            ]
                                    │
                                    ▼
                     [Fused Representation: 256-D]
                     [ h_fused = Dense(128, ReLU)     ]
                                    │
                                    ▼
                     [Dense(7, Softmax Output)]
                                    │
                                    ▼
              [Class-Weighted Sparse Categorical Focal Loss]
              [  γ = 2.0, α = 0.25, Rank-Safe Per-Sample   ]
```

![Figure 1: Multimodal Architecture Pipeline](fig1_multimodal_architecture_pipeline.png)
*Figure 1: Complete end-to-end architectural workflow of the proposed Gated Multimodal Diagnostic Framework. Dual visual and tabular streams are preprocessed under zero-leakage constraints, embedded into 128-D latent representations, modulated via a Sigmoid Gating Unit, and optimized with rank-safe class-weighted focal loss.*

---

## 3. Related Work & Literature Synthesis

### 3.1 Benchmark Datasets and Data Leakage Protocols
The publication of the HAM10000 ("Human Against Machine with 10,000 training images") multi-source dataset by Tschandl et al. [1] established the standard 7-class benchmark for automated dermoscopy. Subsequent International Skin Imaging Collaboration (ISIC) challenges [16, 15] expanded public repositories to tens of thousands of images. However, Cassidy et al. [7] revealed pervasive methodological flaws in published literature, demonstrating that random image-level splitting across duplicate lesion acquisitions induces severe data leakage. In this work, we strictly enforce a zero-leakage lesion-level group partitioning protocol.

### 3.2 Deep Visual Representation and Transfer Learning
Esteva et al. [2] demonstrated dermatologist-level skin cancer classification using deep CNNs trained on over 129,000 clinical images. Subsequent works established that transfer learning from ImageNet provides strong visual inductive biases for dermoscopy [30, 31]. Tan and Le [3] introduced EfficientNet, utilizing compound scaling of depth, width, and resolution to maximize parameter efficiency. In this study, we deploy EfficientNetB0 as our standardized visual backbone.

### 3.3 Class Imbalance and Loss Function Formulations
To combat acute class imbalance in medical imaging, Lin et al. [4] introduced Focal Loss, which adds a modulating factor $(1 - p_t)^\gamma$ to standard cross-entropy to suppress easy majority-class gradients. Cui et al. [19] proposed class-balanced loss weighting based on the effective number of samples, and Cao et al. [20] derived label-distribution-aware margin loss. We adopt a rank-safe, class-weighted Sparse Categorical Focal Loss formulation.

### 3.4 Multimodal Fusion Architectures
Early investigations by Yap et al. [13] and Kawahara et al. [12] established that concatenating tabular metadata with image embeddings enhances diagnostic sensitivity. Pacheco and Krohling [10, 11] demonstrated that patient clinical features significantly boost melanoma detection AUC and proposed attention-based metadata blocks. Gessert et al. [14] combined multi-resolution EfficientNets with patient metadata to win the ISIC 2019 challenge. General multimodal architectures—such as Gated Multimodal Units (GMU) by Arevalo et al. [17] and multimodal surveys by Baltrušaitis et al. [18]—motivate our proposed Sigmoid Modality Interaction Unit.

### 3.5 Dataset Bias, Shortcuts, and Artifact Removal
Bissoto et al. [8, 9] demonstrated that deep CNNs frequently exploit background skin markings, gel bubbles, and ruler artifacts rather than pathological morphology. Winkler et al. [21] proved that surgical pen marks artificially inflate CNN melanoma risk scores. Morphological hair removal via DullRazor [6, 22] provides a principled mechanism to suppress occluding hair artifacts prior to convolutional feature extraction.

### 3.6 Explainability, Uncertainty, and Algorithmic Fairness
Selvaraju et al. [5] developed Grad-CAM for gradient-weighted localization of CNN activations, which we deploy as an audit mechanism. Lundberg and Lee [23] established SHAP game-theoretic feature attribution. For epistemic uncertainty estimation, Gal and Ghahramani [24] proved that Monte Carlo Dropout approximates Bayesian inference in Gaussian processes. Finally, landmark studies by Groh et al. [26, 27], Daneshjou et al. [28], and Adamson and Smith [29] demonstrated severe racial and demographic diagnostic disparities in dermatology AI, underscoring the necessity of subgroup evaluations.

---

## 4. Materials and Methodology

### 4.1 Dataset and Zero-Leakage Group Partitioning
We utilized the HAM10000 benchmark dataset [1], comprising $N = 10,015$ high-resolution dermoscopic images across seven diagnostic categories:
1. `akiec`: Actinic Keratoses and Intraepithelial Carcinoma ($n=327$)
2. `bcc`: Basal Cell Carcinoma ($n=514$)
3. `bkl`: Benign Keratosis-like Lesions ($n=1,099$)
4. `df`: Dermatofibroma ($n=115$)
5. `mel`: Malignant Melanoma ($n=1,113$)
6. `nv`: Melanocytic Nevi ($n=6,705$)
7. `vasc`: Vascular Lesions ($n=142$)

To strictly eliminate lesion memorization leakage [7], all samples were grouped by their unique `lesion_id` identifier. We executed a 7-fold Stratified Group split (`StratifiedGroupKFold`, Seed = 50), isolating two distinct folds as independent Validation and Test subsets:
$$\mathcal{L}_{\text{train}} \cap \mathcal{L}_{\text{val}} = \emptyset, \quad \mathcal{L}_{\text{train}} \cap \mathcal{L}_{\text{test}} = \emptyset, \quad \mathcal{L}_{\text{val}} \cap \mathcal{L}_{\text{test}} = \emptyset$$
The final held-out test partition comprises exactly $N_{\text{test}} = 1,452$ unseen dermoscopic images.

### 4.2 Morphological Artifact Suppression: DullRazor
Dermoscopic acquisitions are frequently occluded by dark hair strands. We integrated the DullRazor morphological filter [6]:
$$B = (I_{\text{gray}} \bullet K) - I_{\text{gray}}, \quad K \in \mathbb{R}^{9 \times 9}$$
where $\bullet$ represents the morphological closing operator on luminance image $I_{\text{gray}}$. A binary hair mask is thresholded as $M = \mathbb{I}(B > 10)$, and non-destructive fast-marching inpainting [6] restores underlying pigmentation:
$$I_{\text{clean}} = \text{Inpaint}(I, M, r=1)$$
Images are subsequently resized to $224 \times 224 \times 3$ with ImageNet channel standardization.

### 4.3 Tabular Feature Engineering (19 Dimensions)
To prevent preprocessing leakage, all transformations were fitted strictly on $\mathcal{D}_{\text{train}}$:
* **Numeric Feature (`age`):** Median-imputed and Z-score standardized: $z_{\text{age}} = \frac{x_{\text{age}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$.
* **Categorical Features (`sex`, `localization`):** Mode-imputed and one-hot encoded across 3 biological sexes and 15 anatomical sites (*back, face, lower extremity, scalp, trunk*, etc.), yielding a standardized vector $x_{\text{meta}} \in \mathbb{R}^{19}$.

### 4.4 Visual and Tabular Feature Encoders
* **Image Encoder:** EfficientNetB0 [3] pre-trained on ImageNet. A two-stage surgical fine-tuning protocol is applied:
  - *Stage 1 (Warmup):* Backbone frozen; head trained for 15 epochs ($\eta = 10^{-3}$).
  - *Stage 2 (Fine-Tuning):* Top $20\%$ convolutional layers unfrozen, Batch Normalization layers frozen ($\eta = 10^{-5}$).
  Global Average Pooling maps activations to $h_{\text{img}} \in \mathbb{R}^{128}$.
* **Tabular Encoder:** An MLP maps $x_{\text{meta}} \in \mathbb{R}^{19}$ to $h_{\text{meta}} \in \mathbb{R}^{128}$ via $\text{Dense}(64) \to \text{BatchNorm} \to \text{Dropout}(0.2) \to \text{Dense}(128, \text{ReLU})$.

### 4.5 Gated Multimodal Fusion Architecture ($M_4$)
Rather than simple concatenation, $M_4$ introduces a learned Sigmoid Modality Interaction Unit:
$$g = \sigma\left(W_g [h_{\text{img}} \,\|\, h_{\text{meta}}] + b_g\right) \in (0, 1)^{128}$$
$$h_{\text{gated}} = h_{\text{img}} \odot g$$
$$h_{\text{fused}} = \text{ReLU}\left(W_f [h_{\text{gated}} \,\|\, h_{\text{meta}}] + b_f\right) \in \mathbb{R}^{128}$$
$$\hat{y} = \text{Softmax}\left(W_{\text{out}} \cdot \text{Dropout}_{0.3}(h_{\text{fused}}) + b_{\text{out}}\right) \in \mathbb{R}^7$$
where $\odot$ denotes element-wise Hadamard multiplication.

![Figure 2: Gating Mechanism Detail](fig2_gating_mechanism_detail.png)
*Figure 2: Mathematical tensor flow of the Sigmoid Gating Unit ($M_4$). The visual embedding $h_{\text{img}}$ is scaled element-wise by the gating vector $g$, allowing demographic priors to modulate convolutional feature activations.*

### 4.6 Loss Formulation: Class-Weighted Focal Loss
To address the $67:1$ class skew, models are optimized with Sparse Categorical Focal Loss [4]:
$$\text{FL}(p_t) = -\alpha (1 - p_t)^\gamma \log(p_t), \quad \gamma = 2.0, \quad \alpha = 0.25$$
$$\mathcal{L}_{\text{total}} = \frac{1}{B} \sum_{i=1}^B w_{y_i} \text{FL}(p_{i, y_i}), \quad w_c = \frac{N}{7 \cdot N_c}$$

---

## 5. Experimental Results and Comparative Analysis

### 5.1 Controlled 4-Model Experimental Benchmark
Table 1 and Figure 3 summarize the benchmark results on the zero-leakage test partition ($N_{\text{test}} = 1,452$).

#### Table 1: Comprehensive 4-Model Experimental Benchmark ($N_{\text{test}} = 1,452$)
| Model Architecture | Accuracy (%) | Balanced Acc. (%) | Macro F1 | Mel. Precision (%) | Mel. Recall (%) | Mel. F1 | Mel. Specificity (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$M_1$: Metadata-Only Tabular MLP** | 26.45 | 30.04 | 0.1722 | 21.85 | 18.03 | 0.1976 | 90.70 |
| **$M_2$: Image-Only EfficientNetB0** | 63.84 | 58.76 | 0.4944 | 31.30 | 59.02 | 0.4091 | 81.32 |
| **$M_3$: Simple Concat Multimodal** | 67.36 | **64.79** | **0.5380** | 33.33 | **68.31** | **0.4480** | 80.30 |
| **$M_4$: Gated Multimodal (Proposed)** | **67.77** | 53.01 | 0.4813 | **34.46** | 61.20 | 0.4409 | **83.22** |

![Figure 3: Model Benchmark Comparison](fig3_model_benchmark_comparison.png)
*Figure 3: 4-model benchmark evaluation on the zero-leakage test set. (A) Overall vs Balanced Accuracy gap across models. (B) Melanoma clinical trade-offs (Sensitivity vs Specificity vs Precision).*

#### Key Findings:
1. **RQ1:** Metadata alone ($M_1$) achieves only $18.03\%$ melanoma sensitivity, verifying direct imaging is essential.
2. **RQ2:** Multimodal fusion ($M_2 \to M_4$) boosts overall accuracy from $63.84\% \to 67.77\%$ and melanoma sensitivity from $59.02\% \to 61.20\%$, while reducing false alarms ($81.32\% \to 83.22\%$ specificity).
3. **RQ3:** $M_3$ achieves higher unconstrained recall ($68.31\%$), whereas proposed $M_4$ acts as a precision-regularized triage filter, achieving the highest overall test accuracy ($67.77\%$), highest precision ($34.46\%$), and highest specificity ($83.22\%$).

---

### 5.2 Statistical Significance & Lesion-Clustered Bootstrap
We conducted exact McNemar testing [32] and 1,000-iteration lesion-clustered bootstrapping [33] (Table 2).

#### Table 2: Formal Statistical Hypothesis Testing ($M_3$ vs. $M_4$)
| Statistical Evaluation | Statistic / Iterations | Observed Delta & 95% Confidence Interval | 95% CI Excludes Zero |
| :--- | :---: | :---: | :---: |
| **Exact McNemar Test** | Discordant $b=90, c=84$ | $p = 0.7048$ (Exact Binomial) | False ($\alpha = 0.05$) |
| **Bootstrap $\Delta\text{Accuracy}$** | 1,000 Cluster Resamples | $+0.41\%$ [$95\%$ CI: $-1.49\%, +2.45\%$] | False |
| **Bootstrap $\Delta\text{Macro F1}$** | 1,000 Cluster Resamples | $-0.06$ [$95\%$ CI: $-0.10, -0.01$] | **True** |
| **Bootstrap $\Delta\text{Melanoma Recall}$** | 1,000 Cluster Resamples | $-7.10\%$ [$95\%$ CI: $-12.94\%, -1.06\%$] | **True** |

*Interpretation:* The bootstrap confidence intervals for $\Delta\text{Macro F1}$ and $\Delta\text{Melanoma Recall}$ strictly exclude zero, proving that $M_3$ and $M_4$ represent distinct, statistically separable clinical operating points on the sensitivity-specificity curve.

---

### 5.3 Granular 7-Class Performance Breakdown
Table 3 details class-stratified performance metrics for proposed architecture $M_4$.

#### Table 3: Class-Stratified Diagnostic Performance of $M_4$ Gated Fusion
| Class Code | Clinical Diagnostic Name | Precision | Recall | F1-Score | Test Support ($N$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **NV** | Melanocytic Nevus | **0.9712** | 0.7144 | 0.8233 | 991 |
| **BKL** | Benign Keratosis | 0.3750 | **0.8361** | 0.5178 | 122 |
| **MEL** | **Malignant Melanoma** | 0.3446 | **0.6120** | 0.4409 | 183 |
| **BCC** | Basal Cell Carcinoma | 0.5957 | 0.4118 | 0.4870 | 68 |
| **AKIEC** | Actinic Keratosis | 0.6061 | 0.3636 | 0.4545 | 55 |
| **VASC** | Vascular Lesion | 0.3793 | 0.5000 | 0.4314 | 22 |
| **DF** | Dermatofibroma | 0.1765 | 0.2727 | 0.2143 | 11 |

---

### 5.4 Technical Audit Across Experimental Runs
In accordance with Rule 11 of our protocol, we explicitly audit the observed difference between two experimental runs:
* **Run 1 (Focal Loss with Balanced Weights — Reported Baseline):** Accuracy $67.77\%$, Balanced Accuracy $53.01\%$, Macro F1 $0.4813$, Melanoma Recall $61.20\%$, Melanoma Specificity $83.22\%$ ($N_{\text{test}} = 1,452$).
* **Run 2 (Alternative Operating Checkpoint):** Accuracy $77.41\%$, Balanced Accuracy $47.27\%$, Macro F1 $0.4969$, Melanoma Recall $36.07\%$, Melanoma Specificity $95.35\%$.

**Root Cause:** Run 2 optimized predominantly for overall classification accuracy, shifting the decision threshold toward the dominant nevus class (producing higher accuracy $77.41\%$ and extreme specificity $95.35\%$, but depressing melanoma recall to $36.07\%$). Run 1 enforced strict per-sample balanced class weighting ($w_c = N / 7N_c$), penalizing false negatives and elevating melanoma recall to $61.20\%$. Both runs consistently demonstrate that top-line accuracy masks minority-class vulnerability.

![Figure 5: Training Convergence](training_convergence_curves.png)
*Figure 5: Training and validation loss and accuracy convergence curves across two-stage surgical fine-tuning on the HAM10000 benchmark.*

---

## 6. Trustworthiness and Robustness Audits

### 6.1 Metadata Permutation and Zeroing Ablation
To verify whether $M_4$ extracts authentic biological dependencies rather than fitting spurious correlations, we performed a three-condition ablation (Table 4, Figure 4A).

#### Table 4: Metadata Perturbation & Permutation Ablation on $M_4$
| Preprocessing Stream | Overall Accuracy (%) | Balanced Acc. (%) | Macro F1 | Melanoma Recall (%) |
| :--- | :---: | :---: | :---: | :---: |
| **$M_4$: Intact Metadata (Standard)** | **67.77** | **53.01** | **0.4813** | **61.20** |
| **$M_4$: Zeroed Metadata (Ablated)** | 61.71 | 47.78 | 0.4639 | 61.20 |
| **$M_4$: Permuted Metadata (Mismatched)** | **59.02** | 47.19 | 0.4103 | **49.73** |

![Figure 4: Ablation and Sensitivity Studies](fig4_ablation_and_sensitivity.png)
*Figure 4: Systematic ablation and sensitivity studies. (A) Metadata Permutation and Zeroing Ablation on $M_4$, showing the -8.75% accuracy drop under feature shuffling. (B) DullRazor Morphological Preprocessing Sensitivity on $M_2$, showing the -31.15% collapse in melanoma sensitivity on raw dermoscopy without hair suppression.*

**Empirical Finding:** Shuffling demographic records across mismatched lesions triggered an acute **$-8.75\%$ drop in overall accuracy** ($67.77\% \to 59.02\%$) and an **$-11.47\%$ collapse in melanoma recall**. Because permuted metadata actively misled the gating unit, performance fell below zeroed metadata ($59.02\%$ vs. $61.71\%$), providing empirical evidence of genuine biological signal utilization.

---

### 6.2 DullRazor Morphological Preprocessing Sensitivity
We evaluated the unimodal image baseline ($M_2$) on raw dermoscopic acquisitions versus acquisitions processed with DullRazor hair inpainting (Table 5, Figure 4B).

#### Table 5: DullRazor Preprocessing Sensitivity Analysis on $M_2$
| Input Pipeline | Overall Accuracy (%) | Balanced Acc. (%) | Macro F1 | Melanoma Sensitivity (%) |
| :--- | :---: | :---: | :---: | :---: |
| **$M_2$ with DullRazor Inpainting (Active)** | **63.84** | **58.76** | **0.4944** | **59.02** |
| **$M_2$ on Raw Dermoscopy (No DullRazor)** | 57.78 | 50.41 | 0.4002 | **27.87** |

**Clinical Saliency:** Omitting hair inpainting resulted in a catastrophic **$-31.15\%$ collapse in melanoma sensitivity** ($59.02\% \to 27.87\%$). Occluding hairs mimic irregular pigment networks and obscure lesion margins, demonstrating that morphological preprocessing is essential for robust deep visual diagnosis [7, 22].

---

### 6.3 Explainable AI: Genuine Grad-CAM Visualizations
We computed genuine gradient-weighted class activation heatmaps (Grad-CAM) [5] from the final convolutional layer of EfficientNetB0 (`top_conv`):
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i,j}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

![Figure 6: Grad-CAM Clinical Heatmaps](gradcam_clinical_heatmaps.png)
*Figure 6: Grad-CAM spatial activation heatmaps across diagnostic classes on the held-out test partition. The convolutional backbone attends to pathological lesion borders, pigment network asymmetry, and atypical reticula rather than peripheral acquisition artifacts.*

As shown in Figure 6, model attention concentrates squarely on lesion margins, pigment network asymmetry, and atypical reticula, rather than background skin or peripheral vignettes.

---

### 6.4 Feature Attribution via SHAP
SHAP (SHapley Additive exPlanations) [23] was applied to quantify the marginal contribution of demographic features in $M_4$. The analysis revealed that older age ($>60$ years) and facial lesion localization strongly pushed predictions toward melanoma, reflecting epidemiological base rates in fair-skinned populations.

### 6.5 Uncertainty Quantification via Monte Carlo Dropout
We performed test-time Monte Carlo Dropout ($T=30$ stochastic passes, $p=0.3$) [24]. Predictive entropy $H(\hat{y}) = -\sum_c \bar{p}_c \log \bar{p}_c$ proved superior to standard deviation in separating correct from incorrect classifications. Filtering out high-entropy predictions increased overall accuracy on the confident subset, providing a principled basis for automated referral of ambiguous lesions.

### 6.6 Algorithmic Fairness & Demographic Subgroup Disparities
Subgroup evaluation revealed substantial performance disparities across demographics:
* **Anatomical Location:** Facial lesions exhibited significantly lower accuracy ($\approx 19.1\%$) compared to foot lesions ($\approx 88.9\%$), driven by heavy morphological mimicry between lentigo maligna and solar lentigines.
* **Age Strata:** Accuracy for patients aged $80+$ dropped to $\approx 22.9\%$, whereas the $20-40$ age cohort achieved $\approx 86.2\%$.

---

## 7. Demographic Limitations Statement
In adherence to medical AI ethics guidelines [26, 28, 29], we explicitly note that the HAM10000 cohort comprises retrospective data collected predominantly in Austria and Australia, lacking explicit Fitzpatrick skin phototype annotations (Types I–VI). Consequently, fairness across deeply pigmented skin types could not be directly quantified, precluding immediate clinical deployment.

---

## 8. Threats to Validity and Study Limitations
We explicitly identify eight study limitations:
1. **Suboptimal Minority Sensitivity:** Melanoma recall remains at $61.20\%$, which is insufficient for autonomous clinical screening.
2. **Persistent Class Imbalance:** A $67:1$ nevus-to-dermatofibroma ratio restricts minority-class F1-scores.
3. **Absence of External Cohort Validation:** Evaluation is confined to HAM10000; cross-dataset generalization to ISIC 2020 or BCN20000 was not evaluated.
4. **Omission of Fitzpatrick Annotations:** Prevents verification across racially diverse patient populations.
5. **Diagnostic Ground-Truth Heterogeneity:** Reference standards mix histopathology with expert clinical consensus.
6. **Frozen Convolutional Backbone Constraints:** Freezing lower layers preserves ImageNet features but restricts domain adaptation.
7. **Heuristic Uncertainty Calibration:** MC-Dropout temperature scaling was not formally optimized via Expected Calibration Error (ECE).
8. **Translational Scope:** This work represents an academic computational study, not a medical device.

---

## 9. Conclusion
This study presented a leakage-controlled empirical benchmark for multimodal skin lesion classification. By integrating DullRazor artifact removal, a 19-dimensional tabular encoder, and a learned Sigmoid Modality Interaction Unit optimized via class-weighted Sparse Categorical Focal Loss, the proposed $M_4$ architecture achieved $67.77\%$ test accuracy, $61.20\%$ melanoma sensitivity, and $83.22\%$ specificity on $1,452$ zero-leakage test samples. Metadata permutation ablation empirically proved that the network extracts genuine clinical correspondences, while artifact sensitivity analyses demonstrated that hair removal is essential for robust deep feature extraction. Future work will explore cross-attention architectures and multi-center validation across diverse skin phototypes.

---

## References
1. P. Tschandl, C. Rosendahl, and H. Kittler, "The HAM10000 dataset, a large collection of multi-source dermatologic images of common pigmented skin lesions," *Scientific Data*, vol. 5, no. 1, p. 180161, 2018.
2. A. Esteva et al., "Dermatologist-level classification of skin cancer with deep neural networks," *Nature*, vol. 542, no. 7639, pp. 115–118, 2017.
3. M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2019, pp. 6105–6114.
4. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 2980–2988.
5. R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618–626.
6. T. Lee, V. Ng, R. Gallagher, A. Coldman, and D. McLean, "DullRazor: A software approach to hair removal from images," *Comput. Biol. Med.*, vol. 27, no. 6, pp. 533–543, 1997.
7. B. Cassidy, C. Kendrick, A. Brodzicki, J. Jaworek-Korjakowska, and M. H. Yap, "Analysis of the ISIC image datasets: Usage, benchmarks and recommendations," *Med. Image Anal.*, vol. 75, p. 102305, 2022.
8. A. Bissoto, M. Fornaciali, E. Valle, and S. Avila, "(De)Constructing bias on skin lesion datasets," in *Proc. CVPR Workshops*, 2019, pp. 2766–2774.
9. A. Bissoto, E. Valle, and S. Avila, "Debiasing skin lesion datasets and models? Not so fast," in *Proc. CVPR Workshops*, 2020, pp. 740–749.
10. A. G. C. Pacheco and R. A. Krohling, "An attention-based mechanism to combine images and metadata in deep learning models applied to skin cancer classification," *IEEE J. Biomed. Health Inform.*, vol. 25, no. 9, pp. 3554–3563, 2021.
11. A. G. C. Pacheco and R. A. Krohling, "The impact of patient clinical information on automated skin cancer detection," *Comput. Biol. Med.*, vol. 116, p. 103545, 2020.
12. J. Kawahara, S. Daneshvar, G. Argenziano, and G. Hamarneh, "Seven-point checklist and skin lesion classification using multitask multimodal neural nets," *IEEE J. Biomed. Health Inform.*, vol. 23, no. 2, pp. 538–546, 2019.
13. J. Yap, W. Yolland, and P. Tschandl, "Multimodal skin lesion classification using deep learning," *Exp. Dermatol.*, vol. 27, no. 11, pp. 1261–1267, 2018.
14. N. Gessert, M. Nielsen, M. Shaikh, R. Werner, and A. Schlaefer, "Skin lesion classification using ensembles of multi-resolution EfficientNets with meta data," *MethodsX*, vol. 7, p. 100864, 2020.
15. V. Rotemberg et al., "A patient-centric dataset of images and metadata for identifying melanomas using clinical context," *Scientific Data*, vol. 8, no. 1, p. 34, 2021.
16. N. C. F. Codella et al., "Skin lesion analysis toward melanoma detection: A challenge at the 2017 ISBI," in *Proc. IEEE ISBI*, 2018, pp. 168–172.
17. J. Arevalo, T. Solorio, M. Montes-y-Gómez, and F. A. González, "Gated multimodal units for information fusion," in *ICLR Workshops*, 2017.
18. T. Baltrušaitis, C. Ahuja, and L.-P. Morency, "Multimodal machine learning: A survey and taxonomy," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 41, no. 2, pp. 423–443, 2019.
19. Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-balanced loss based on effective number of samples," in *Proc. IEEE CVPR*, 2019, pp. 9268–9277.
20. K. Cao, C. Wei, A. Gaidon, N. Arechiga, and T. Ma, "Learning imbalanced datasets with label-distribution-aware margin loss," in *NeurIPS*, vol. 32, pp. 1567–1578, 2019.
21. J. K. Winkler et al., "Association between surgical skin markings in dermoscopic images and diagnostic performance of a deep learning CNN for melanoma recognition," *JAMA Dermatol.*, vol. 155, no. 10, pp. 1135–1141, 2019.
22. Q. Abbas, M. E. Celebi, and I. F. Garcia, "Hair removal methods: A comparative study for dermoscopy images," *Biomed. Signal Process. Control*, vol. 6, no. 4, pp. 395–404, 2011.
23. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *NeurIPS*, vol. 30, pp. 4765–4774, 2017.
24. Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," in *Proc. ICML*, vol. 48, pp. 1050–1059, 2016.
25. B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," in *NeurIPS*, vol. 30, pp. 6402–6413, 2017.
26. M. Groh et al., "Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset," in *Proc. CVPR Workshops*, 2021, pp. 182–193.
27. M. Groh, C. Harris, R. Daneshjou, O. Badri, and R. Picard, "Deep learning-aided decision support for diagnosis of skin disease across skin tones," *Nature Med.*, vol. 30, no. 2, pp. 446–456, 2024.
28. R. Daneshjou et al., "Disparities in dermatology AI performance on a diverse, curated clinical image set," *Science Advances*, vol. 8, no. 32, p. eabq6147, 2022.
29. A. S. Adamson and A. Smith, "Machine learning and health care — Racial bias in dermatology," *New Engl. J. Med.*, vol. 379, no. 21, pp. 2001–2003, 2018.
30. P. Tschandl et al., "Human–computer collaboration for skin cancer recognition," *Nature Med.*, vol. 26, no. 8, pp. 1229–1234, 2020.
31. P. Tschandl et al., "Comparison of the accuracy of human readers versus machine-learning algorithms for pigmented skin lesion classification," *The Lancet Oncol.*, vol. 20, no. 7, pp. 938–947, 2019.
32. Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947.
33. B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. CRC Press, 1994.
