import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Ensure output directories
out_dir1 = r"C:\Users\Nikhil\Downloads\Skin_Lesion_Web_App\outputs"
out_dir2 = r"C:\Users\Nikhil\Downloads\ML Project\outputs"
os.makedirs(out_dir1, exist_ok=True)
os.makedirs(out_dir2, exist_ok=True)

# Publication style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.edgecolor'] = '#333333'

# ==============================================================================
# FIGURE 1: End-to-End System Architecture Pipeline Flowchart
# ==============================================================================
fig = plt.figure(figsize=(14, 7), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.5)
ax.axis('off')

def draw_box(ax, x, y, w, h, title, subtitle, color, text_color='#111827', edge_color='#4B5563', corner_radius=0.15):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.08,rounding_size={corner_radius}",
                                  facecolor=color, edgecolor=edge_color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=text_color)
    if subtitle:
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=7.8, color='#374151')

def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color='#1F2937', lw=1.5, mutation_scale=14))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, label, ha='center', va='bottom', fontsize=7.5, color='#1F2937', fontweight='semibold')

# Title Banner
ax.text(7.0, 7.2, "End-to-End Gated Multimodal Deep Learning Diagnostic Framework (HAM10000)",
        ha='center', va='center', fontsize=13, fontweight='bold', color='#0F172A')
ax.axhline(6.95, xmin=0.05, xmax=0.95, color='#CBD5E1', lw=1)

# Stream A: Dermoscopy Image
draw_box(ax, 0.4, 4.8, 2.2, 1.4, "Dermoscopic Image", "HAM10000 (224×224×3)\nGroup-Aware Split", "#E0F2FE", edge_color="#0284C7")
draw_box(ax, 3.2, 4.8, 2.3, 1.4, "DullRazor Filter", "Black-Hat Structuring (9×9)\n+ Telea Inpaint (r=1)", "#BAE6FD", edge_color="#0284C7")
draw_box(ax, 6.1, 4.8, 2.5, 1.4, "Visual Backbone", "EfficientNetB0 (Top 20% Unfrozen)\nGAP → Dense(128)", "#7DD3FC", edge_color="#0369A1")

# Stream B: Patient Metadata
draw_box(ax, 0.4, 1.2, 2.2, 1.4, "Clinical Metadata", "Age, Sex, 15 Anatomical Sites\nZero-Leakage Grouping", "#FEF3C7", edge_color="#D97706")
draw_box(ax, 3.2, 1.2, 2.3, 1.4, "Preproc & Encoding", "Median Impute + Z-Score\n+ One-Hot Encoding (19-D)", "#FDE68A", edge_color="#D97706")
draw_box(ax, 6.1, 1.2, 2.5, 1.4, "Tabular Sub-Network", "Dense(64) → BN → Drop(0.2)\n→ Dense(128, ReLU)", "#FCD34D", edge_color="#B45309")

# Fusion Center
draw_box(ax, 9.2, 2.8, 2.4, 2.0, "Gated Modality Unit", "g = σ(W_g [h_img || h_meta])\nh_gated = h_img ⊙ g\nConcat [h_gated || h_meta] (256-D)", "#DCFCE7", edge_color="#16A34A")

# Classification Head & Loss
draw_box(ax, 12.0, 2.8, 1.6, 2.0, "Decision Head", "Dense(128, ReLU)\n→ Dropout(0.3)\n→ Softmax(7 Classes)\nClass-Weighted Focal Loss", "#FCE7F3", edge_color="#DB2777")

# Arrows
draw_arrow(ax, 2.6, 5.5, 3.2, 5.5)
draw_arrow(ax, 5.5, 5.5, 6.1, 5.5)
draw_arrow(ax, 8.6, 5.5, 9.2, 4.2, "h_img (128-D)")

draw_arrow(ax, 2.6, 1.9, 3.2, 1.9)
draw_arrow(ax, 5.5, 1.9, 6.1, 1.9)
draw_arrow(ax, 8.6, 1.9, 9.2, 3.4, "h_meta (128-D)")

draw_arrow(ax, 11.6, 3.8, 12.0, 3.8, "h_fused (256-D)")

plt.tight_layout()
p1 = os.path.join(out_dir1, "fig1_multimodal_architecture_pipeline.png")
p1_copy = os.path.join(out_dir2, "fig1_multimodal_architecture_pipeline.png")
plt.savefig(p1, bbox_inches='tight', dpi=300)
plt.savefig(p1_copy, bbox_inches='tight', dpi=300)
plt.close()
print("Saved Fig 1:", p1)


# ==============================================================================
# FIGURE 2: Detailed Sigmoid Gating Mechanism Tensor Flow
# ==============================================================================
fig = plt.figure(figsize=(11, 6), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 11)
ax.set_ylim(0, 6.5)
ax.axis('off')

ax.text(5.5, 6.2, "Detailed Sigmoid Modality Interaction Gating Unit (Mathematical Flow)",
        ha='center', va='center', fontsize=12, fontweight='bold', color='#0F172A')
ax.axhline(5.95, xmin=0.08, xmax=0.92, color='#CBD5E1', lw=1)

# Inputs
draw_box(ax, 0.5, 4.0, 2.0, 1.2, "Image Embedding", "h_img ∈ ℝ^128", "#E0F2FE", edge_color="#0284C7")
draw_box(ax, 0.5, 1.2, 2.0, 1.2, "Metadata Vector", "h_meta ∈ ℝ^128", "#FEF3C7", edge_color="#D97706")

# Concat for Gating
draw_box(ax, 3.2, 2.4, 2.0, 1.6, "Cross-Modal Concat", "[h_img || h_meta]\n∈ ℝ^256", "#EDE9FE", edge_color="#7C3AED")

# Dense + Sigmoid
draw_box(ax, 5.8, 2.4, 1.8, 1.6, "Gating Activation", "Dense(128) + σ\ng ∈ (0, 1)^128", "#DDD6FE", edge_color="#6D28D9")

# Element-wise Product
draw_box(ax, 8.2, 4.0, 2.2, 1.4, "Feature Modulation", "h_gated = h_img ⊙ g\n(Learned Attention Filter)", "#DCFCE7", edge_color="#16A34A")

# Final Representation
draw_box(ax, 8.2, 1.2, 2.2, 1.4, "Fused Embedding", "Dense([h_gated || h_meta])\n→ Classifier", "#FCE7F3", edge_color="#DB2777")

# Connectors
draw_arrow(ax, 2.5, 4.6, 3.2, 3.4)
draw_arrow(ax, 2.5, 1.8, 3.2, 3.0)
draw_arrow(ax, 5.2, 3.2, 5.8, 3.2)
draw_arrow(ax, 7.6, 3.2, 8.2, 4.5, "g ∈ (0,1)^128")
draw_arrow(ax, 2.5, 4.8, 8.2, 4.8, "h_img pass-through")
draw_arrow(ax, 9.3, 4.0, 9.3, 2.6, "h_gated")
draw_arrow(ax, 2.5, 1.5, 8.2, 1.7, "h_meta skip-connection")

plt.tight_layout()
p2 = os.path.join(out_dir1, "fig2_gating_mechanism_detail.png")
p2_copy = os.path.join(out_dir2, "fig2_gating_mechanism_detail.png")
plt.savefig(p2, bbox_inches='tight', dpi=300)
plt.savefig(p2_copy, bbox_inches='tight', dpi=300)
plt.close()
print("Saved Fig 2:", p2)


# ==============================================================================
# FIGURE 3: 4-Model Benchmark Comparative Analysis
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

models = ['M1: Metadata\nOnly (Tabular)', 'M2: Image\nOnly (EffNet)', 'M3: Simple\nConcat Fusion', 'M4: Gated\nFusion (Prop.)']
colors = ['#94A3B8', '#38BDF8', '#F59E0B', '#10B981']

# Left Plot: Accuracy vs Balanced Accuracy
x = np.arange(len(models))
width = 0.35

acc = [26.45, 63.84, 67.36, 67.77]
bal_acc = [30.04, 58.76, 64.79, 53.01]

rects1 = ax1.bar(x - width/2, acc, width, label='Top-1 Accuracy (%)', color='#2563EB', alpha=0.85, edgecolor='#1E3A8A', lw=1.2)
rects2 = ax1.bar(x + width/2, bal_acc, width, label='Balanced Accuracy (%)', color='#0D9488', alpha=0.85, edgecolor='#115E59', lw=1.2)

ax1.set_ylabel('Metric Score (%)', fontsize=10, fontweight='bold')
ax1.set_title('(A) Aggregate vs Balanced Accuracy Gap', fontsize=11, fontweight='bold', color='#1E293B', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=8.5)
ax1.set_ylim(0, 80)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9)

for r in rects1:
    h = r.get_height()
    ax1.text(r.get_x() + r.get_width()/2., h + 1.0, f'{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
for r in rects2:
    h = r.get_height()
    ax1.text(r.get_x() + r.get_width()/2., h + 1.0, f'{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Right Plot: Clinical Melanoma Operating Trade-offs (Recall vs Specificity vs Precision)
mel_rec = [18.03, 59.02, 68.31, 61.20]
mel_spec = [90.70, 81.32, 80.30, 83.22]
mel_prec = [21.85, 31.30, 33.33, 34.46]

w3 = 0.25
r1 = ax2.bar(x - w3, mel_rec, w3, label='Melanoma Recall (Sensitivity)', color='#EF4444', alpha=0.85, edgecolor='#991B1B', lw=1.2)
r2 = ax2.bar(x, mel_spec, w3, label='Melanoma Specificity', color='#10B981', alpha=0.85, edgecolor='#065F46', lw=1.2)
r3 = ax2.bar(x + w3, mel_prec, w3, label='Melanoma Precision', color='#8B5CF6', alpha=0.85, edgecolor='#5B21B6', lw=1.2)

ax2.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
ax2.set_title('(B) Melanoma Clinical Trade-off (Recall vs Specificity vs Precision)', fontsize=11, fontweight='bold', color='#1E293B', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=8.5)
ax2.set_ylim(0, 100)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=8.5, loc='upper right')

for r in r1:
    h = r.get_height()
    ax2.text(r.get_x() + r.get_width()/2., h + 1.2, f'{h:.1f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
for r in r2:
    h = r.get_height()
    ax2.text(r.get_x() + r.get_width()/2., h + 1.2, f'{h:.1f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

plt.tight_layout()
p3 = os.path.join(out_dir1, "fig3_model_benchmark_comparison.png")
p3_copy = os.path.join(out_dir2, "fig3_model_benchmark_comparison.png")
plt.savefig(p3, bbox_inches='tight', dpi=300)
plt.savefig(p3_copy, bbox_inches='tight', dpi=300)
plt.close()
print("Saved Fig 3:", p3)


# ==============================================================================
# FIGURE 4: Metadata Perturbation & DullRazor Sensitivity Analysis
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Panel A: Metadata Ablation
ablation_conds = ['Intact Metadata\n(Standard M4)', 'Zeroed Metadata\n(x_meta = 0)', 'Permuted Metadata\n(Mismatched Shuffled)']
ab_acc = [67.77, 61.71, 59.02]
ab_rec = [61.20, 61.20, 49.73]
ab_x = np.arange(len(ablation_conds))
w = 0.35

b1 = ax1.bar(ab_x - w/2, ab_acc, w, label='Overall Accuracy (%)', color='#3B82F6', edgecolor='#1D4ED8', lw=1.2)
b2 = ax1.bar(ab_x + w/2, ab_rec, w, label='Melanoma Recall (%)', color='#F59E0B', edgecolor='#B45309', lw=1.2)

ax1.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
ax1.set_title('(A) Metadata Permutation & Zeroing Ablation (M4)', fontsize=11, fontweight='bold', color='#1E293B', pad=10)
ax1.set_xticks(ab_x)
ax1.set_xticklabels(ablation_conds, fontsize=8.5)
ax1.set_ylim(0, 80)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9)

# Annotation for drop
ax1.annotate('Δ = -8.75% collapse\n(Proves biological signal)', xy=(2 - w/2, 59.02), xytext=(1.4, 71),
             arrowprops=dict(facecolor='#DC2626', shrink=0.08, width=1.5, headwidth=6),
             fontsize=8, fontweight='bold', color='#DC2626', bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#EF4444", lw=1))

for r in b1:
    h = r.get_height()
    ax1.text(r.get_x() + r.get_width()/2., h + 1.0, f'{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
for r in b2:
    h = r.get_height()
    ax1.text(r.get_x() + r.get_width()/2., h + 1.0, f'{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Panel B: DullRazor Sensitivity
dull_conds = ['DullRazor Inpainting\n(Active Baseline)', 'Raw Dermoscopy\n(No Hair Removal)']
d_acc = [63.84, 57.78]
d_rec = [59.02, 27.87]
d_x = np.arange(len(dull_conds))

b3 = ax2.bar(d_x - w/2, d_acc, w, label='Overall Accuracy (%)', color='#6366F1', edgecolor='#4338CA', lw=1.2)
b4 = ax2.bar(d_x + w/2, d_rec, w, label='Melanoma Recall (%)', color='#EF4444', edgecolor='#B91C1C', lw=1.2)

ax2.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
ax2.set_title('(B) DullRazor Morphological Preprocessing Sensitivity (M2)', fontsize=11, fontweight='bold', color='#1E293B', pad=10)
ax2.set_xticks(d_x)
ax2.set_xticklabels(dull_conds, fontsize=9)
ax2.set_ylim(0, 80)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9)

ax2.annotate('Δ = -31.15% sensitivity collapse\n(Hair artifacts obscure malignant borders)',
             xy=(1 + w/2, 27.87), xytext=(0.5, 45),
             arrowprops=dict(facecolor='#DC2626', shrink=0.08, width=1.5, headwidth=6),
             fontsize=8, fontweight='bold', color='#DC2626', bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#EF4444", lw=1))

for r in b3:
    h = r.get_height()
    ax2.text(r.get_x() + r.get_width()/2., h + 1.0, f'{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
for r in b4:
    h = r.get_height()
    ax2.text(r.get_x() + r.get_width()/2., h + 1.0, f'{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
p4 = os.path.join(out_dir1, "fig4_ablation_and_sensitivity.png")
p4_copy = os.path.join(out_dir2, "fig4_ablation_and_sensitivity.png")
plt.savefig(p4, bbox_inches='tight', dpi=300)
plt.savefig(p4_copy, bbox_inches='tight', dpi=300)
plt.close()
print("Saved Fig 4:", p4)
print("ALL 4 PUBLICATION FIGURES GENERATED SUCCESSFULLY!")
