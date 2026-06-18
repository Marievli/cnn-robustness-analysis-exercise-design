# Shape or Color? A CNN Robustness Study on Fruit & Vegetable Recognition

**Assignment 4 · Task 2 — Introduction to Data Science · University of Tehran**  
Designed by [Maryam Vali](https://github.com/Marievli)\
Instructors: Dr. Razavi, Dr. Yaghoobzadeh

---

## Overview

This task goes beyond standard image classification. Students train a CNN on 10 classes of fruit and vegetable images, then systematically probe *what* the model has actually learned: genuine morphological shape, or a shortcut through color statistics?

The answer has real consequences. A model that learned "bananas are yellow" will fail the moment lighting conditions change or a camera captures in grayscale. A model that learned "bananas are curved" will not. This task is designed to surface that difference rigorously.

---

## Learning Objectives

By completing this task, students will:

- Build a **complete CNN pipeline** from raw images to a deployed classifier, including preprocessing, augmentation, normalization, and early stopping — without any data leakage between splits
- Derive the **mathematical foundations** of convolutional architectures from first principles: output dimensions, trainable parameter counts, and receptive field growth across stacked layers
- Conduct **statistical EDA on image data** beyond pixel inspection — computing per-class RGB distributions, Euclidean color distances, and predicting failure modes before training begins
- Design and execute a **systematic robustness analysis** across four controlled perturbation scenarios (baseline, grayscale, channel swap, Gaussian noise), isolating whether the model's decision boundary is driven by color shortcuts or genuine shape features
- Interpret model behavior through **Grad-CAM explainability**, identifying spurious spatial correlations (e.g., background bias) and connecting attention map shifts across scenarios to the model's internal reasoning
- Synthesize EDA predictions with post-hoc evaluation — verifying whether color-proximity warnings from the distance matrix align with actual confusion patterns in grayscale conditions, and drawing principled conclusions about robustness in real deployment environments

---

## Dataset

**Fruits and Vegetables Image Recognition** ([Kaggle](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition))

- **10 classes** selected to stress-test color-vs-shape discrimination
- Baseline selection: Apple, Banana, Bell Pepper, Carrot, Grapes, Kiwi, Lemon, Orange, Pear, Tomato
  - Several of these share similar color profiles (e.g., Orange/Lemon, Tomato/Apple), deliberately forcing the model to rely on shape
  - Deviations from this selection require scientific justification in the report
- All images resized to **128 × 128**, RGB
- Pre-existing Train / Validation / Test splits must be used as-is

---

## Task Structure

### Section 1 — Dataset Configuration
Load exactly 10 classes. Report per-class sample counts for each split and plot a class distribution bar chart. Analyze whether class imbalance exists and justify the choice of evaluation metric accordingly.

### Section 2 — Statistical EDA & Preprocessing

**2.1 Color Distribution Analysis**  
For each class, compute per-channel (R, G, B) means and visualize them as:
- A grouped bar chart (R/G/B side-by-side per class)
- Pairwise RGB scatter plots (R vs G, R vs B, G vs B)
- A Euclidean distance matrix heatmap across all class pairs in RGB space

> **Analytical Question:** Which 3 class pairs are at highest risk of confusion based on color proximity? Revisit this prediction after computing the confusion matrix from model evaluation.

**2.2 Global Normalization**  
Compute per-channel mean (μ) and standard deviation (σ) **exclusively on the training set**. Apply Z-score normalization to all splits using these statistics. Explain analytically why fitting the scaler on the test set constitutes data leakage.

**2.3 Data Augmentation Pipeline**  
Design a training-only augmentation pipeline with: random rotation (≤15°), brightness/contrast jitter, random horizontal flip, and optionally random cutout. Display ≥5 original/augmented image pairs side-by-side.

---

### Section 3 — Mathematical Foundations *(PDF report only, 20% of grade)*

All calculations must be shown step-by-step with formulas. Consider the following architecture applied to a 128 × 128 × 3 input:

| Layer | Parameters |
|-------|-----------|
| Conv2D | 32 filters · 5×5 kernel · stride 1 · padding 2 |
| MaxPool2D | 2×2 window · stride 2 |
| Conv2D | 64 filters · 3×3 kernel · stride 1 · padding 1 |

**Q1:** Compute output dimensions (H × W × C) at each stage using:

$$\text{Output Size} = \left\lfloor \frac{\text{Input} + 2P - K}{S} \right\rfloor + 1$$

**Q2:** Compute the exact number of trainable parameters (weights + biases) per layer. State the formula explicitly.

**Q3:** Compute the Receptive Field of a neuron at the third layer's output using the recursive formula:

$$RF_l = RF_{l-1} + (K_l - 1) \times \prod_{i=1}^{l-1} S_i$$

**Q4:** If the first layer's stride changes from 1 to 2, what is the quantitative change in the third layer's receptive field? Discuss the trade-off between spatial resolution and computational efficiency.

---

### Section 4 — Model Architecture & Training

**Feature Extractor:** At least 4 convolutional blocks following the pattern:

```
[Conv2D → BatchNorm → ReLU → MaxPool2D]
```

Start with 32 filters, doubling in each subsequent block.

**Classifier Head:** Use Global Average Pooling (GAP) instead of Flatten. Explain why GAP reduces overfitting and parameter count relative to a fully connected flatten layer.

**Training Configuration:**

| Component | Setting |
|-----------|---------|
| Optimizer | Adam (lr = 1e-3) |
| Loss | Cross-Entropy |
| LR Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Batch Size | 32 or 64 |
| Max Epochs | 50 with Early Stopping (patience=5 on val loss) |

Document training curves (loss and accuracy on shared axes), mark the early stopping epoch, report average epoch time, and identify epochs where the LR scheduler fired.

---

### Section 5 — Robustness Analysis: Color vs. Shape *(25% of grade)*

Four evaluation scenarios on the held-out test set:

| Scenario | Description |
|----------|-------------|
| A — Baseline | Standard test set evaluation |
| B — Grayscale | Convert test images to grayscale (replicated across 3 channels) |
| C — Channel Swap | Swap R and B channels (turns bananas purple, grapes orange) |
| D — Gaussian Noise *(Bonus)* | Add Gaussian noise (σ = 0.1) |

For each scenario: report overall accuracy, accuracy drop vs. baseline, the class with the largest individual drop, and the most confused class pair.

> **Analytical Question (Required):** Which scenario caused the greatest accuracy drop? Did the class pairs identified as high-risk in the EDA color distance matrix show the highest drop in the Grayscale test? What does this imply for deploying vision systems in low-light or monochrome environments?

---

### Section 6 — Explainability with Grad-CAM *(10% of grade)*

Implement Grad-CAM on the final convolutional layer. Select at least 2 correct and 2 incorrect predictions across different classes, overlaying heatmaps on the original images.

For cross-scenario comparison: display the Grad-CAM heatmap of a single image side-by-side under Scenario A (color) and Scenario B (grayscale). Has the model's attention region shifted?

> **Analytical Question:** In incorrect predictions, does the heatmap indicate the model attended to the image background rather than the object? If so, what techniques (Cutout Augmentation, background removal, etc.) would address this spurious correlation?

---

## Evaluation Criteria

| Criterion | Weight |
|-----------|--------|
| Mathematical calculations (Section 3) | 20% |
| EDA quality and visualisation depth | 25% |
| Model performance (target: >75% accuracy) | 20% |
| Robustness analysis depth | 25% |
| Grad-CAM interpretation | 10% |

> **Note:** A model with 75% accuracy and rigorous robustness/Grad-CAM analysis outweighs a model with 80% accuracy and superficial interpretation. The depth of analysis is the primary criterion.

---

## Repository Structure

```
├── CNN-task.ipynb       # Student notebook
└── README.md
```

Solution notebooks and trained model weights are withheld until the submission deadline.

---

## References

- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017
- Lin et al., *Network in Network* (Global Average Pooling), ICLR 2014
- Szegedy et al., *Rethinking the Inception Architecture* (Receptive Field analysis), CVPR 2016
- DeVries & Taylor, *Improved Regularization of Convolutional Neural Networks with Cutout*, 2017
