# Satellite Image Land-Use Classification & Temporal Change Detection
### Project Report (max 8 pages)

---

## 1. Problem Statement (≈0.5 page)
- Motivation: monitoring land-use / land-cover change from satellite imagery
  (urbanization, deforestation, agricultural shifts).
- Two sub-problems: (a) 10-class land-use classification, (b) temporal change
  detection between two time periods using learned embeddings.

## 2. Data (≈1 page)
- EuroSAT: 27,000 RGB tiles, 10 classes, 64×64 px. Class distribution figure
  (`outputs/class_distribution.png`).
- UC Merced Land Use: 2,100 images, 21 classes, used as an out-of-distribution
  holdout — remapped onto the 10 EuroSAT classes (`data/download_eurosat.py::UCM_TO_EUROSAT_MAP`).
- Spatial block train/val/test split and why a naive random split leaks
  (`evaluation/spatial_leakage.py`, results table + explanation).

## 3. Method (≈2 pages)
### 3.1 Baseline
- 3-layer scratch CNN architecture (`models/baseline_cnn.py`), loss curves.

### 3.2 Transfer learning
- Backbone: ResNet-18 / EfficientNet-B0 (state which was used and why).
- Two-phase fine-tuning: Phase 1 (frozen backbone, 3 epochs), Phase 2
  (unfreeze last 2 conv blocks, LR/10, 5 epochs).
- Frozen vs unfrozen ablation table.

### 3.3 Change detection
- Backbone reused as a 512-d feature extractor (classifier head stripped).
- Simulated T1/T2 region pairs (`change_detection/temporal_split.py`) — be
  transparent that EuroSAT lacks true multi-date imagery and this is a
  documented simulation.
- Cosine similarity, ROC curve, Youden's-J threshold selection.

### 3.4 Dashboard
- Streamlit app architecture: classifier + embedding extractor + similarity
  scorer + heatmap renderer.

## 4. Results (≈2.5 pages)
- Per-class F1 + macro-F1 tables: EuroSAT val, UC Merced holdout.
- Confusion matrices (embed the two PNGs).
- ROC curve + AUC + chosen operating point (embed `roc_curve.png`).
- 5+ change heatmaps (embed a couple, reference the rest in an appendix or repo link).
- Spatial leakage quantified gap.

## 5. Error Analysis (≈1 page)
- Top-5 misclassified pairs, image grid, and your hypothesis for each failure
  (fill in from `outputs/error_analysis/hypotheses.md`).

## 6. Bonus Work (if attempted) (≈0.5–1 page)
- Clearly label which bonus tasks (A/B/C/D) were attempted and summarise
  results (GradCAM interpretation, threshold toggle behaviour, t-SNE/UMAP
  comparison, imbalance mitigation table).

## 7. Limitations & Future Work (≈0.5 page)
- Change detection is validated on a simulated T1/T2 split, not true
  multi-temporal imagery — flag this clearly as a limitation.
- Backbone choice trade-offs, dataset size constraints, domain gap between
  EuroSAT and UC Merced (different sensors/resolutions).

## Appendix
- Full per-class metric tables, additional heatmaps, training logs.
