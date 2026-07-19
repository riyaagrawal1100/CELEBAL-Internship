# Change Detection — Threshold Selection

ROC AUC: **0.940**

Operating point chosen via Youden's J statistic (maximises TPR - FPR):
- change-score threshold = 0.531  (i.e. cosine similarity threshold = 0.469)
- True Positive Rate at this point: 0.920
- False Positive Rate at this point: 0.088

**Justification:** Youden's J balances sensitivity and specificity without
assuming a particular cost ratio between missed changes (false negatives) and
false alarms (false positives). For a monitoring dashboard where both change
types carry meaningful but not wildly asymmetric costs (missing a real change
delays action; a false alarm wastes an analyst's review time), this balanced
operating point is a reasonable default. The bonus multi-threshold dashboard
(`bonus/multi_threshold_dashboard.py`) additionally exposes high-recall and
high-precision operating points for users who want to shift this trade-off.

Region pairs with cosine similarity below **0.469** are
flagged as **changed**.
