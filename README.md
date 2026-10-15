# Improving the Weisfeiler–Lehman Kernel with Log-Binned Degree Labels and an RBF Kernel

**Course:** Machine Learning with Graphs (23CSE479) — Amrita Vishwa Vidyapeetham  
**Dataset:** REDDIT-BINARY (2,000 graphs, TUDataset format)

---

## Team

| Name | Roll Number |
|---|---|
| Sidharth S Nair | CB.SC.U4CSE23443 |
| Yash Bardia | CB.SC.U4CSE23458 |
| S Sanjeev Srinivas | CB.SC.U4CSE23466 |
| Karthik P | CB.SC.U4CSE23522 |

**Faculty Supervisor:** Ramraj T

---

## Overview

We re-implement the Weisfeiler–Lehman (WL) subtree kernel from scratch (following Hamilton, *Graph Representation Learning* §2.1.2 & §7.3) and improve it with two targeted modifications — **log-binned degree labels** and an **RBF kernel** on WL histograms — without changing the WL algorithm itself.

### Results (10-fold × 3 nested CV)

| Model | Accuracy |
|---|---|
| Graph size only (baseline) | 77.33% |
| Textbook WL (degree labels + linear kernel) | 81.75% ± 2.46 |
| + log-binned degree labels | 88.10% |
| **Ours: log-binned labels + RBF kernel** | **90.72% ± 1.83** |

**+9.0 pp** over the textbook kernel; **49% fewer errors** (corrected resampled t-test, p ≈ 3×10⁻¹¹).

---

## Repository Structure

```
.
├── WL_Kernel_REDDIT_BINARY.ipynb   # Full study: theory, implementation, experiments, figures
├── data/raw/REDDIT-BINARY/         # Raw dataset (TUDataset format)
├── results/                        # CSV tables produced by the notebook
├── figures/                        # PNG figures produced by the notebook
├── report/Report.pdf               # Term project report
└── requirements.txt
```

---

## Experiments

1. **Initial label strategies** — uniform, degree, log-binned degree, clustering coefficient, PageRank, combined
2. **Update rule & depth** — Hamilton Eq. 2.6 vs. Eq. 7.60 (provably equivalent for degree labels), depth h = 0…5
3. **Kernel choice** — linear, cosine, histogram intersection (WL-OA), RBF

---

## Setup & Run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace WL_Kernel_REDDIT_BINARY.ipynb
```

> A full run takes approximately **50–60 minutes** on a 16-core machine.

---

## References

- Hamilton (2020), *Graph Representation Learning*
- Shervashidze et al., JMLR 2011
- Kriege et al., NeurIPS 2016
- Yanardag & Vishwanathan, KDD 2015
- Morris et al., TUDataset 2020
- Nadeau & Bengio, 2003
