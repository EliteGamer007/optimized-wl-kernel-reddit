# Weisfeiler-Lehman Graph Kernels on REDDIT-BINARY

Course project for **23CSE479 – Machine Learning with Graphs**. We re-implement the WL subtree kernel from scratch, following
Hamilton, *Graph Representation Learning* §2.1.2 and §7.3, and improve it with modifications that leave the WL algorithm itself unchanged.

| Model | Accuracy (10-fold × 3, nested CV) |
|---|---|
| Graph size only (reference) | 77.33% |
| **Textbook WL** (degree labels, linear kernel) | **81.75% ± 2.46** |
| + log-binned degree labels | 88.10% |
| **Ours**: log-binned degree labels + RBF kernel on WL histograms | **90.72% ± 1.83** |

That is +9.0 points and 49% fewer errors than the textbook kernel (corrected resampled t-test, p ≈ 3e-11).

## Contents
- `WL_Kernel_REDDIT_BINARY.ipynb`: the full study, with theory, an implementation that writes out every formula, correctness tests,
  three experiments, the final ablation, and figures with explanations
- `data/raw/REDDIT-BINARY/`: the dataset ([TUDataset](https://chrsmrrs.github.io/datasets/))
- `results/*.csv`: every table produced by the notebook
- `figures/*.png`: every figure produced by the notebook

## What is studied
1. **Initial labels:** uniform, degree, log-binned degree, clustering coefficient, PageRank, combined
2. **Update rule and depth:** Hamilton Eq. 2.6 vs Eq. 7.60 (provably identical for degree-based labels), with h = 0…5
3. **Kernel on WL histograms:** linear, cosine, histogram intersection (WL-OA), RBF

## Run
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace WL_Kernel_REDDIT_BINARY.ipynb
```
A full run takes about 50–60 minutes on a 16-core laptop.

## References
Hamilton (2020); Shervashidze et al., JMLR 2011; Kriege et al., NeurIPS 2016; Yanardag & Vishwanathan, KDD 2015;
Morris et al., TUDataset 2020; Nadeau & Bengio, 2003.
