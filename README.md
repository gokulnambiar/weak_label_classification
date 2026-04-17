# Weak Label Classification

This project benchmarks AG News topic classification when only a small fraction of gold labels is available and the rest of the supervision comes from simple keyword rules. It keeps the setup deliberately compact: TF-IDF features, a linear SVM, interpretable weak labeling rules, and a lightweight refinement step based on rule precision estimated from the small labeled subset.

## Problem

Weak supervision is cheaper than full annotation, but it introduces label noise and uneven class coverage. The benchmark compares how much performance is lost when gold labels are replaced with weak labels, and how much of that gap can be recovered with a simple calibration step.

## Dataset

The data is the standard AG News benchmark with four classes: `World`, `Sports`, `Business`, and `Sci/Tech`. The download script pulls the original `train.csv` and `test.csv` files directly into `data/` and verifies them with the same MD5 checksums used by the `torchtext` AG News loader.

## Weak Labeling Setup

The weak labels come from readable keyword groups for each class:

- world and politics terms such as `president`, `government`, `war`, and `united nations`
- sports terms such as `match`, `championship`, `coach`, and `world cup`
- business terms such as `stocks`, `earnings`, `bank`, and `federal reserve`
- science and technology terms such as `software`, `internet`, `chip`, and `nasa`

Some articles receive no weak label, and some trigger multiple classes. The raw weak label is the class with the strongest rule vote, while the refinement step reweights rules by their precision on the small labeled subset and filters low-confidence assignments.

## Evaluation

The pipeline trains and compares three models on the AG News test split:

- full supervision with all gold labels
- weak supervision with a 2% gold subset plus keyword-derived labels on the remaining training pool
- refined weak supervision with the same 2% gold subset plus calibrated, filtered weak labels

Saved outputs include scenario metrics, per-class metrics, rule diagnostics, weak-label quality summaries, confusion matrices, a bar chart for the three supervision settings, class balance summaries, and example rows showing weak versus refined labels.

## Results

On the default run with 2,400 gold-labeled training examples and 30,000 TF-IDF features, the benchmark produced these test-set results:

- full supervision: `0.9216` accuracy, `0.9214` macro F1
- weak supervision: `0.7876` accuracy, `0.7857` macro F1
- refined weak supervision: `0.8170` accuracy, `0.8148` macro F1

The weak rules covered `59.3%` of the small labeled subset at `80.7%` precision. After rule calibration and confidence filtering, coverage dropped to `54.6%` but precision improved to `84.3%`. Business was the hardest class under weak supervision, and the refinement step recovered about `2.9` macro F1 points over the raw weak-label model without closing the full gap to gold labels.

## Run

```bash
cd /Users/gokulnambiar/Codex/weak_label_classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data/download_ag_news.py
python main.py
```

## Project Layout

```text
weak_label_classification/
├── data/
├── outputs/
├── src/
│   ├── classifier.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── label_refinement.py
│   ├── visualization.py
│   └── weak_labeling.py
├── main.py
├── README.md
└── requirements.txt
```
