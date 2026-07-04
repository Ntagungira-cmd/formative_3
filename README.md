# Probability, Bayesian Inference & Gradient Descent — Formative 3

> **Course:** Mathematics & Machine Learning — Year 2  
> **Assignment:** Formative 3  
> **Submission date:** July 2026

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Team Members & Contributions](#team-members--contributions)
3. [Repository Structure](#repository-structure)
4. [Part 1 — Probability Distributions (EM Algorithm)](#part-1--probability-distributions-em-algorithm)
5. [Part 2 — Bayesian Probability (IMDb Sentiment)](#part-2--bayesian-probability-imdb-sentiment)
6. [Part 3 — Gradient Descent Manual Calculations](#part-3--gradient-descent-manual-calculations)
7. [Part 4 — Gradient Descent in Code](#part-4--gradient-descent-in-code)
8. [Setup & Running the Notebooks](#setup--running-the-notebooks)
9. [Datasets](#datasets)

---

## Project Overview

This assignment investigates three foundational pillars of machine learning mathematics:

| # | Topic | Approach |
|---|-------|----------|
| 1 | Gaussian Mixture Models | EM algorithm from scratch on the Galton Families dataset (Fathers vs. Children heights) |
| 2 | Bayesian Inference | Naïve Bayes sentiment classification on the IMDb Movie Reviews dataset |
| 3 | Gradient Descent (Manual) | Handwritten step-by-step derivation of MSE gradients and parameter updates |
| 4 | Gradient Descent (Code) | Python implementation using matrix operations + SciPy + Matplotlib |

All code is written from scratch — **no ML libraries** are used for learning algorithms. Every calculation step is kept explicit and visible.

---

## Team Members

| Name |
|------|
| **Ntagungira Ali Rashid**|
| **Mildred Ebomah**|
| **Gabriella Ange Ahirwe**|
| **Sheja Queen Linda**|

---

## Repository Structure

```
formative_3/
│
├── README.md                        ← this file
│
├── my_probability_project.ipynb     ← Part 1: EM Algorithm (Gaussian Mixture Model)
├── Bayesian_Probability.ipynb       ← Part 2: Bayesian Probability (IMDb)
├── part4_gradient_descent.ipynb     ← Part 4: Gradient Descent in Code
├── PART 3 GRADIENT.pdf              ← Part 3: Handwritten manual calculations (PDF)
|__ formative_3_presentation.pdf     ← PDF file with all the collabs merged into one.
│
└── data/
    ├── GaltonFamilies.csv           ← Heights dataset (Fathers & Children)
    └── IMDB Dataset.csv             ← IMDb movie reviews dataset
```

---

## Part 1 — Probability Distributions (EM Algorithm)

**Notebook:** `my_probability_project.ipynb`  
**Dataset:** `data/GaltonFamilies.csv` — Galton Families heights (Fathers vs. Children)

### What it does

The dataset contains heights for two distinct populations — **Fathers** and **Children** — but we deliberately ignore the labels and treat the data as an unlabelled mixture of two Gaussian distributions. The goal is to recover both distributions from scratch.

### Algorithm: Expectation-Maximization (EM)

The EM algorithm iterates between two steps:

**E-Step (Expectation)** — compute the *posterior responsibility* each Gaussian component claims for every data point

**M-Step (Maximisation)** — update the parameters using the responsibilities computed above

Both steps repeat until the **log-likelihood** converges.

### Tracking Table (first two iterations)

The notebook prints the following table at runtime. The values below are representative; exact values are shown during the live presentation:

| Iteration | μ₁ (Children) | μ₂ (Fathers) | σ₁² | σ₂² | π₁ | π₂ | Log-Likelihood |
|-----------|--------------|-------------|-----|-----|----|----|----------------|
| 0 (Init)  | _printed_    | _printed_   | _printed_ | _printed_ | _printed_ | _printed_ | _printed_ |
| 1         | _printed_    | _printed_   | _printed_ | _printed_ | _printed_ | _printed_ | _printed_ |
| 2         | _printed_    | _printed_   | _printed_ | _printed_ | _printed_ | _printed_ | _printed_ |

> Run the notebook to populate the table with exact values.

## Part 2 — Bayesian Probability (IMDb Sentiment)

**Notebook:** `Bayesian_Probability.ipynb`  
**Dataset:** `data/IMDB Dataset.csv`

### Keyword Selection

| Sentiment | Keywords chosen |
|-----------|----------------|
| **Positive** | `brilliant`, `excellent`, `amazing`, `great` |
| **Negative** | `awful`, `boring`, `terrible`, `waste` |

### Probability computed: P(Positive \| keyword)

For each keyword the notebook computes and displays:

| Quantity | Formula |
|----------|---------|
| Prior | P(Positive) |
| Likelihood | P(keyword \| Positive) |
| Marginal | P(keyword) |
| **Posterior** | **P(Positive \| keyword) = P(keyword \| Positive) × P(Positive) / P(keyword)** |

### Implementation constraints

- **No ML libraries** (no scikit-learn, NLTK, etc.)  
- Only `pandas` for data loading and basic Python arithmetic for probability computation.  
- All Bayes formula steps are kept explicit.

---

## Part 3 — Gradient Descent Manual Calculations

**File:** `PART 3 GRADIENT.pdf`

| Parameter | Initial value |
|-----------|--------------|
| **m** (weights) | `[-1, 2]` |
| **b** (bias) | `[1, 1]` |
| **X** (inputs) | `[[1, 3], [4, 10]]` |
| **y** (targets) | `[5, 6]` |

The PDF contains the complete handwritten walkthrough for **all four iterations** (one per group member), showing intermediate predictions, residuals, and updated parameters at each step.

### Observed trend

After each update, both **m** and **b** shift in the direction that reduces the residual error — the predicted values ŷ move closer to the targets y, confirming that gradient descent is converging towards the optimal parameters.

---

## Part 4 — Gradient Descent in Code

**Notebook:** `part4_gradient_descent.ipynb`

### What it implements

1. **Matrix-form forward pass** — computes predictions $\hat{y} = X\mathbf{m} + \mathbf{b}$
2. **Explicit gradient computation** — no autograd, every derivative step is visible
3. **SciPy derivative verification** — uses `scipy.misc.derivative` / `scipy.optimize` to cross-check analytical gradients
4. **Parameter update loop** — runs for 4 iterations (one per group member), printing m, b, and MSE at each step
5. **Matplotlib plots:**
   - How **m** and **b** evolve over iterations
   - How the **MSE error** decreases over iterations

### Sample output structure

```python
Iteration 0 | m=[-1.  2.] | b=[1. 1.] | MSE=XX.XXXX
Iteration 1 | m=[...    ] | b=[...  ] | MSE=XX.XXXX
Iteration 2 | m=[...    ] | b=[...  ] | MSE=XX.XXXX
Iteration 3 | m=[...    ] | b=[...  ] | MSE=XX.XXXX
```

---

## Setup & Running the Notebooks

### Prerequisites

```bash
pip install numpy pandas matplotlib scipy jupyter
```

### Running locally

```bash
git clone https://github.com/Ntagungira-cmd/formative_3.git
cd formative_3
jupyter notebook
```

Then open the relevant notebook in your browser:

| Notebook | Topic |
|----------|-------|
| `my_probability_project.ipynb` | Part 1 — EM Algorithm |
| `Bayesian_Probability.ipynb` | Part 2 — Bayesian Probability |
| `part4_gradient_descent.ipynb` | Part 4 — Gradient Descent |

> The notebooks were also tested on **Google Colab**. Use the "Open in Colab" badge inside `Bayesian_Probability.ipynb` to run it in the cloud without any local setup.

---

## Datasets

| File | Source | Description |
|------|--------|-------------|
| `data/GaltonFamilies.csv` | [Galton Families dataset](https://vincentarelbundock.github.io/Rdatasets/csv/HistData/GaltonFamilies.csv) | Heights (inches) of parents and their adult children from Francis Galton's 1885 study |
| `data/IMDB Dataset.csv` | [Kaggle — IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 50,000 labelled movie reviews (positive / negative) |

---
