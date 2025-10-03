# SHAP Example: Continuous and Categorical Features

This example illustrates how SHAP (SHapley Additive exPlanations) works with a simple model that includes both a continuous feature and a categorical feature.

---

## Model Setup
We predict **house price** (\(f(x)\)) with two features:

- **Size** (continuous, in square feet)  
- **Location** (categorical: Urban = 1, Suburban = 0)  

The model is defined as:

f(Size, Location) = 50 + 0.1 Size + 20 Location

- Baseline = 50 (intercept)  
- Each additional square foot adds \$0.10  
- Urban location adds \$20,000  

---

## Example House
- **Size = 2000**  
- **Location = Urban (1)**  

Prediction:

\[
f(2000, 1) = 50 + 0.1(2000) + 20(1) = 270
\]

---

## Step 1: Subsets
We want the **Shapley value for Size**.  
Subsets of features are:

- {} (no features)  
- {Size}  
- {Location}  
- {Size, Location}  

---

## Step 2: Expected Predictions
We need expectations over missing features.  
Suppose in our dataset:  
- Average Size = 1500  
- Average Location = 0.5 (half Urban)

- **{} (empty set):**  
  \[
  \mathbb{E}[f] = 50 + 0.1(1500) + 20(0.5) = 215
  \]

- **{Size} only (Size = 2000):**  
  \[
  f_{\{Size\}}(2000) = 50 + 0.1(2000) + 20(0.5) = 260
  \]

- **{Location} only (Urban = 1):**  
  \[
  f_{\{Location\}}(1) = 50 + 0.1(1500) + 20(1) = 220
  \]

- **{Size, Location} (both known):**  
  \[
  f_{\{Size, Location\}}(2000, 1) = 270
  \]

---

## Step 3: Shapley Value for Size
- Contribution when added to {}:  
  \(260 - 215 = 45\)  
- Contribution when added to {Location}:  
  \(270 - 220 = 50\)  

Average:  
\[
\phi_{\text{Size}} = \frac{1}{2}(45 + 50) = 47.5
\]

---

## Step 4: Shapley Value for Location
- Contribution when added to {}:  
  \(220 - 215 = 5\)  
- Contribution when added to {Size}:  
  \(270 - 260 = 10\)  

Average:  
\[
\phi_{\text{Location}} = \frac{1}{2}(5 + 10) = 7.5
\]

---

## Step 5: Verify Additivity
SHAP guarantees:

\[
\text{Baseline (215)} + \phi_{\text{Size}} + \phi_{\text{Location}} = \text{Prediction (270)}
\]

Check:  
\[
215 + 47.5 + 7.5 = 270 \quad \checkmark
\]

---

## Takeaway
- Even though **Size** is continuous, SHAP excludes it by averaging over its distribution (using mean Size = 1500).  
- Contributions are computed by comparing predictions with vs. without the feature, across subsets, then averaged.  
- SHAP ensures additivity: baseline + contributions = actual prediction.  
