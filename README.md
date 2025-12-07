# Multimin

This is a package for multimineral inversion from well logs. The goal is to estimate volume fractions of Quartz, Calcite, Dolomite, Porosity (fluid). 

This is a collaborative project between me and my student Mariam Alawadi, that revisits multimineral inversion using Python and replicate the inner working of modern softwares like Techlog. 

<img width="1300" height="1000" alt="newplot (5)" src="https://github.com/user-attachments/assets/08053dcb-6e8c-49c7-baba-84b60f584eac" />

There are 2 options of inversion: 

1. **Constrained Least Squares**  
2. **Weighted Constrained Least Squares**

---

## How to run? 🚀

Multimin supports LAS file. An example well log from Volve `15_9-F-11A.LAS` is given in folder `data`

Run: 

```python
uv run python log_example.py
```

---

## Concept

If you keen about the mathematical concept, read below: 

### 1) What the model assumes 🧱

At one depth, the logs are modeled as a linear mix of endpoints:

$$
\mathbf{d} \approx \mathbf{G}\mathbf{x}
$$

where

- $\mathbf{d} = [DT,\ RHOB,\ NPHI]^T$  
- $\mathbf{x} = [x_Q,\ x_C,\ x_D,\ x_\phi]^T$

The kernel $\mathbf{G}$ is built from mineral/fluid endpoints.

---

### 2) Why a closure equation is added 🧩

There are **3 logs** but **4 unknowns**, so the system needs an extra physical rule:

- **Volumes must sum to 1**

This is written as:

$$
x_Q + x_C + x_D + x_\phi = 1
$$

An augmented kernel form is:

$$
\mathbf{G}_{\text{aug}}=
\begin{bmatrix}
\mathbf{G}\\
1\ 1\ 1\ 1
\end{bmatrix},
\qquad
\mathbf{d}_{\text{aug}}=
\begin{bmatrix}
\mathbf{d}\\
1
\end{bmatrix}
$$

A plain least-squares solve of this can still give **negative volumes**, which are not physical.  
That motivates constrained methods.

---

### 3) Constrained Least Squares (CLS) ✅

#### What it enforces 📌

CLS aims to find a solution that is **physically valid**:

- non-negative volumes  
- sum-to-one (closure)

In short:

$$
\min_{\mathbf{x}} \ \|\mathbf{G}'\mathbf{x}-\mathbf{d}'\|^2
\quad
\text{s.t.}\quad
\mathbf{x}\ge 0,\ \sum x_i = 1
$$

#### Why normalization is used ⚙️

To keep DT, RHOB, and NPHI on similar numerical scale, each row is scaled using simple factors:

- $DT/100$  
- $RHOB/1$  
- $NPHI/0.5$

These are **conditioning factors**, not guaranteed maxima.  
They help balance how strongly each log influences the fit.

#### How the custom solver works 🧮

The custom method is:

- projected gradient descent  
- with backtracking  
- and **Nesterov acceleration** 🚀

The important idea:

1. take a step that reduces misfit  
2. **project back** to the valid volume space:

$$
x_i \ge 0,\quad \sum x_i = 1
$$

**Nesterov is used** to speed convergence.

---

### 4) Weighted Constrained LS (W-CLS) ⚖️

Sometimes certain logs are more reliable.

Weights increase the importance of trusted logs in the misfit:

$$
\min_{\mathbf{x}} \ \|\mathbf{W}(\mathbf{G}'\mathbf{x}-\mathbf{d}')\|^2
\quad
\text{s.t.}\quad
\mathbf{x}\ge 0,\ \sum x_i = 1
$$

A typical default choice is:

- $[1,\ 2,\ 1]$ for **[DT, RHOB, NPHI]**

Meaning:

- RHOB gets more influence 🧱

This is useful when density is stable and sonic/neutron are noisier.

---

### 5) Depth-by-depth inversion 🧭

Solving independently at each depth is standard practice:

$$
\mathbf{x}(z) = \arg\min_{\mathbf{x}\in\Delta}
\|\mathbf{G}'\mathbf{x}-\mathbf{d}'(z)\|^2
$$

If results look noisy, light post-smoothing across depth is a common next step.

---
