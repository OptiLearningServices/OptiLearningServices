# Optimization & Machine Learning

Welcome! This profile presents my work on **optimization methods**, **machine learning**, and **sustainable energy systems**.

## Focus Areas
- **Optimization**: existence and uniqueness of solutions, convex analysis, numerical methods
- **Machine learning**: neural networks with Keras / TensorFlow
- **Applied mathematics & control**: modeling and analysis of dynamical systems, with applications to sustainable energy

## Tools
Python · NumPy · Keras · TensorFlow · MATLAB · LaTeX · Git
# Decision Trees: A Mathematical Formulation

## Intuition

A decision tree searches for a partition of the feature space $\mathcal{X}$ such that the values of $Y$ within each cell of the partition are as close as possible to one another. In other words, it looks for regions that are **homogeneous with respect to $Y$**.

## Setting

Let $(x_i, y_i)_{i=1,\dots,n}$ be a training sample, with $x_i \in \mathcal{X} \subset \mathbb{R}^p$ and $y_i \in \mathbb{R}$.

## Regression

The tree seeks a partition $\mathcal{P} = \{R_1, \dots, R_K\}$ of $\mathcal{X}$, i.e.

$$
\mathcal{X} = \bigsqcup_{k=1}^{K} R_k, \qquad R_k \cap R_l = \emptyset \quad (k \neq l),
$$

that minimizes the within-region variability of $Y$:

$$
\min_{\mathcal{P}} \ \sum_{k=1}^{K} \sum_{i \,:\, x_i \in R_k} \left(y_i - \bar{y}_{R_k}\right)^2,
\qquad
\bar{y}_{R_k} = \frac{1}{|R_k|} \sum_{i \,:\, x_i \in R_k} y_i .
$$

This is the **within-region (intra-class) variance**. The resulting predictor is piecewise constant:

$$
\hat f(x) = \sum_{k=1}^{K} \bar{y}_{R_k} \, \mathbf{1}_{\{x \in R_k\}} .
$$

## Structural constraint: recursive binary splitting

Minimizing over all possible partitions is combinatorial (NP-hard). The tree therefore restricts the search to partitions obtained by **recursive axis-aligned binary splits**. At a node $t$ with region $R_t$, we choose a variable $j$ and a threshold $s$:

$$
R_t^{-}(j,s) = \{x \in R_t : x_j \le s\}, \qquad
R_t^{+}(j,s) = \{x \in R_t : x_j > s\},
$$

with

$$
(j^*, s^*) = \arg\min_{j,\,s} \left[
\sum_{i : x_i \in R_t^{-}(j,s)} \left(y_i - \bar{y}_{R_t^-}\right)^2
+ \sum_{i : x_i \in R_t^{+}(j,s)} \left(y_i - \bar{y}_{R_t^+}\right)^2
\right].
$$

The algorithm is **greedy**: one split is optimized at a time, so the result is not globally optimal.

## Classification

If $Y \in \{1,\dots,C\}$, the variance is replaced by an impurity measure $H$ (Gini index or entropy):

$$
\min_{\mathcal{P}} \ \sum_{k=1}^{K} \frac{n_k}{n} \, H(R_k),
\qquad
H_{\text{Gini}}(R_k) = \sum_{c=1}^{C} \hat p_{kc}\,(1 - \hat p_{kc}),
$$

where $\hat p_{kc}$ is the proportion of class $c$ in $R_k$ and $n_k = |\{i : x_i \in R_k\}|$.

## Summary

> A decision tree builds a partition $\{R_1,\dots,R_K\}$ of the covariate space $\mathcal{X}$ that minimizes the within-region heterogeneity of $Y$, measured by $\sum_{k}\sum_{x_i\in R_k}(y_i-\bar y_{R_k})^2$. The regions $R_k$ are homogeneous zones with respect to $Y$, obtained through successive binary splits $\{x_j \le s\}$ / $\{x_j > s\}$.
## Projects
-  *Coming soon*: comparison of optimization algorithms (gradient descent, Newton, constrained methods)
-  *Coming soon*: neural networks on MNIST and CIFAR-10

## Contact
📧 [optilearningservices@gmail.com](mailto:optilearningservices@gmail.com)
