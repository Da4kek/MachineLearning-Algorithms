# Support Vector Machines
---
*Using the perceptron algorithm, we minimized the misclassification errors.*
*In SVMs out optimization objective is to maximize the margin. The margin is defined as the distance between the separating hyperplane(decision boundary) and the training examples that are closest to this hyperplane, which are so called support vectors.*


![image](https://user-images.githubusercontent.com/61820492/139179958-b932221c-0d04-4c31-9e4e-217a84b01543.png)

---

## Maximum margin intuition
---
> The idea behind having decision boundaries with large margins is that they tend to have a lower generalization error, whereas models with small margins are most prone to overfitting.
---
## Dealing with a nonlinearly separable case using slack variables
---
> The motivation for introducing the slack variable was that the linear constraints needs to be relaxed for nonlinearly separable data to allow the convergence of the optimization in the presence of misclassifications, under appropriate cost penalization.
> Via a variable C, we can then control the penalty for misclassification.Large values of C correspond to large error penalties, whereas we are less strict about misclassification errors if we choose smaller values for C.
---
### Logistic regression versus SVMs:
- In practical classification tasks, linear logistic regression and linear SVMs often yield very similar results.
- Logistic regression tries to maximize the conditional likelihoods of the training data which makes it more pron to outliers than SVMs.
- Logistic regression models can be easily updated, which is attractive when working with streaming data.
---
**Kernel methods:**
> - The basic idea behind `kernel methods` to deal with such linearly inseparable data is to create nonlinear combinations of the original features to project them onto a higher dimensional space via a mapping function, where the data becomes linearly separable.
> - However, one problem with this mapping approach is that the contruction of the new features is computationally very expensive, especially if we are dealing with high-dimensional data.This is where the so-called `kernel trick` comes into play.
> - In order to save the expensive step of calculating this dot product between two points explicitly, we define a so-called `kernel function`.
> - One of the most widely used kernels is the **Radial basis Function(RBF)** kernel, which can simply be called the **Gaussian kernel**.
---
