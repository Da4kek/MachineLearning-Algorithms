# Principal Component Analysis
---

## The main steps behind `PCA`:

* `pca` or principal component analysis is an unsupervised linear transformation technique that is widely used across different fields.
* other popular applications of `pca` include exploratory data analyses and the denoising of signals in stock market trading and the analysis of genome data and gene expression levels in the field of bioformatics.

 > * `pca` helps us to identify patterns in data based on the correlation between features.
 > * `pca` aims to find the directions of maximum variance in high-dimensional data and projects the data onto a new subspace with equal or fewer dimensions than the original one.

 ## Approaching `pca`:

 * Standardize the d-dimensional dataset.
 * Construct the covariance matrix.
 * Decompose the covariance matrix into its eigenvectors and eigenvalues.
 * Sort the eigenvalues by decreasing order to rank the corresponding eigenvectors.
 * select `k` eigenvectors, which correspond to the `k` largest eigenvalues, where `k` is the dimensionality of the new feature subspace (k <= d).
 * Construct a projection matrix **W** , from the top `k` eigenvectors.
 * Transform the d-dimensional input dataset, **X**, using the projection matrix,**W** to obtain the new k-dimensional feature subspace.