# Linear Discriminant Analysis

---

> LDA can be used as a technique for feature extraction to increase the computational efficiency and reduce the degree of overfitting due to the curse of dimensionality in non-regularized models.

## Principal component analysis vs Linear discriminant analysis:

> * Both PCA and LDA are linear transformation techniques that can be used to reduce the number of dimensions in a dataset; the former is an unsupervised algorithm, whereas the latter is supervised.
> * PCA reduces the number of dimensions by finding the maximum variance in high dimensional data.
> * The goal of LDA is to find a feature subspace that best optimizes class separability.   

## Approach of LDA:

1. Standardize the d-dimensional dataset (d is the number of features).
2. For each class, compute the d-dimensional mean vector.
3. Construct the between-class scatter matrix, **S***B*, and the within-class scatter matrix,**S***w*.
4. Compute the eigenvectors and corresponding eigenvalues of the matrix.
5. Sort the eigenvalues by decreasing order to rank the corresponding eigenvectors. 
6. Choose the k eigenvectors that correspond to the k largest eigenvalues to construct a d*x*k-dimensional transformation matrix, **W**; the eigenvectors are the columns of this matrix.
7. Project the examples onto the new feature subspace using the transformation matrix, **W**.