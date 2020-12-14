import numpy as np
from scipy.linalg import hadamard


def gaussian_random_sketch(m, n, mean=0, variance=1):
    """
    m: Sketch size,
    n: Number of columns of the input matrix,
    mean: mean of gaussian distribution
    variance: variance of gaussian distribution

    returns: An m x n matrix sample from the Gaussian Distribution
    """
    return np.random.normal(loc=mean, scale=variance, size=(m, n))


def subsampled_randomized_hadamard_transforms(m, n):
    """
    m: Sketch size,
    n: Number of rows in the input matrix,

    returns: An m x n matrix subsampled randomized Hadamard transform
    """

    # if n is None:
    #     # Select n such that it is the smallest power of 2 that is larger than m
    #     p = int(round(np.log2(m), 0) + 1)
    #     n = 2 ** p
    #
    # Create the n x n Walsh-Hadamard matrix:
    H = hadamard(n)

    # Calculate diagonal - uniformly randomly 1 or -1
    d = np.random.uniform(0, 1, size=(n,))
    d = np.where(d > 0.5, 1, -1)

    # Use dot product to calculate HD
    HD = (H * d)

    # Calculate row sampling matrix
    # I_n = np.eye(n)
    # R = I_n[np.random.choice(n, m, replace=False), :]

    # Put everything together with 1 matrix multiplication
    # S = (1 / (m ** 0.5)) * np.matmul(R, HD)
    S = (1 / (m ** 0.5)) * HD[np.random.choice(n, m, replace=True), :]
    # H = hadamard(n)
    # b = np.random.binomial(1, m / n, size=(n,))
    # d = np.random.uniform(0, 1, size=(n,))
    # d = np.where(d > 0.5, 1, -1)
    # S = np.random.permutation(((b * H) * d).T)
    # S = S[np.where(S.any(axis=1))[0], :]

    return S


def sparse_johnson_lindenstrauss_transform(m, n, s):
    """
    m: Sketch size,
    n: Number of rows in the input matrix,
    s: Number of elements in each column to be non-zero.

    returns: An m x n matrix Sparse Johnson Lindenstrauss Transforms
    """

    # Calculate 1/sqrt{s} once
    a = 1 / (s ** 0.5)

    # Initialize s to be an m x n matrix filled with zeros
    S = np.zeros((m, n))

    # For each column, uniformly randomly select s rows that will be non-zero. Uniformly randomly assign it to be +/- 1/sqrt{s}
    sparse_array = np.random.uniform(0, 1, size=(s, n))
    sparse_array = np.where(sparse_array > 0.5, a, -1 * a)
    S[np.random.choice(m, size=(s, n)), np.arange(n)] = sparse_array

    return S