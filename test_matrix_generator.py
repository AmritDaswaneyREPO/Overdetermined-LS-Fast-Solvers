import numpy as np
from numpy.linalg import norm
import random
import math
import scipy
import os


def overdetermined_ls_test_matrix_generator(m, n, diagonal=None, fill_diagonal_method='one large', condition_number=None, a_cond=24, distribution_fn=np.random.uniform, a=0, b=1, seed=123):
    """
    Test matrix generator

    Keyword arguments:

    m: Number of rows of A
    n: Number of columns of A
    distribution_fn: Distribution function from which to draw entries from
    a: alpha
    b: beta
    diagonal: User-defined diagonal entries
    fill_diagonal_method: Method to use when filling diagonal {'one large', 'one small', 'geometric', 'arithmetic', 'loguniform', 'random'}
    condition_number: Condition Number
    a_cond: We pick condition_number with log_2 k distributed uniformly in [0, a_cond]. Defaults to 24, assuming machine precision is 2^{-24}

    Returns: A (m x n matrix), b (n x 1 matrix)

    TO-DO:

    1. Add Householders Transformation / Given Rotations from Lawn 9

    """
    
    np.random.seed(seed)
    # ----------------------------------------------------------------------------------------------------------------------------
    # 1. Generate condition_number: Randomly pick a condition number k with log_2 k distributed uniformly in [0, a_cond]
    # ----------------------------------------------------------------------------------------------------------------------------
    if condition_number is None:
        condition_number = 2 ** np.random.uniform(0, a_cond, 1)[0]

    # #----------------------------------------------------------------------------------------------------------------------------
    # # 1. Generate initial matrix if one is not specified
    # #----------------------------------------------------------------------------------------------------------------------------
    # if A is None:
    #     A = distribution_fn(a, b, n*m).reshape(n, m)
    # else:
    #     (n, m) = A.shape

    # ----------------------------------------------------------------------------------------------------------------------------
    # 2. Set diagonal based on user specification
    # ----------------------------------------------------------------------------------------------------------------------------
    # If specified, use input from user.
    # if diagonal is not None:
    #     np.fill_diagonal(A, diagonal)

    N = min(m, n)

    # If fill_diagonal_method = 'one large', set D(1) = 1 and D(i) = 1 / condition_number for i > 1
    if fill_diagonal_method == 'one large':
        diagonal = np.array([1] + [1 / condition_number] * (n - 1))

    # If fill_diagonal_method = 'one small', set D(N) = 1 / condition_number  and D(i) = 1  for i > 1
    elif fill_diagonal_method == 'one small':
        diagonal = ([1] * (N - 1)) + [1 / condition_number]

    # If fill_diagonal_method = 'geometric', set D(i) form a geometric sequence from 1 to 1/COND
    elif fill_diagonal_method == 'geometric':
        # N = min(m, n). We want r^{N-1} = 1 / condition_number => r = (1 / condition_number)^{1/N-1}
        r = (1 / condition_number) ** (1 / (N - 1))
        diagonal = np.array([r ** i for i in range(N)])

    # If fill_diagonal_method = 'arithmetic', set D(i) form a geometric sequence from 1 to 1/COND
    elif fill_diagonal_method == 'arithmetic':
        # We want 1 + d(N-1) = 1 / condition_number => d = (1 - COND) / COND * (N-1)
        d = (1 - condition_number) / (condition_number * (N - 1))
        diagonal = np.array([1 + (d * i) for i in range(N)])

    # If fill_diagonal_method = 'loguniform', then D(i) are random in the range [1 / COND, 1] with uniformly distributed logarithms
    elif fill_diagonal_method == 'loguniform':
        diagonal = np.exp(np.random.uniform(1 / condition_number, 1, N))

    # if fill_diagonal_method == 'random':
    #     diagonal = np.diag(A)
    # else:
    #     np.fill_diagonal(A, diagonal)

    # ----------------------------------------------------------------------------------------------------------------------------
    # 3. Generate A
    # ----------------------------------------------------------------------------------------------------------------------------
    # Swap largest diagonal entry with sigma_11 and smallest with sigma_kk
    k = random.sample([3, math.floor(n / 2), n], 1)[0]

    max_idx = np.argmax(diagonal)
    if max_idx != 0:
        diagonal[0], diagonal[max_idx] = diagonal[max_idx], diagonal[0]

    min_idx = np.argmin(diagonal)
    if min_idx != k:
        diagonal[k - 1], diagonal[min_idx] = diagonal[min_idx], diagonal[k - 1]

    U, _ = np.linalg.qr(np.random.standard_normal((m, n)))
    if n == k:
        V, _ = np.linalg.qr(np.random.standard_normal((n, n)))
    else:
        V_1, _ = np.linalg.qr(np.random.standard_normal((k, k)))
        V_2, _ = np.linalg.qr(np.random.standard_normal((n - k, n - k)))
        V = np.concatenate((np.concatenate((V_1, np.zeros([k, n - k])), axis=1), np.concatenate((np.zeros([n - k, k]), V_2), axis=1)), axis=0)

    A = (U*diagonal).dot(V.T)

    # Q_1 = ortho_group.rvs(n)

    # if n == m and check_symmetric(A):
    #     A = np.matmul(np.matmul(Q_1, A), Q_1.T)
    # else:
    #     Q_2 = ortho_group.rvs(m)
    #     A = np.matmul(np.matmul(Q_1, A), Q_2)

    # ----------------------------------------------------------------------------------------------------------------------------
    # 3. Generate b
    # ----------------------------------------------------------------------------------------------------------------------------
    # Select random x. b_1 = Ax then normalize b_1 so that ||b_1||_2 = 1
    x = np.random.rand(n, 1)
    b_1 = np.matmul(A, x)
    b_1 = b_1 / np.sqrt(np.sum(b_1 ** 2))

    # A = QR. b_2 = d - QQ.T d. Normalize b_2
    Q = np.linalg.qr(A)[0]
    d = np.random.uniform(-1, 1, (m, 1))
    b_2 = d - np.matmul(np.matmul(Q, Q.T), d)
    b_2 = b_2 / np.sqrt(np.sum(b_2 ** 2))

    # Generate initial theta
    u = np.random.uniform(-26, -1, 1)[0]
    theta = math.pi * (2 ** u)
    # Flip theta with probability 0.5
    if np.random.uniform(0, 1, 1)[0] >= 0.5:
        theta = (math.pi / 2) - theta
    # Calculate final b
    b = (b_1 * math.cos(theta)) + (b_2 * math.sin(theta))

    # Calculate true x
    x = np.matmul(np.linalg.pinv(A), b)

    return A, x, b


def create_fast_solver_test_matrix(n, d, path, save_matrices=False, condition_number=400, sparse=False, density=0.25):
    if sparse:
        A = scipy.sparse.random(n, d, density=density).toarray()
    else:
        # Calculate sigma based on condition number required.
        sigma = round(1 / (condition_number ** (1 / (d - 1))), 2)

        # 1. Create test matrix A
        U, _ = np.linalg.qr(np.random.standard_normal((n, d)))
        V, _ = np.linalg.qr(np.random.standard_normal((d, d)))
        s = [sigma ** (i + 1) for i in range(d)]
        A = (U * s).dot(V.T)
        if n < d:
            A = A.T

    # 2. Create x
    x = np.random.normal(loc=0, scale=1 / d, size=d).reshape(d, 1)

    # 3. Sample a response vector b ~ N(Ax, I_n)
    b = np.random.multivariate_normal(np.matmul(A, x).reshape(n, ), np.eye(n)).reshape(n, 1)

    # 4. True x
    true_x = np.matmul(np.linalg.pinv(A), b)

    # Option to save generated matrices.
    if save_matrices:
        np.savetxt(os.path.join(path, 'A_{}_{}.txt'.format(n, d)), A, fmt='%d')
        np.savetxt(os.path.join(path, 'x_{}_{}.txt'.format(n, d)), true_x, fmt='%d')
        np.savetxt(os.path.join(path, 'b_{}_{}.txt'.format(n, d)), b, fmt='%d')

    return A, true_x, b

