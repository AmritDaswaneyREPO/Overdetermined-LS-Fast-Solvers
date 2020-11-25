import numpy as np
from scipy.linalg import hadamard
from numpy.linalg import norm
import matplotlib.pyplot as plt
from operator import add
import datetime
import os


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

    if n is None:
        # Select n such that it is the smallest power of 2 that is larger than m
        p = int(round(np.log2(m), 0) + 1)
        n = 2 ** p

    # Create the n x n Walsh-Hadamard matrix:
    H = hadamard(n)

    # Calculate diagonal - uniformly randomly 1 or -1
    d = np.array([1 if i >= 0.5 else -1 for i in np.random.uniform(size=(n,))])

    # Use dot product to calculate HD
    HD = (H * d)

    # Calculate row sampling matrix
    # I_n = np.eye(n)
    # R = I_n[np.random.choice(n, m, replace=False), :]

    # Put everything together with 1 matrix multiplication
    # S = (1 / (m ** 0.5)) * np.matmul(R, HD)
    S = (1 / (m ** 0.5)) * HD[np.random.choice(n, m, replace=True), :]
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
    S = np.zeros(n * m).reshape(m, n)

    # For each column, uniformly randomly select s rows that will be non-zero. Uniformly randomly assign it to be +/- 1/sqrt{s}
    for i in range(n):
        non_zero_entries_for_column = np.random.choice(range(m), size=s, replace=False)
        for j in non_zero_entries_for_column:
            if np.random.uniform() > 0.5:
                S[j][i] = a
            else:
                S[j][i] = -1 * a

    return S


def calculate_hessian_using_sketch(A, m, sketch, sketch_fn):
    # TODO 2: Add sketches
    # 1. Generate sketching matrix S_t
    if sketch_fn is not None:
        S_t = sketch_fn(A, m, 0)
    else:
        if sketch == 'gaussian':
            S_t = gaussian_random_sketch(m, A.shape[0], 0, (1 / m)**0.5)
        elif sketch == 'SHRT':
            S_t = subsampled_randomized_hadamard_transforms(m, A.shape[0])

    # TODO 3: Can we do better here?
    # 2. Compute Sketched matrix
    S_A = np.matmul(S_t, A)

    # 3. Compute approximate Hessian
    H_t = np.matmul(S_A.T, S_A)
    return np.linalg.pinv(H_t)


def iterative_hessian_sketch(A, b, m, x_1, T, step_size=None, sketch='gaussian', sketch_fn=None, x_0=None, beta=None, iteration_type='vanilla', fixed=False, x_list=None):
    """
    Calculates {iterated, fixed} Hessian Sketch {with, without} moment

    A: Input m x n matrix
    b: Input n x 1 matrix
    m: Size of sketch
    x_1: Starting point
    T: Number of iterations
    step_size: Step Size
    sketch: Sketch functions present in ristretto
    sketch_fn: Custom sketch function that follows same semantics
    x_0: Required for sketch with momentum
    beta: Beta for sketch with momentum
    iteration_type: {vanilla, heavy_ball}
    fixed: Fixed Hessian or calculated at every iteration
    x_list: List of solutions at each iteration.

    returns: List of x_1 solutions
    """

    if iteration_type == 'vanilla' and step_size is None:
        k = A.shape[1]
        step_size = ((m - k) * (m - k - 3)) / (m * (m - 1))

    if x_list is None:
        x_list = [np.copy(x_1)]

    if fixed:
        H_t_pinv = calculate_hessian_using_sketch(A, m, sketch, sketch_fn)

    for t in range(T):
        # 1. Get pseudoinverse of H
        if not fixed:
            H_t_pinv = calculate_hessian_using_sketch(A, m, sketch, sketch_fn)

        # 2. Compute gradient
        g_t = np.matmul(A.T, (np.matmul(A, x_1) - b))

        # 3. If heavy ball, deep copy x_1 so we can assign it to x_0 after update
        if iteration_type == 'heavy_ball' and beta != 0:
            x_1_temp = np.copy(x_1)

        # 4. Perform update
        x_1 -= step_size * (np.matmul(H_t_pinv, g_t))

        # 5. If heavy ball, add momentum
        if iteration_type == 'heavy_ball' and beta != 0:
            x_1 += beta * (x_1_temp - x_0)
            x_0 = np.copy(x_1_temp)

        x_list.append(np.copy(x_1))

    return x_list


def create_fast_solver_test_matrix(n, d, path, save_matrices=False):
    # 1. Create test matrix A
    U, _ = np.linalg.qr(np.random.standard_normal((n, d)))
    V, _ = np.linalg.qr(np.random.standard_normal((d, d)))
    s = [0.97 ** (i + 1) for i in range(d)]
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
        np.savetxt(os.path.join(path, 'A.txt'), A, fmt='%d')
        np.savetxt(os.path.join(path, 'x.txt'), x, fmt='%d')
        np.savetxt(os.path.join(path, 'b.txt'), b, fmt='%d')

    return A, true_x, b


def create_figure(chart_config, x_axis, path, image_name, figure_title):
    # Close any open plots
    plt.clf()

    # Plot graph for each solver
    for key, value in chart_config.items():
        plt.plot(x_axis, value, label=key)

    plt.title(figure_title)
    plt.xlabel('Number of iterations')
    plt.ylabel('n^{-1}||A(x_t - x*)||_2^2')
    plt.yscale("log")
    plt.legend()
    plt.savefig(path + '{}_{}.png'.format(image_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))


def generate_chart_config(A, b, true_x, m, x_0, solver_config, number_of_iterations, number_of_trials):
    chart_config = {}
    for solver in solver_config:
        print(solver['name'])
        avg_error = [0] * (number_of_iterations + 1)
        for trial in range(number_of_trials):
            # print(trial)
            # Run 1 iteration of vanilla iterated hessian sketch to get another point
            x_list = iterative_hessian_sketch(np.copy(A), np.copy(b), m, np.copy(x_0), 1,
                                        step_size=solver['step_size'], sketch=solver['sketch'], beta=solver['beta'], fixed=solver['fixed'])

            # Run (number_of_iterations - 1) iterations
            x_list = iterative_hessian_sketch(np.copy(A), np.copy(b), m, np.copy(x_list[1]), number_of_iterations - 1,
                                        step_size=solver['step_size'], sketch=solver['sketch'], beta=solver['beta'], fixed=solver['fixed'],
                                        x_0=np.copy(x_list[0]), iteration_type='heavy_ball', x_list=x_list[:])

            # Get ||A(x_t - x*)||_2^2 for each iteration
            x_list = [norm(np.matmul(A, x_list[i] - true_x), ord=2) ** 2 for i in range(len(x_list))]

            # For every trial, add the error at every iteration.
            avg_error = list(map(add, avg_error, x_list[:]))

        # For every solver, divide by the (number of trials * n) to get the normalized error.
        chart_config[solver['name']] = [i / (number_of_trials * A.shape[0]) for i in avg_error]
    return chart_config


def main(working_directory, number_of_iterations=20, number_of_trials=50):
    """
    Algorithm, test matrix, step size, and beta taken from: https://arxiv.org/pdf/1911.02675.pdf
    """

    # 1. Get test matrices
    A, true_x, b = create_fast_solver_test_matrix(1024, 10, working_directory)

    # 2. Have initial point
    x_0 = np.random.uniform(size=(A.shape[1], 1))

    # 3. Solver config
    for sketch in ['gaussian', 'SHRT']:
        for alpha in [4, 8, 16, 32]:
            alpha_inverse = 1 / alpha
            m = alpha * A.shape[1]

            if sketch == 'gaussian':
                beta_1, beta_2 = 0,  alpha_inverse
                step_size_1, step_size_2 = (1 - alpha_inverse) ** 2, ((1 - alpha_inverse) ** 2) / (1 + alpha_inverse)
            else:
                shrt_beta = ((alpha ** 0.5) - ((alpha - 1) ** 0.5)) / ((alpha ** 0.5) + ((alpha - 1) ** 0.5))
                shrt_step_size = (2 * (alpha - 1)) / (alpha + ((alpha ** 2 - alpha) ** 0.5))
                beta_1, beta_2 = shrt_beta, shrt_beta
                step_size_1, step_size_2 = shrt_step_size, 1 - alpha_inverse

            solver_config = [
                {'name': 'Refreshed {}'.format(sketch), 'sketch': sketch, 'fixed': False, 'beta': beta_1, 'step_size': step_size_1},
                {'name': 'Fixed {}'.format(sketch), 'sketch': sketch, 'fixed': True, 'beta': 0, 'step_size': step_size_2},
                {'name': 'Fixed Mom. {}'.format(sketch), 'sketch': sketch, 'fixed': True, 'beta': beta_2, 'step_size': step_size_2},
            ]

            # 4. Generate chart config
            chart_config = generate_chart_config(np.copy(A), np.copy(b), np.copy(true_x), m, np.copy(x_0), solver_config, number_of_iterations, number_of_trials)

            # 5. Generate figure
            x_axis = range(number_of_iterations + 1)
            create_figure(chart_config, x_axis, working_directory, 'Comparison_with_{}_sketch_alpha_{}'.format(sketch, alpha), 'Figure 2: {} Sketch for alpha = {}'.format(sketch, alpha))
