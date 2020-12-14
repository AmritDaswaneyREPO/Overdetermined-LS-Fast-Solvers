import numpy as np
from numpy.linalg import norm
import time
from sketching_matrices import gaussian_random_sketch, subsampled_randomized_hadamard_transforms, sparse_johnson_lindenstrauss_transform


def calculate_hessian_using_sketch(A, m, sketch, sketch_fn, sketch_time, factor_time):
    """

    Args:
        A: Data matrix
        m: sketch size
        sketch: Takes string input = {Gaussian, SRHT, Sparse JL}. Defaults to Gaussian
        sketch_fn: user-defined sketch function
        sketch_time: Total sketch time taken already
        factor_time: Total factor time taken already

    Returns: pseudo-inverse(A^TS^TSA), total sketch time, total factor time.

    """
    # 1. Generate sketching matrix S_t
    sketch_start_time = time.time()
    if sketch_fn is not None:
        S_t = sketch_fn(A, m, 0)
    else:
        if sketch == 'SRHT':
            S_t = subsampled_randomized_hadamard_transforms(m, A.shape[0])
        elif sketch == 'Uniform':
            S_t = (1 / (m ** 0.5)) * np.random.uniform(-1, 1, size=(m, A.shape[0]))
        elif sketch == 'Sparse JL':
            s = int(round(((m / A.shape[1]) ** 0.5) * (np.log10(A.shape[1] ** 2) ** 3), 0))
            S_t = sparse_johnson_lindenstrauss_transform(m, A.shape[0], s)
        else:
            S_t = gaussian_random_sketch(m, A.shape[0], 0, 1 / (m ** 0.5))

    # 2. Compute Sketched matrix
    S_A = np.matmul(S_t, A)
    sketch_time += time.time() - sketch_start_time

    # 3. Compute approximate Hessian
    factor_start_time = time.time()
    H_t = np.matmul(S_A.T, S_A)
    H_t = np.linalg.pinv(H_t)
    factor_time += time.time() - factor_start_time
    return H_t, sketch_time, factor_time


def get_optimal_step_size_and_momentum_parameters(sketch, fixed, momentum, n, d, m):
    """
        As defined in:
         1. Jonathan Lacotte, Mert Pilanci, Optimal Randomized First-Order Methods for Least-Squares Problems (https://arxiv.org/abs/2002.09488)
         2. Jonathan Lacotte, Mert Pilanci, Faster Least Squares Optimization (https://arxiv.org/pdf/1911.02675.pdf)
        Args:
            sketch: Sketching matrix used
            fixed: Whether sketching matrix is redrawn at every turn
            momentum: Whether heavy-ball is used or not
            n: Number of rows of data matrix
            d: Number of columns of data matrix
            m: Sketch size

        Returns: step size and momentum parameter

        """
    if sketch == 'SRHT':
        if fixed:
            if momentum:
                alpha = m / d
                step_size = (2 * (alpha - 1)) / (alpha + ((alpha ** 2 - alpha) ** 0.5))
                # beta = ((alpha ** 0.5) - ((alpha - 1) ** 0.5)) / (alpha ** 0.5) + ((alpha - 1) ** 0.5)
                xi, gamma = m / n, d / n
                lambda_h = (((1 - gamma) * xi) ** 0.5 - ((1 - xi) * gamma) ** 0.5) ** 2
                Lambda_h = (((1 - gamma) * xi) ** 0.5 + ((1 - xi) * gamma) ** 0.5) ** 2
                # step_size = 4 / ((1 / Lambda_h)**0.5 + (1 / lambda_h)**0.5)**2
                beta = ((Lambda_h ** 0.5 - lambda_h ** 0.5) / (Lambda_h ** 0.5 + lambda_h ** 0.5)) ** 2
            else:
                beta = 0
                step_size = 1 - d/m
        else:
            beta = 0
            xi, gamma = m/n, d/n
            step_size = (xi - gamma)**2 / (xi**2 + gamma - (2 * gamma * xi))
    else:
        if fixed:
            if momentum:
                beta = d / m
                step_size = (1 - beta) ** 2
            else:
                beta = 0
                step_size = (4 * m * d) / (m**2 + d**2)
        else:
            beta = 0
            step_size = ((m - d) * (m - d - 3)) / (m * (m - 1))
    return beta, step_size


def iterative_hessian_sketch(A, b, m, x_1, no_of_iterations, sketch='Gaussian', sketch_fn=None, x_0=None, step_size=None, beta=None,
                             iteration_type='vanilla', fixed=False, x_list=None, return_h=False, h_t_pinv=None,
                             total_time=0, sketch_time=0, factor_time=0, epsilon=None, max_iter=50, max_time=None):
    """
    Calculates {iterated, fixed} Hessian Sketch {with, without} momentum

    A: Input m x n matrix
    b: Input n x 1 matrix
    m: Size of sketch
    x_1: Starting point
    no_of_iterations: Number of iterations
    step_size: Step Size
    sketch: Sketch functions present in ristretto
    sketch_fn: Custom sketch function that follows same semantics
    x_0: Required for sketch with momentum
    beta: Beta for sketch with momentum
    iteration_type: {vanilla, heavy_ball}
    fixed: Fixed Hessian or calculated at every iteration
    x_list: List of solutions at each iteration.
    h_t_pinv: For fixed iteration if already calculated then don't calculate again
    return_h: If we are running the first iteration of fixed sketch return h_t_pinv as well

    returns: List of x_1 solutions
    """

    # if iteration_type == 'vanilla' and step_size is None:
    #     k = A.shape[1]
    #     step_size = ((m - k) * (m - k - 3)) / (m * (m - 1))

    opt_beta, opt_step_size = get_optimal_step_size_and_momentum_parameters(sketch, fixed, iteration_type == 'heavy_ball', A.shape[0], A.shape[1], m)
    if step_size is None:
        step_size = opt_step_size
    if beta is None:
        beta = opt_beta

    t0 = time.time()

    if x_list is None:
        x_list = [np.copy(x_1)]

    if fixed and h_t_pinv is None:
        h_t_pinv, sketch_time, factor_time = calculate_hessian_using_sketch(np.copy(A), m, sketch, sketch_fn, sketch_time, factor_time)
    has_converged = False
    iteration = 0

    # # Define constants for fixed SRHT with momentum
    # if iteration_type == 'heavy_ball' and fixed and sketch == 'SRHT':
    #     kappa, omega, c = define_constants_for_fixed_srht_with_momentum(A.shape[0], A.shape[1], m)
    # else:
    #     kappa, omega, c = None, None, None

    while not has_converged:
        # 1. Get pseudoinverse of H
        if not fixed:
            h_t_pinv, sketch_time, factor_time = calculate_hessian_using_sketch(np.copy(A), m, sketch, sketch_fn, sketch_time, factor_time)

        # 2. Compute gradient
        g_t = np.matmul(A.T, (np.matmul(A, x_1) - b))

        # 3. If heavy ball, deep copy x_1 so we can assign it to x_0 after update
        if iteration_type == 'heavy_ball' and beta != 0:
            x_1_temp = np.copy(x_1)

        # # If fixed SRHT, we need to calculate beta and step_size at every iteration
        # if iteration_type == 'heavy_ball' and fixed and sketch == 'SRHT':
        #     u, beta, step_size = get_step_size_for_fixed_srht_with_momentum(u, kappa, omega, c)

        # 4. Perform update
        x_1 -= step_size * (np.dot(np.copy(h_t_pinv), g_t))

        # 5. If heavy ball, add momentum
        if iteration_type == 'heavy_ball' and beta != 0:
            x_1 += beta * (x_1_temp - x_0)
            x_0 = np.copy(x_1_temp)

        x_list.append(np.copy(x_1))

        # 6. Check if convergence condition is met
        iteration += 1
        if epsilon is None:
            has_converged = iteration == no_of_iterations
        else:
            relative_error = norm(x_list[-1] - x_list[-2], ord=2) / norm(x_list[-1], ord=2)
            if max_time is not None:
                has_converged = ((time.time() - t0) > max_time) or (relative_error < epsilon)
            else:
                has_converged = (iteration == max_iter) or ((relative_error < epsilon) and (iteration == no_of_iterations))

    total_time += time.time() - t0
    if return_h:
        return x_list, h_t_pinv, total_time, sketch_time, factor_time
    else:
        return x_list, None, total_time, sketch_time, factor_time
