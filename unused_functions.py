def define_constants_for_fixed_srht_with_momentum(n, d, m):
    """
    As defined in Jonathan Lacotte, Mert Pilanci, Optimal Randomized First-Order Methods for Least-Squares Problems (https://arxiv.org/abs/2002.09488)
    Args:
        n: Number of rows of data matrix
        d: Number of columns of data matrix
        m: Sketch size

    Returns: Constants kappa, omega, c

    """
    gamma = d / n
    xi = m / n

    lambda_h = (((1 - gamma) * xi) ** 0.5 - ((1 - xi) * gamma) ** 0.5) ** 2
    Lambda_h = (((1 - gamma) * xi) ** 0.5 + ((1 - xi) * gamma) ** 0.5) ** 2

    tau = ((Lambda_h ** 0.5 - lambda_h ** 0.5) / (Lambda_h ** 0.5 + lambda_h ** 0.5)) ** 2
    c = 4 / ((1 / Lambda_h) ** 0.5 + (1 / lambda_h) ** 0.5) ** 2
    alpha = (1 - (tau ** 0.5)) ** 2
    beta = (1 + (tau ** 0.5)) ** 2

    omega = 4 / ((beta - c) ** 0.5 + (alpha - c) ** 0.5) ** 2
    kappa = (((beta - c) ** 0.5 - (alpha - c) ** 0.5) / ((beta - c) ** 0.5 + (alpha - c) ** 0.5)) ** 2

    return kappa, omega, c


def get_step_size_for_fixed_srht_with_momentum(u_0, kappa, omega, c):
    """
    As defined in Jonathan Lacotte, Mert Pilanci, Optimal Randomized First-Order Methods for Least-Squares Problems (https://arxiv.org/abs/2002.09488)
    Args:
        u_0: Prior step size
        kappa: Problem specific constant
        omega: Problem specific constant
        c: Problem specific constant

    Returns: Momentum parameter a, Step size b

    """
    eta = 1 + kappa + (omega * c)

    x_1 = (eta / 2) + (((eta ** 2) / 4) - kappa) ** 0.5
    x_2 = (eta / 2) - (((eta ** 2) / 4) - kappa) ** 0.5

    u_1 = (((x_1 - kappa) * x_1) + ((kappa - x_2) * x_2)) / (x_1 - x_2)

    # Calculate step sizes
    a = 1 - ((eta * u_0) / u_1)
    b = - (omega * c * u_0) / u_1
    return u_1, a, b