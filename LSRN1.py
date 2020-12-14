# This algorithm is used when m >> n.
def LSRN1(A,b):
    m = A.shape[0]
    n = A.shape[1]
    #oversample factor gamma
    gamma = 2
    s = int(math.ceil(gamma * n))
    G = np.random.normal(0, 1, (s, m))
    A_new = G @ A;
    U_new,s_new,Vh_new = lalg.svd(A_new,full_matrices=False)
    N = Vh_new @ np.diag(s_new)
    y_new = splalg.lsqr(A @ N, b)[0]
    x_new = N @ y_new
    return x_new
