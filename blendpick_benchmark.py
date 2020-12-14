A, x, b = overdetermined_ls_test_matrix_generator(4000,100)

t0 = perf_counter()
blendenpik = Blendenpik(A, b)
x1 = blendenpik.solve()
# x1 = LSRN1(A,b)
t1 = perf_counter() - t0
print("Residual (L2-norm):", np.linalg.norm(x1-x,ord =2))
print("Computational time (sec.) for the Blendenpik algorithm:",t3)

t2 = perf_counter()
x2 = np.linalg.lstsq(A,b,rcond=None)[0]
t3 = perf_counter() - t2
print("Residual (L2-norm):", np.linalg.norm(x2-x,ord =2))
print("Computational time (sec.) for numpy least-squares algorithm:",t3)
