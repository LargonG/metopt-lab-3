import numpy as np
from numpy.linalg import inv, norm


def gauss_newton(func, jacobian, start, eps, max_iter):
    param = np.array(start)
    for i in range(max_iter):
        old_param = param

        J = jacobian(param)

        d = inv(J.transpose() @ J) @ J.transpose() @ func(param)
        param = old_param - d  # NOTE: + -- if regression else -

        if norm(d) < eps:
            break

    return param
