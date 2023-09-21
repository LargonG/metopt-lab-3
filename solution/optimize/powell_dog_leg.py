"""
Using powell dog leg algorithm to optimize function
@:param func -- function, that would be updated
@:param grad -- gradient of current function
@:param H -- Hessian of current function
@:param init_point -- start position/point
@:param delta -- trust region
@:param eps -- minimal step, if the step is smaller then minimization stops
@:param max_iter -- maximal count of iterations
"""

import numpy as np
from numpy.linalg import inv, norm


def powell_dog_leg(func, grad, H, init_point, delta, eps, max_iter):
    point = init_point

    for i in range(max_iter):
        g = grad(point)
        B = H(point)
        p = -inv(B).dot(g)

        if norm(p) > delta:
            pu = -g.dot(g) / g.dot(B.dot(g)) * g

            if norm(pu) <= delta:
                D = (delta * norm(pu - p)) ** 2 + p.dot(pu) ** 2 - (norm(pu) * norm(p)) ** 2
                s = (p.dot(p - pu) - np.sqrt(D)) / norm(pu - p) ** 2
                p = s * pu + (1 - s) * p
            else:
                p = pu * delta / norm(pu)

        point = point + p

        if norm(p) < eps:
            break

    return point
