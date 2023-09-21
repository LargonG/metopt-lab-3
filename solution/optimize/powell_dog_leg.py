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
import sys
from memory_profiler import profile
import numpy as np
from numpy.linalg import inv, norm
import time

from solution.tests import ProcInfo


@profile(precision=4)
def powell_dog_leg(func, grad, H, init_point, delta, eps, max_iter):
    trace = [init_point]

    # memory = 0
    actions = 0
    iter = 0

    start_time = time.time()

    point = init_point

    for i in range(max_iter):
        g = grad(point)
        B = H(point)
        p = -inv(B).dot(g)

        # memory += sys.getsizeof(g) + sys.getsizeof(B) * 2 + sys.getsizeof(p)

        if norm(p) > delta:
            pu = -g.dot(g) / g.dot(B.dot(g)) * g
            # memory += sys.getsizeof(pu)

            if norm(pu) <= delta:
                D = (delta * norm(pu - p)) ** 2 + p.dot(pu) ** 2 - (norm(pu) * norm(p)) ** 2
                s = (p.dot(p - pu) - np.sqrt(D)) / norm(pu - p) ** 2
                p = s * pu + (1 - s) * p

                # memory += sys.getsizeof(D) + sys.getsizeof(s) + sys.getsizeof(p)
            else:
                p = pu * delta / norm(pu)

                # memory += sys.getsizeof(p)

        point = point + p

        # memory += sys.getsizeof(point)
        trace.append(point)

        iter += 1
        if norm(p) < eps:
            break

    end_time = time.time()

    return point, ProcInfo(time=end_time - start_time,
                           memory=None,
                           points=trace,
                           arithmetic=actions,
                           iterations=iter
                           )
