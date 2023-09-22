import time

import numpy as np
from numpy.linalg import norm
from memory_profiler import profile

from solution.optimize.linear.util import partial, lin_grad
from solution.tests import ProcInfo


@profile
def adam(x, y, start,
         b1=0.9, b2=0.99,
         eps=1e-8, learning_rate=0.01, batch_size=1,
         max_iter=100):
    trace = [start]
    iters = 0
    start_time = time.time()

    dim = len(start)
    point = start

    G = np.zeros(dim)
    v = np.zeros(dim)

    for iters in range(max_iter):
        x_partial, y_partial = partial(x, y, batch_size)
        g = lin_grad(x_partial, y_partial, point)

        v = b1 * v - (1 - b1) * g
        G = b2 * G + (1 - b2) * (g ** 2)
        delta = - learning_rate * v / (np.sqrt(G) + eps)
        point = point + delta

        trace.append(point)

        if norm(delta) < eps:
            break

    end_time = time.time()

    return point, ProcInfo(time=end_time - start_time,
                           points=trace,
                           iterations=iters)
