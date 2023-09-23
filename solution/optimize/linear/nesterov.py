import time
import tracemalloc

import numpy as np
from numpy.linalg import norm

from solution.optimize.linear.util import partial, lin_grad
from solution.tests import ProcInfo

from memory_profiler import profile


@profile(precision=4)
def nesterov(x, y, start,
             decay_factor=0.01,
             eps=1e-4, learning_rate=0.01, batch_size=1,
             max_iter=1000):
    trace = [start]
    iters = 0

    tracemalloc.start()
    start_time = time.time()

    dim = len(start)

    point = start
    delta = np.zeros(dim)
    for iters in range(max_iter):
        x_partial, y_partial = partial(x, y, batch_size)
        g = lin_grad(x_partial, y_partial, point + decay_factor * delta)

        delta = decay_factor * delta - learning_rate * g

        point = point + delta

        trace.append(point)

        if norm(g) < eps:
            break

    end_time = time.time()
    memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return point, ProcInfo(time=end_time - start_time,
                           memory=memory,
                           points=trace,
                           iterations=iters)
