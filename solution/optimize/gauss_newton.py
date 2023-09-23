import tracemalloc

import numpy as np
import time
import sys
from numpy.linalg import inv, norm
from memory_profiler import memory_usage, profile

from solution.tests import ProcInfo


@profile(precision=4)
def gauss_newton(func, jacobian, start, eps, max_iter):
    trace = [start]

    actions = 0
    iter = 0

    tracemalloc.start()
    start_time = time.time()

    param = np.array(start)
    for i in range(max_iter):
        old_param = param

        J = jacobian(param)

        d = inv(J.transpose() @ J) @ J.transpose() @ func(param)
        param = old_param - d  # NOTE: + -- if regression else -

        trace.append(param)
        actions += 1

        iter += 1
        if norm(d) < eps:
            break

    end_time = time.time()

    memory = tracemalloc.get_traced_memory()
    print(memory)
    tracemalloc.stop()

    return param, ProcInfo(time=end_time - start_time,
                           memory=memory,
                           points=trace,
                           arithmetic=actions,
                           iterations=iter
                           )
