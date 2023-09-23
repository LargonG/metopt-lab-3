import tracemalloc

import numpy as np
from numpy.linalg import norm
import solution.optimize.line_search.wolfe as descent
import time
from memory_profiler import profile
from solution.tests import ProcInfo


@profile(precision=4)
def bfgs(func, grad, init_point, eps=1e-4, max_iter=100):
    trace = [init_point]

    iters = 0
    actions = 0

    tracemalloc.start()
    start_time = time.time()

    point = init_point  # row
    H = np.eye(len(point))  # matrix
    E = np.eye(len(point))  # matrix

    for it in range(max_iter):
        g = grad(point)  # row
        p = -H.dot(g)  # row

        iters += 1
        if norm(g) < eps or norm(p) < eps:
            break

        # too slow
        # sol = solve(simplify(func(a * p + point)).diff(), a)
        # alpha = float(sol[0])

        alpha = descent.find_wolfe_lr(func, grad, point, p, 1000)

        if norm(alpha * p) < eps:
            break

        new_point = point + alpha * p  # row + const * row = row
        new_grad = grad(new_point)  # row

        s = alpha * p  # scalar * row = row
        y = new_grad - g  # row - row = row

        tau = (y @ s)

        rho = 1 / tau  # scalar

        s_2d = np.atleast_2d(s).transpose()  # col
        y_2d = np.atleast_2d(y).transpose()  # col
        H = ((E - rho * s_2d.dot(y_2d.transpose()))
             .dot(H)
             .dot(E - rho * y_2d.dot(s_2d.transpose())) + rho * s_2d.dot(s_2d.transpose()))

        point = point + s

        trace.append(point)

    end_time = time.time()
    memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return point, ProcInfo(time=end_time - start_time,
                           memory=memory,
                           points=trace,
                           arithmetic=actions,
                           iterations=iters
                           )
