import numpy as np
from numpy.linalg import norm
import time

from solution.tests import ProcInfo
from memory_profiler import profile


@profile(precision=4)
def lbfgs(func, grad, generations, start, t=0.5, eps=1e-4, max_iter=100):
    trace = [start]

    iters = 0
    actions = 0

    start_time = time.time()

    dim = len(start)
    point = start
    old_g = grad(point)
    alpha = np.zeros(generations)

    s = []
    y = []
    rho = []

    z = t * old_g
    for k in range(max_iter):
        point = point - z
        g = grad(point)

        trace.append(point)

        delta_x = -z
        delta_y = g - old_g
        tau = delta_y @ delta_x
        old_g = g

        actions += 6

        iters += 1

        if norm(g) < eps:
            break

        if tau > eps or k == 0:
            if k >= generations:
                s = s[1:]
                y = y[1:]
                rho = rho[1:]

            s.append(delta_x)
            y.append(delta_y)
            rho.append(1. / tau)

        q = g
        gens = len(s)
        for it in range(gens):
            actions += 4
            i = gens - (it + 1)
            alpha[i] = rho[i] * (s[i] @ q)
            q = q - alpha[i] * y[i]

        gamma = (s[-1] @ y[-1]) / (y[-1] @ y[-1])
        H = gamma * np.eye(dim)
        z = H @ q

        actions += 4

        for i in range(gens):
            actions += 5
            beta = rho[i] * (y[i] @ z)
            z = z + s[i] * (alpha[i] - beta)

    end_time = time.time()

    return point, ProcInfo(time=end_time - start_time,
                           memory=None,
                           points=trace,
                           arithmetic=actions,
                           iterations=iters
                           )
