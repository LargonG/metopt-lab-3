import numpy as np
from numpy.linalg import norm


def lbfgs(func, grad, generations, start, t=0.5, eps=1e-4, max_iter=100):
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

        delta_x = -z
        delta_y = g - old_g
        tau = delta_y @ delta_x
        old_g = g

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
            i = gens - (it + 1)
            alpha[i] = rho[i] * (s[i] @ q)
            q = q - alpha[i] * y[i]

        gamma = (s[-1] @ y[-1]) / (y[-1] @ y[-1])
        H = gamma * np.eye(dim)
        z = H @ q

        for i in range(gens):
            beta = rho[i] * (y[i] @ z)
            z = z + s[i] * (alpha[i] - beta)

    return point
