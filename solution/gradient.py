import numpy as np
import sympy as sm
from sympy import Symbol


def analytic(func, dim):
    def run(point):
        mp = dict()
        it = 0
        for s in symbols:
            mp[s.name] = point[it]
            it += 1
        return np.array([grad[i].subs(mp) for i in range(dim)])

    symbols = [Symbol('a' + str(i)) for i in range(dim)]
    grad = [sm.diff(func(symbols), s) for s in symbols]

    return run


def calc(func, delta=1e-4):
    def at_index(i, n):
        vector = np.zeros(n)
        vector[i] = delta
        return vector

    def run(point):
        dim = len(point)
        f = func(point)
        return np.array([(func(point + at_index(i, dim)) - f) / delta for i in range(dim)])

    return run
