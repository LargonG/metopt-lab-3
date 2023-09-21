import numpy as np
import util
from models import funcs, funcs_params


class ProcInfo:
    def __init__(self, time, memory, points, arithmetic):
        self.time = time
        self.memory = memory
        self.points = points
        self.generations = len(points)
        self.arithmetic = arithmetic


class Test:
    def __init__(self, model, init_point, count, func, begin, end, steps):
        self.init_point = init_point[:count]
        self.func = func
        self.X, self.Y = util.generate_test(func, begin, end, steps)
        self.model = model
        self.begin = begin
        self.end = end
        self.steps = steps


intervals = [
    (0, 5, 20)
]

init_points = [
    np.ones(5),
    np.array([1., 2., 3., 4., 5.]),
    np.array([100., 100., 100., 100., 100.]),
    np.array([-100., -100., -100., -100., -100])
]

real_functions = [
    np.sin,
    np.cos,
    np.exp,
    lambda x: sum(x ** i for i in range(10))
]


def create_tests():
    return [Test(funcs[name], point, funcs_params[name], real, begin, end, steps)
            for name in funcs
            for (begin, end, steps) in intervals
            for point in init_points
            for real in real_functions]


