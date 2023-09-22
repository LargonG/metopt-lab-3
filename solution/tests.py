import numpy as np
import util
from models import funcs, funcs_params
import sys
from memory_profiler import profile


class ProcInfo:
    def __init__(self,
                 time=0,
                 memory=None,
                 points=None,
                 arithmetic=None,
                 iterations=0):
        if points is None:
            points = []

        self.time = time
        self.memory = memory
        self.points = points
        self.generations = len(points)
        self.arithmetic = arithmetic
        self.iterations = iterations


def matrix_multiplication(A, B):
    return len(A) * len(A[0]) * len(B)


def matrix_inv(A):
    return len(A) * len(A[0]) * len(A)


def matrix_square(A):
    return len(A) * len(A[0])


def create_info(start_time, end_time, points, arithmetic):
    return ProcInfo(end_time - start_time, sys.getsizeof(points), points, arithmetic)


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
    lambda x: sum(x ** i for i in range(3))
]


def create_tests():
    return [Test(funcs[name], point, funcs_params[name], real, begin, end, steps)
            for name in funcs
            for (begin, end, steps) in intervals
            for point in init_points
            for real in real_functions]


def create_test(func,
                model_name='exp',
                point=np.array([1., 1., 1., 1., 1.]),
                params=funcs_params['exp'],
                begin=0, end=5, steps=50):
    return Test(funcs[model_name], point, params, func, begin, end, steps)
