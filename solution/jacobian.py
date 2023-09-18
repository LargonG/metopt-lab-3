import numpy as np


def one_value(shape, index, value) -> np.ndarray:
    res = np.zeros(shape)
    res[index] = value
    return res


def by_index(n: int, i: int, eps: float = 1e-6) -> callable:
    return lambda function: \
        lambda x: (
                (function(one_value((n,), (i,), x + eps)) -
                 function(one_value((n,), (i,), x))) / eps)


def jacobi(functions: np.ndarray,
           derivatives: np.ndarray) -> np.ndarray:
    res = np.array([[derivative(func) for derivative in derivatives] for func in functions])
    return res


def jacobi(n: int,
           functions: np.ndarray) -> np.ndarray:
    derivatives = np.array([by_index(n, i) for i in range(0, n)])
    return jacobi(functions, derivatives)
