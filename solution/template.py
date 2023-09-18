"""
There is data for analysis optimization algorithms

"""

import numpy as np
import sympy as sm


def combine(combiner, func1, func2):
    return lambda x: combiner(func1(x), func2(x))


def plus(func1, func2):
    return combine(lambda a, b: a + b, func1, func2)


def mul(func1, func2):
    return combine(lambda a, b: a * b, func1, func2)


def random_function(amplitude=1.0):
    return lambda x: np.array([np.random.uniform(-amplitude, amplitude) for _ in range(0, len(x))])


def exp(x: np.ndarray) -> np.ndarray:
    return np.exp(x)


def polynomial(n: int, x: np.ndarray):
    return lambda betta: sum(betta[i] * (x ** i) for i in range(n))


def hyperbola(n: int, const: float = 0):
    return lambda x: 1.0 / (x ** n + const)


def generate_test(func, start=0, end=100, steps=100):
    x = np.linspace(start, end, steps)
    y = func(x)

    return x, y


def show(real_x, real_y, model_x, model_y):
    import matplotlib.pyplot as plt
    plt.plot(real_x, real_y, 'r-')
    plt.plot(model_x, model_y, 'b-')


if __name__ == "__main__":
    print("This is data file")
