import numpy as np
import template as example


def exp_test(x):
    return lambda betta: betta[0] * np.exp(betta[1] * x)


def exp_jac(x):
    return lambda betta: np.array(
        [-np.exp(betta[1] * x), -betta[0] * x * np.exp(betta[1] * x)]
    ).transpose()


def polynomial_test(n):
    return lambda x: example.polynomial(n, x)


def polynomial_jac(n):
    return lambda x: lambda betta: np.array([-x ** i for i in range(n)]).transpose()


def square_test(x):
    return lambda betta: betta[0] * x ** 2 + betta[1] * x + 1


def square_jac(x):
    return lambda betta: np.vstack((-x ** 2, -x)).transpose()


funcs_array = [
    exp_test, polynomial_test, square_test
]

jacs_array = [
    exp_jac, polynomial_jac, square_jac
]

n = 5
funcs = {
    "exp": (exp_test, exp_jac),
    "pol": (polynomial_test(n), polynomial_jac(n)),
    "square": (square_test, square_jac)
}
