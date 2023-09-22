import random
import numpy as np

from solution import models
from solution import regression as reg


def lin_grad(x, y, point):
    r = reg.regression(models.lin_test, x, y)
    return reg.grad(x, models.lin_jac, r)(point)


def partial(x, y, batch_size):
    count = len(y)
    ids = [random.randint(0, count - 1) for _ in range(batch_size)]

    res_x = np.array(list(map(lambda i: x[i], ids)))
    res_y = np.array(list(map(lambda i: y[i], ids)))

    return res_x, res_y
