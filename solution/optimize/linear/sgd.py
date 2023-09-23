import time
import tracemalloc

from numpy.linalg import norm
from memory_profiler import profile

from solution.optimize.linear.util import partial, lin_grad
from solution.tests import ProcInfo


@profile(precision=4)
def sgd(x, y, start,
        eps=1e-4, learning_rate=0.01, batch_size=1,
        max_iter=100):
    trace = [start]
    iters = 0

    tracemalloc.start()
    start_time = time.time()
    point = start

    for _ in range(max_iter):
        x_partial, y_partial = partial(x, y, batch_size)
        g = lin_grad(x_partial, y_partial, point)

        point = point - learning_rate * g

        trace.append(point)
        iters += 1

        if norm(g) < eps:
            break

    end_time = time.time()
    memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return point, ProcInfo(time=end_time - start_time,
                           memory=memory,
                           iterations=iters,
                           points=trace
                           )
