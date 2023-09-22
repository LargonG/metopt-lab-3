import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches
import sys
import time
from memory_profiler import profile

plt.rcParams["figure.figsize"] = (10, 10)


def gradient(x, y, w, batch_size):
    return (-2) * np.dot(x.T, (y - np.dot(x, w))) / len(y)


def Kramer_method(x, y):
    sx = sum(x)
    sy = sum(y)
    list_xy = []
    [list_xy.append(x[i] * y[i]) for i in range(len(x))]
    sxy = sum(list_xy)
    list_x_sq = []
    [list_x_sq.append(x[i] ** 2) for i in range(len(x))]
    sx_sq = sum(list_x_sq)
    n = len(x)
    det = sx_sq * n - sx * sx
    det_a = sx_sq * sy - sx * sxy
    a = (det_a / det)
    det_b = sxy * n - sy * sx
    b = (det_b / det)
    return [round(a, 4), round(b, 4)]


points = np.array([[0, 0]])

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 10, 10, 3, 3])
y = np.array(
    [-6.0, -7.0, -7.5, -8.2, -10.0, -11.7, -12.5, -13.0, -13.5, -15.3, -16.0, -16.4, -17.5, -18.4, -20.0, -21.0, -22.3,
     -22.4, -24.5, -25.9, -10.3, -20.3, -3.5, -11.5])


def loss_function(x, y, w):
    return np.sum((y - np.dot(x, w)) ** 2)


@profile(precision=4)
def sgd(grad, x, y, w_init, eps=0.01, learning_rate=0.01, batch_size=1,
        max_iter=100):
    start = time.time()
    w = w_init
    kramer = Kramer_method(x, y)
    points = [w_init]
    epochs = 0
    mem = 0
    data = {}
    arifm_counter = 0
    idshuffle = random.sample(list(range(len(y))), len(y))
    for _ in range(max_iter):
        idarr = np.random.randint(len(y), size=batch_size)

        x_partial = np.array(list(map(lambda i: np.array([1, x[idshuffle[i % len(y)]]]), idarr)))
        y_partial = np.array(list(map(lambda i: y[idshuffle[i % len(y)]], idarr)))

        g = gradient(x_partial, y_partial, w, batch_size)
        w = w - learning_rate * gradient(x_partial, y_partial, w, batch_size)
        points.append(w)
        epochs += 1

        if np.isnan(np.sum(w)) or np.isinf(np.sum(w)):
            break

        if np.linalg.norm(kramer - w) < eps:
            break

        arifm_counter += 2 + 2 * (len(x) * 2 - 1) + len(x) + len(w) * 2 + 1

    timer = time.time() - start
    data = {'memory_usage': sys.getsizeof(points),
            'actions': arifm_counter,
            'time': timer,
            'epochs': epochs,
            'points': points}

    return w, data


@profile(precision=4)
def momentum(grad, x, y, w_init, eps=0.01, learning_rate=0.01, b=0.01, batch_size=1,
             max_iter=100):
    start = time.time()
    w = w_init
    kramer = Kramer_method(x, y)
    points = [w_init]
    epochs = 0
    mem = 0
    arifm_counter = 0
    idshuffle = random.sample(list(range(len(y))), len(y))
    v = np.array([0, 0])
    for _ in range(max_iter):
        idarr = np.random.randint(len(y), size=batch_size)

        x_partial = np.array(list(map(lambda i: np.array([1, x[idshuffle[i % len(y)]]]), idarr)))
        y_partial = np.array(list(map(lambda i: y[idshuffle[i % len(y)]], idarr)))

        g = gradient(x_partial, y_partial, w, batch_size)
        v = b * v + (1 - b) * g
        w = w - learning_rate * v

        points.append(w)
        epochs += 1

        if np.isnan(np.sum(w)) or np.isinf(np.sum(w)):
            return np.nan

        if np.linalg.norm(kramer - w) <= eps:
            break

        arifm_counter += 2 + 2 * (len(x) * 2 - 1) + len(x) + len(w) * 2 + 1

    timer = time.time() - start
    data = {'memory_usage': sys.getsizeof(points),
            'actions': arifm_counter,
            'time': timer,
            'epochs': epochs,
            'points': points}
    return w, data


@profile(precision=4)
def nesterov(grad, x, y, w_init, eps=0.1, learning_rate=0.01, b=0.01, batch_size=1,
             max_iter=100):
    start = time.time()
    w = w_init
    kramer = Kramer_method(x, y)
    points = [w_init]
    epochs = 0
    mem = 0
    arifm_counter = 0
    idshuffle = random.sample(list(range(len(y))), len(y))
    v = np.array([0, 0])
    for _ in range(max_iter):
        idarr = np.random.randint(len(y), size=batch_size)

        x_partial = np.array(list(map(lambda i: np.array([1, x[idshuffle[i % len(y)]]]), idarr)))
        y_partial = np.array(list(map(lambda i: y[idshuffle[i % len(y)]], idarr)))

        g = gradient(x_partial, y_partial, w - learning_rate * b * w, batch_size)

        v = b * v + (1 - b) * g
        w = w - learning_rate * v

        points.append(w)
        epochs += 1

        if (np.isnan(np.sum(w)) or np.isinf(np.sum(w))):
            return np.nan
        if (np.linalg.norm(w - kramer) <= eps):
            break
        arifm_counter += 2 + 2 * (len(x) * 2 - 1) + len(x) + len(w) * 2 + 1
        timer = time.time() - start
        data = {'memory_usage': sys.getsizeof(points),
                'actions': arifm_counter,
                'time': timer,
                'epochs': epochs,
                'points': points}
    return w, data


@profile(precision=4)
def ada_grad(grad, x, y, w_init, eps=0.1, learning_rate=0.01, batch_size=1,
             max_iter=100):
    start = time.time()
    w = w_init
    kramer = Kramer_method(x, y)
    points = [w_init]
    epochs = 0
    mem = 0
    arifm_counter = 0
    idshuffle = random.sample(list(range(len(y))), len(y))
    v = np.array([0, 0])
    G = np.zeros(2)
    for _ in range(max_iter):
        idarr = np.random.randint(len(y), size=batch_size)

        x_partial = np.array(list(map(lambda i: np.array([1, x[idshuffle[i % len(y)]]]), idarr)))
        y_partial = np.array(list(map(lambda i: y[idshuffle[i % len(y)]], idarr)))

        g = gradient(x_partial, y_partial, w, batch_size)

        G += g ** 2
        w = w - learning_rate * g / (np.sqrt(G))

        points.append(w)
        epochs += 1

        if (np.isnan(np.sum(w)) or np.isinf(np.sum(w))):
            return np.nan
        if (np.linalg.norm(kramer - w) <= eps):
            break
        arifm_counter += 2 + 2 * (len(x) * 2 - 1) + len(x) + len(w) * 2 + 1
        timer = time.time() - start
        data = {'memory_usage': sys.getsizeof(points),
                'actions': arifm_counter,
                'time': timer,
                'epochs': epochs,
                'points': points}
    return w, data


@profile(precision=4)
def RMSProp(grad, x, y, w_init, eps=0.1, learning_rate=0.01, b=0.01, batch_size=1,
            max_iter=100):
    start = time.time()
    w = w_init
    kramer = Kramer_method(x, y)
    points = [w_init]
    epochs = 0
    mem = 0
    arifm_counter = 0
    s = np.array([0, 0])
    idshuffle = random.sample(list(range(len(y))), len(y))
    for _ in range(max_iter):
        idarr = np.random.randint(len(y), size=batch_size)
        x_partial = np.array(list(map(lambda i: np.array([1, x[idshuffle[i % len(y)]]]), idarr)))
        y_partial = np.array(list(map(lambda i: y[idshuffle[i % len(y)]], idarr)))
        g = gradient(x_partial, y_partial, w - learning_rate * b * w, batch_size)
        s = b * s + (1 - b) * g * g
        w = w - learning_rate * g / ((s + 0.0001) ** (1 / 2))
        points.append(w)
        epochs += 1
        if (np.isnan(np.sum(w)) or np.isinf(np.sum(w))):
            return np.nan
        if (np.linalg.norm(kramer - w) <= eps or epochs == 5000):
            break
        arifm_counter += 2 + 2 * (len(x) * 2 - 1) + len(x) + len(w) * 2 + 1
        timer = time.time() - start
        data = {'memory_usage': sys.getsizeof(points),
                'actions': arifm_counter,
                'time': timer,
                'epochs': epochs,
                'points': points}
    return w, data


@profile(precision=4)
def adam(grad, x, y, w_init, eps=0.1, learning_rate=0.01, b1=0.01, b2=0.01, batch_size=1,
         max_iter=100):
    start = time.time()
    w = w_init
    kramer = Kramer_method(x, y)
    points = [w_init]
    epochs = 0
    mem = 0
    arifm_counter = 0
    v = np.array([0, 0])
    s = np.array([0, 0])
    idshuffle = random.sample(list(range(len(y))), len(y))
    k = 0
    for _ in range(max_iter):
        idarr = np.random.randint(len(y), size=batch_size)
        x_partial = np.array(list(map(lambda i: np.array([1, x[idshuffle[i % len(y)]]]), idarr)))
        y_partial = np.array(list(map(lambda i: y[idshuffle[i % len(y)]], idarr)))
        g = gradient(x_partial, y_partial, w, batch_size)
        v = b1 * v + (1 - b1) * g
        s = b2 * s + (1 - b2) * g * g
        v_ = np.empty([len(v)])
        s_ = np.empty([len(s)])
        v_ = v / (1 - b1 ** (k + 1))
        s_ = s / (1 - b2 ** (k + 1))
        w = w - learning_rate * v_ / ((s_ + 0.00001) ** (1 / 2))
        points.append(w)
        k += 1
        epochs += 1
        if (np.linalg.norm(kramer - w) <= eps or epochs == 5000):
            break
        arifm_counter += 2 + 2 * (len(x) * 2 - 1) + len(x) + len(w) * 2 + 1
        timer = time.time() - start
        data = {'memory_usage': sys.getsizeof(points),
                'actions': arifm_counter,
                'time': timer,
                'epochs': epochs,
                'points': points}
    return w, data

# if __name__ == "__main__":
#     num = 100
#     w_init = np.array([0, 0])
#     kramer = Kramer_method(x, y)
#     arr = dict()
#     arr['sgd'] = list()
#     arr['momentum'] = list()
#     arr['nesterov'] = list()
#     arr['ada_grad'] = list()
#     arr['rmsprop'] = list()
#     arr['adam'] = list()
#     batch = 1
#     for i in range(num):
#         print(i)
#         points = np.array([[0, 0]])
#         data = sgd(x, y, [0, 0], eps = 0.05, learning_rate=0.0055, batch_size=batch)[1]
#         arr["sgd"].append(data)
#         points = np.array([[0, 0]])
#         data = momentum(x, y, [0, 0], eps=0.05, learning_rate=0.009, b=0.6, batch_size=batch)[1]
#         arr["momentum"].append(data)
#         points = np.array([[0, 0]])
#         data = nesterov(x, y, w_init, eps=0.05, learning_rate=0.009, b=0.6, batch_size=batch)[1]
#         arr['nesterov'].append(data)
#         points = np.array([[0, 0]])
#         data = ada_grad(x, y, w_init, eps=0.05, learning_rate=0.9, batch_size=batch)[1]
#         arr['ada_grad'].append(data)
#         points = np.array([[0, 0]])
#         data = RMSProp(x, y, w_init, eps=0.05, learning_rate=0.012, b = 0.5, batch_size=batch)[1]
#         arr['rmsprop'].append(data)
#         points = np.array([[0, 0]])
#         data = adam(x, y, w_init, eps=0.05, learning_rate=0.012, b1 = 0.65, b2 = 0.65, batch_size=batch)[1]
#         arr['adam'].append(data)
#     table = np.empty((6, 4))
#     table.fill(0)
#     a = 0
#     for i in arr:
#         for j in arr[i]:
#             b = 0
#             for k in j:
#                 table[a][b] += j[k]
#                 b += 1
#         a += 1
#     table = table.T
#     table /= num
#     for i in range(0, 4):
#         for j in range(0, 6):
#             print(table[i][j], end=' ')
#         print()
#
#     arr = dict()
#     arr['sgd'] = list()
#     arr['momentum'] = list()
#     arr['nesterov'] = list()
#     arr['ada_grad'] = list()
#     arr['rmsprop'] = list()
#     arr['adam'] = list()
#     batch = 5
#     for i in range(num):
#         print(i)
#         points = np.array([[0, 0]])
#         data = sgd(x, y, [0, 0], eps = 0.05, learning_rate=0.0055, batch_size=batch)[1]
#         arr["sgd"].append(data)
#         points = np.array([[0, 0]])
#         data = momentum(x, y, [0, 0], eps=0.05, learning_rate=0.009, b=0.6, batch_size=batch)[1]
#         arr["momentum"].append(data)
#         points = np.array([[0, 0]])
#         data = nesterov(x, y, w_init, eps=0.05, learning_rate=0.009, b=0.6, batch_size=batch)[1]
#         arr['nesterov'].append(data)
#         points = np.array([[0, 0]])
#         data = ada_grad(x, y, w_init, eps=0.05, learning_rate=0.9, batch_size=batch)[1]
#         arr['ada_grad'].append(data)
#         points = np.array([[0, 0]])
#         data = RMSProp(x, y, w_init, eps=0.05, learning_rate=0.012, b = 0.5, batch_size=batch)[1]
#         arr['rmsprop'].append(data)
#         points = np.array([[0, 0]])
#         data = adam(x, y, w_init, eps=0.05, learning_rate=0.012, b1 = 0.65, b2 = 0.65, batch_size=batch)[1]
#         arr['adam'].append(data)
#     table = np.empty((6, 4))
#     table.fill(0)
#     a = 0
#     for i in arr:
#         for j in arr[i]:
#             b = 0
#             for k in j:
#                 table[a][b] += j[k]
#                 b += 1
#         a += 1
#     table = table.T
#     table /= num
#     for i in range(0, 4):
#         for j in range(0, 6):
#             print(table[i][j], end=' ')
#         print()
#
#     arr = dict()
#     arr['sgd'] = list()
#     arr['momentum'] = list()
#     arr['nesterov'] = list()
#     arr['ada_grad'] = list()
#     arr['rmsprop'] = list()
#     arr['adam'] = list()
#     batch = len(y)
#     for i in range(num):
#         print(i)
#         points = np.array([[0, 0]])
#         data = sgd(x, y, [0, 0], eps = 0.05, learning_rate=0.0055, batch_size=batch)[1]
#         arr["sgd"].append(data)
#         points = np.array([[0, 0]])
#         data = momentum(x, y, [0, 0], eps=0.05, learning_rate=0.009, b=0.6, batch_size=batch)[1]
#         arr["momentum"].append(data)
#         points = np.array([[0, 0]])
#         data = nesterov(x, y, w_init, eps=0.05, learning_rate=0.009, b=0.6, batch_size=batch)[1]
#         arr['nesterov'].append(data)
#         points = np.array([[0, 0]])
#         data = ada_grad(x, y, w_init, eps=0.05, learning_rate=0.9, batch_size=batch)[1]
#         arr['ada_grad'].append(data)
#         points = np.array([[0, 0]])
#         data = RMSProp(x, y, w_init, eps=0.05, learning_rate=0.012, b = 0.5, batch_size=batch)[1]
#         arr['rmsprop'].append(data)
#         points = np.array([[0, 0]])
#         data = adam(x, y, w_init, eps=0.05, learning_rate=0.012, b1 = 0.65, b2 = 0.65, batch_size=batch)[1]
#         arr['adam'].append(data)
#     table = np.empty((6, 4))
#     table.fill(0)
#
#     a = 0
#     for i in arr:
#         for j in arr[i]:
#             b = 0
#             for k in j:
#                 table[a][b] += j[k]
#                 b += 1
#         a += 1
#
#     table = table.T
#     table /= num
#     for i in range(0, 4):
#         for j in range(0, 6):
#             print(table[i][j], end=' ')
#         print()
