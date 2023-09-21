from solution.tests import ProcInfo


def to_normal_function(legacy_function, grad):
    def run(x, y, init_point, eps=0.01, learning_rate=0.01, batch_size=1, max_iter=100):
        point, data = legacy_function(
            to_legacy_gradient(grad),
            x, y, init_point,
            eps, learning_rate, batch_size,
            max_iter)
        return point, to_normal_observer(data)

    return run


def to_legacy_gradient(grad):
    return lambda x, y, w, batch_size: grad(w)


def to_normal_observer(data):
    return ProcInfo(time=data['time'],
                    memory=data['memory_usage'],
                    points=data['points'],
                    arithmetic=data['actions'])