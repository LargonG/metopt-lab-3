def to_normal_function(legacy_function, grad):
    def run(x, y, init_point, eps=0.01, learning_rate=0.01, batch_size=1):
        return legacy_function(to_legacy_gradient(grad),
                               x, y, init_point, eps, learning_rate, batch_size)

    return run


def to_legacy_gradient(grad):
    return lambda x, y, w, batch_size: grad(w)
