def find_wolfe_lr(func, fgrad, start, direction, high,
                  c1=1e-4, c2=1 - 1e-4, eps=1e-4):
    start_value = func(start)

    low = eps
    mid = (low + high) / 2

    while high - low > eps:
        mid = (low + high) / 2

        point = start + mid * direction
        value = func(point)
        grad = fgrad(point)
        projection = grad @ direction

        if value <= start_value + c1 * mid * projection:
            if grad @ direction >= c2 * projection:
                return mid
            else:
                low = mid
        else:
            high = mid

    return mid
