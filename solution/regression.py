def regression(func_model, x, y):
    return lambda param: y - func_model(x)(param)


def grad(x, jacobian, r):
    J = jacobian(x)
    return lambda betta: 2 * J(betta).transpose() @ r(betta)


def hessian(x, jacobian):
    j = jacobian(x)
    return lambda betta: 2 * j(betta).transpose() @ j(betta)


def regression_function(func_model, x, y):
    return lambda betta: sum((y - func_model(x)(betta)) ** 2)
