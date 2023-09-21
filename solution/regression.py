def regression(func_model, x, y):
    return lambda param: y - func_model(x)(param)


def grad(x, jacobian, r):
    return lambda betta: 2 * jacobian(x)(betta).transpose() @ r(betta)


def hessian(x, jacobian):
    j = jacobian(x)
    return lambda betta: 2 * j(betta).transpose() @ j(betta)
