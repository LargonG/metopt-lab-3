from solution.optimize.bfgs import *
from solution.optimize.lbfgs import *
from solution.optimize.gauss_newton import *
from solution.optimize.powell_dog_leg import *


def optimize(func, jac, reg, grad=None, hess=None, method='GAUSS-NEWTON'):
    iterations = 100

    def run_gauss(x, y, init_point, eps=1e-4, max_iter=iterations):
        return gauss_newton(reg(func, x, y), jac(x), init_point, eps, max_iter)

    def run_powell(x, y, init_point, delta, eps, max_iter=iterations):
        r = reg(func, x, y)
        return powell_dog_leg(r, grad(x, jac, r), hess(x, jac), init_point, delta, eps, max_iter)

    def run_bfgs(x, y, init_point, eps, max_iter=iterations):
        r = reg(func, x, y)
        F = lambda betta: sum(r(betta) ** 2)
        return bfgs(F, grad(x, jac, r), init_point, eps, max_iter)

    def run_lbfgs(x, y, init_point, t, generations, eps, max_iter=iterations):
        r = reg(func, x, y)
        g = grad(x, jac, r)
        F = lambda betta: sum(r(betta) ** 2)
        return lbfgs(F, g, generations, init_point, t, eps, max_iter)

    mp = {
        'GAUSS-NEWTON': run_gauss,
        'POWELL-DOG-LEG': run_powell,
        'BFGS': run_bfgs,
        'L_BFGS': run_lbfgs
    }

    return mp[method]