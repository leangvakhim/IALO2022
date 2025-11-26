import numpy as np

def sphere(x):
    return np.sum(x**2)

def schwefel_2_22(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def schwefel_1_2(x):
    cum_sum = np.cumsum(x)
    return np.sum(cum_sum**2)

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def quartic(x):
    dim = len(x)
    i_vec = np.arange(1, dim + 1)
    return np.sum(i_vec * (x**4)) + np.random.rand()

def rastrigin(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def ackley(x):
    dim = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim)) - \
           np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.e

def griewank(x):
    dim = len(x)
    # 1-based index for denominator
    i_vec = np.arange(1, dim + 1)
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(i_vec)))
    return sum_part - prod_part + 1

def penalized(x):
    dim = len(x)
    a = 10
    k = 100
    m = 4

    y = 1 + (x + 1) / 4
    u = np.zeros(dim)
    mask_high = x > a
    mask_low = x < -a
    u[mask_high] = k * (x[mask_high] - a)**m
    u[mask_low] = k * (-x[mask_low] - a)**m
    sum_u = np.sum(u)

    term1 = 10 * (np.sin(np.pi * y[0]))**2
    term2 = np.sum((y[:-1] - 1)**2 * (1 + 10 * (np.sin(np.pi * y[1:]))**2))
    term3 = (y[-1] - 1)**2

    term_main = (np.pi / dim) * (term1 + term2 + term3)

    return term_main + sum_u


