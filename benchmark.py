import numpy as np

def sphere(x):
    # F1
    return np.sum(x**2)

def schwefel_2_22(x):
    # F2
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def rosenbrock(x):
    # F4
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def rastrigin(x):
    # F6
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def ackley(x):
    # F7
    dim = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.e