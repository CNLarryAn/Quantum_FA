import numpy as np
from functools import reduce

def f1(x_):  # Sphere
    return np.linalg.norm(x_) ** 2


def f2(x_):  # Schwefel 2.22
    xa = reduce(lambda x, y: x * y, abs(x_))
    xb = np.linalg.norm(x_, ord=1)
    return xa + xb


def f3(x_):  # Rosenbrock
    xa = 100 * (x_[1:] - x_[:-1] * x_[:-1])
    xb = (x_[:-1] - 1) * (x_[:-1] - 1)
    return reduce(lambda x, y: x + y, xa + xb)


def f4(x_):  # Step
    return np.linalg.norm(np.floor(x_ + 0.5))


def f5(x_):  # Schwefel 2.26
    x_new = x_ * np.sin(np.sqrt(abs(x_)))
    return reduce(lambda x, y: x + y, x_new)


def f6(x_):  # Rastrigin
    x_new = (x_ * x_) - (10 * np.cos(2 * np.pi * x_)) + 10
    return reduce(lambda x, y: x + y, x_new)


def f7(x_):  # Ackley
    D = len(x_)
    x_a = 20 * np.exp(-0.2 * np.sqrt(np.linalg.norm(x_)) / np.sqrt(D))
    x_b = np.exp(reduce(lambda x, y: x + y, np.cos(2 * np.pi * x_)) / D)
    return (20 - x_a) + (np.e - x_b)


def f8(x_):  # Girewank
    x_a = (np.linalg.norm(x_) ** 2) / 4000
    x_b = reduce(lambda x, y: x * y,
                 np.cos(x_ / np.array(range(1, len(x_) + 1))))
    return x_a - x_b + 1