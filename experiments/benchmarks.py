"""Classical 23 benchmark functions (Yao, Liu & Lin, IEEE TEVC 1999).
Every function takes X of shape (N, d) and returns shape (N,).
F7 additionally takes an rng for its uniform noise term."""
import numpy as np

PI = np.pi


def f1(X, rng=None):  # Sphere
    return np.sum(X**2, axis=1)


def f2(X, rng=None):  # Schwefel 2.22
    A = np.abs(X)
    return np.sum(A, axis=1) + np.prod(A, axis=1)


def f3(X, rng=None):  # Schwefel 1.2
    return np.sum(np.cumsum(X, axis=1)**2, axis=1)


def f4(X, rng=None):  # Schwefel 2.21
    return np.max(np.abs(X), axis=1)


def f5(X, rng=None):  # Rosenbrock
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1]**2)**2 + (X[:, :-1] - 1.0)**2, axis=1)


def f6(X, rng=None):  # Step
    return np.sum(np.floor(X + 0.5)**2, axis=1)


def f7(X, rng):  # Quartic with noise
    i = np.arange(1, X.shape[1] + 1)
    return np.sum(i * X**4, axis=1) + rng.random(X.shape[0])


def f8(X, rng=None):  # Schwefel 2.26
    return np.sum(-X * np.sin(np.sqrt(np.abs(X))), axis=1)


def f9(X, rng=None):  # Rastrigin
    return np.sum(X**2 - 10.0 * np.cos(2 * PI * X) + 10.0, axis=1)


def f10(X, rng=None):  # Ackley
    d = X.shape[1]
    return (-20.0 * np.exp(-0.2 * np.sqrt(np.sum(X**2, axis=1) / d))
            - np.exp(np.sum(np.cos(2 * PI * X), axis=1) / d) + 20.0 + np.e)


def f11(X, rng=None):  # Griewank
    i = np.sqrt(np.arange(1, X.shape[1] + 1))
    return np.sum(X**2, axis=1) / 4000.0 - np.prod(np.cos(X / i), axis=1) + 1.0


def _u(X, a, k, m):
    return np.sum(k * ((X - a)**m) * (X > a) + k * ((-X - a)**m) * (X < -a), axis=1)


def f12(X, rng=None):  # Generalized penalized 1
    d = X.shape[1]
    Y = 1.0 + (X + 1.0) / 4.0
    s = (10.0 * np.sin(PI * Y[:, 0])**2
         + np.sum((Y[:, :-1] - 1.0)**2 * (1.0 + 10.0 * np.sin(PI * Y[:, 1:])**2), axis=1)
         + (Y[:, -1] - 1.0)**2)
    return PI / d * s + _u(X, 10, 100, 4)


def f13(X, rng=None):  # Generalized penalized 2
    s = (np.sin(3 * PI * X[:, 0])**2
         + np.sum((X[:, :-1] - 1.0)**2 * (1.0 + np.sin(3 * PI * X[:, 1:])**2), axis=1)
         + (X[:, -1] - 1.0)**2 * (1.0 + np.sin(2 * PI * X[:, -1])**2))
    return 0.1 * s + _u(X, 5, 100, 4)


_A14 = np.array([[-32, -16, 0, 16, 32] * 5,
                 sum([[v] * 5 for v in [-32, -16, 0, 16, 32]], [])], dtype=float)


def f14(X, rng=None):  # Shekel's foxholes
    j = np.arange(1, 26)
    s = np.sum((X[:, :, None] - _A14[None, :, :])**6, axis=1)  # (N,25)
    return 1.0 / (1.0 / 500.0 + np.sum(1.0 / (j + s), axis=1))


_A15 = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627,
                 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
_B15 = 1.0 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])


def f15(X, rng=None):  # Kowalik
    x1, x2, x3, x4 = (X[:, k:k + 1] for k in range(4))
    b = _B15[None, :]
    return np.sum((_A15 - x1 * (b**2 + b * x2) / (b**2 + b * x3 + x4))**2, axis=1)


def f16(X, rng=None):  # Six-hump camel back
    x1, x2 = X[:, 0], X[:, 1]
    return 4 * x1**2 - 2.1 * x1**4 + x1**6 / 3 + x1 * x2 - 4 * x2**2 + 4 * x2**4


def f17(X, rng=None):  # Branin
    x1, x2 = X[:, 0], X[:, 1]
    return ((x2 - 5.1 / (4 * PI**2) * x1**2 + 5 / PI * x1 - 6)**2
            + 10 * (1 - 1 / (8 * PI)) * np.cos(x1) + 10)


def f18(X, rng=None):  # Goldstein-Price
    x1, x2 = X[:, 0], X[:, 1]
    a = 1 + (x1 + x2 + 1)**2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)
    b = 30 + (2 * x1 - 3 * x2)**2 * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2)
    return a * b


_C_H = np.array([1.0, 1.2, 3.0, 3.2])
_A19 = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
_P19 = np.array([[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.7470],
                 [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
_A20 = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                 [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_P20 = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                 [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                 [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                 [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])


def _hartman(X, A, P):
    e = np.sum(A[None] * (X[:, None, :] - P[None])**2, axis=2)  # (N,4)
    return -np.sum(_C_H * np.exp(-e), axis=1)


def f19(X, rng=None):
    return _hartman(X, _A19, _P19)


def f20(X, rng=None):
    return _hartman(X, _A20, _P20)


_A_SH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                  [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
_C_SH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])


def _shekel(X, m):
    s = np.sum((X[:, None, :] - _A_SH[None, :m])**2, axis=2) + _C_SH[None, :m]
    return -np.sum(1.0 / s, axis=1)


def f21(X, rng=None):
    return _shekel(X, 5)


def f22(X, rng=None):
    return _shekel(X, 7)


def f23(X, rng=None):
    return _shekel(X, 10)


# name, function, lower bound(s), upper bound(s), dimension, known optimum, type
FUNCS = {
    'F1': ('Sphere', f1, -100, 100, 30, 0.0, 'U'),
    'F2': ('Schwefel 2.22', f2, -10, 10, 30, 0.0, 'U'),
    'F3': ('Schwefel 1.2', f3, -100, 100, 30, 0.0, 'U'),
    'F4': ('Schwefel 2.21', f4, -100, 100, 30, 0.0, 'U'),
    'F5': ('Rosenbrock', f5, -30, 30, 30, 0.0, 'U'),
    'F6': ('Step', f6, -100, 100, 30, 0.0, 'U'),
    'F7': ('Quartic with noise', f7, -1.28, 1.28, 30, 0.0, 'U'),
    'F8': ('Schwefel 2.26', f8, -500, 500, 30, -418.9829 * 30, 'M'),
    'F9': ('Rastrigin', f9, -5.12, 5.12, 30, 0.0, 'M'),
    'F10': ('Ackley', f10, -32, 32, 30, 0.0, 'M'),
    'F11': ('Griewank', f11, -600, 600, 30, 0.0, 'M'),
    'F12': ('Penalized 1', f12, -50, 50, 30, 0.0, 'M'),
    'F13': ('Penalized 2', f13, -50, 50, 30, 0.0, 'M'),
    'F14': ("Shekel's foxholes", f14, -65.536, 65.536, 2, 0.998004, 'F'),
    'F15': ('Kowalik', f15, -5, 5, 4, 0.0003075, 'F'),
    'F16': ('Six-hump camel', f16, -5, 5, 2, -1.0316285, 'F'),
    'F17': ('Branin', f17, [-5, 0], [10, 15], 2, 0.397887, 'F'),
    'F18': ('Goldstein-Price', f18, -2, 2, 2, 3.0, 'F'),
    'F19': ('Hartman 3', f19, 0, 1, 3, -3.86278, 'F'),
    'F20': ('Hartman 6', f20, 0, 1, 6, -3.32200, 'F'),
    'F21': ('Shekel 5', f21, 0, 10, 4, -10.1532, 'F'),
    'F22': ('Shekel 7', f22, 0, 10, 4, -10.4029, 'F'),
    'F23': ('Shekel 10', f23, 0, 10, 4, -10.5364, 'F'),
}


def bounds(key):
    _, _, lb, ub, d, _, _ = FUNCS[key]
    lb = np.broadcast_to(np.asarray(lb, float), (d,)).copy()
    ub = np.broadcast_to(np.asarray(ub, float), (d,)).copy()
    return lb, ub, d
