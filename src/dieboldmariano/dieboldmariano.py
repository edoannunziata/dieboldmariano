from itertools import islice
from typing import Sequence, Callable, List, Tuple
from math import lgamma, fabs, isnan, nan, exp, log, log1p, sqrt


class InvalidParameterException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class ZeroVarianceException(ArithmeticError):
    def __init__(self, message: str):
        super().__init__(message)


def autocovariance(X: Sequence[float], k: int, mean: float) -> float:
    """
    Returns the k-lagged autocovariance for the input iterable.
    """
    return sum((a - mean) * (b - mean) for a, b in zip(islice(X, k, None), X)) / len(X)


def log_beta(a: float, b: float) -> float:
    """
    Returns the natural logarithm of the beta function computed on
    arguments `a` and `b`.
    """
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def evaluate_continuous_fraction(
    fa: Callable[[int, float], float],
    fb: Callable[[int, float], float],
    x: float,
    *,
    epsilon: float = 1e-10,
    maxiter: int = 10000,
    small: float = 1e-50
) -> float:
    """
    Evaluate a continuous fraction.
    """
    h_prev = fa(0, x)
    if fabs(h_prev < small):
        h_prev = small

    n: int = 1
    d_prev: float = 0.0
    c_prev: float = h_prev
    hn: float = h_prev

    while n < maxiter:
        a = fa(n, x)
        b = fb(n, x)

        dn = a + b * d_prev
        if fabs(dn) < small:
            dn = small

        cn = a + b / c_prev
        if fabs(cn) < small:
            cn = small

        dn = 1 / dn
        delta_n = cn * dn
        hn = h_prev * delta_n

        if fabs(delta_n - 1.0) < epsilon:
            break

        d_prev = dn
        c_prev = cn
        h_prev = hn

        n += 1

    return hn


def regularized_incomplete_beta(
    x: float, a: float, b: float, *, epsilon: float = 1e-10, maxiter: int = 10000
) -> float:
    if isnan(x) or isnan(a) or isnan(b) or x < 0 or x > 1 or a <= 0 or b <= 0:
        return nan

    if x > (a + 1) / (2 + b + a) and 1 - x <= (b + 1) / (2 + b + a):
        return 1 - regularized_incomplete_beta(
            1 - x, b, a, epsilon=epsilon, maxiter=maxiter
        )

    def fa(n: int, x: float) -> float:
        return 1.0

    def fb(n: int, x: float) -> float:
        if n % 2 == 0:
            m = n / 2.0
            return (m * (b - m) * x) / ((a + (2 * m) - 1) * (a + (2 * m)))

        m = (n - 1.0) / 2.0
        return -((a + m) * (a + b + m) * x) / ((a + (2 * m)) * (a + (2 * m) + 1.0))

    return exp(
        a * log(x) + b * log1p(-x) - log(a) - log_beta(a, b)
    ) / evaluate_continuous_fraction(fa, fb, x, epsilon=epsilon, maxiter=maxiter)


def dm_test(
    V: Sequence[float],
    P1: Sequence[float],
    P2: Sequence[float],
    *,
    loss: Callable[[float, float], float] = lambda u, v: (u - v) ** 2,
    h: int = 1,
    one_sided: bool = False,
    harvey_correction: bool = True
) -> Tuple[float, float]:
    r"""
    Performs the Diebold-Mariano test. The null hypothesis is that the two forecasts (`P1`, `P2`) have the same accuracy.

    Parameters
    ----------
    V: Sequence[float]
        The actual timeseries.

    P1: Sequence[float]
        First prediction series.

    P2: Sequence[float]
        Second prediction series.

    loss: Callable[[float, float], float]
        Loss function. At each time step of the series, each prediction is charged a loss, 
        computed as per this function. The Diebold-Mariano test is agnostic with respect to 
        the loss function, and this implementation supports arbitrarily specified (for example asymmetric) 
        functions. The two arguments are, *in this order*, the actual value and the predicted value. 
        Default is squared error (i.e. `lambda u, v: (u - v) ** 2`)

    h: int
        The forecast horizon. Default is 1.

    one_sided: bool
        If set to true, returns the p-value for a one-sided test instead of a two-sided test. Default is false.

    harvey_correcetion: bool
        If set to true, uses a modified test statistics as per Harvey, Leybourne and Newbold (1997).

    Returns
    -------
    A tuple of two values. The first is the test statistic, the second is the p-value.
    """
    if not (len(V) == len(P1) == len(P2)):
        raise InvalidParameterException(
            "Actual timeseries and prediction series must have the same length."
        )

    if h <= 0:
        raise InvalidParameterException(
            "Invalid parameter for horizon length. Must be a positive integer."
        )

    n = len(P1)
    mean = 0.0
    loss1 = 0.0
    loss2 = 0.0
    D: List[float] = []

    for v, p1, p2 in zip(V, P1, P2):
        l1 = loss(v, p1)
        l2 = loss(v, p2)
        D.append(l1 - l2)
        mean += l1 - l2
        loss1 += l1
        loss2 += l2

    mean /= n

    V_d = 0.0
    for i in range(h):
        V_d += autocovariance(D, i, mean)
        if i == 0:
            V_d /= 2

    V_d = 2 * V_d / n

    if V_d == 0:
        raise ZeroVarianceException(
            "Variance of the DM statistic is zero. Maybe the prediction series are identical?"
        )

    if harvey_correction:
        harvey_adj = sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dmstat = harvey_adj / sqrt(V_d) * mean
    else:
        dmstat = mean / sqrt(V_d)

    pvalue = regularized_incomplete_beta(
        (n - 1) / ((n - 1) + dmstat ** 2), 0.5 * (n - 1), 0.5
    )

    if one_sided:
        pvalue = pvalue / 2 if dmstat < 0 else 1 - pvalue / 2

    return dmstat, pvalue
