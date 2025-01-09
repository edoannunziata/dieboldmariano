from typing import Sequence, Tuple, Callable, Literal


class InvalidParameterException(Exception):
    ...

class ZeroVarianceException(ArithmeticError):
    ...

class NegativeVarianceException(ArithmeticError):
    ...

def autocovariance(X: Sequence[float], k: int, mean: float) -> float: ...

def log_beta(a: float, b: float) -> float: ...

def evaluate_continuous_fraction(
        fa: Callable[[int, float], float],
        fb: Callable[[int, float], float],
        x: float,
        *,
        epsilon: float = ...,
        maxiter: int = ...,
        small: float) -> float: ...

def regularized_incomplete_beta(
        x: float,
        a: float,
        b: float,
        *,
        epsilon: float = ...,
        maxiter: int = ...) -> float: ...

def dm_test(
        V: Sequence[float],
        P1: Sequence[float],
        P2: Sequence[float],
        *,
        loss: Callable[[float, float], float] = ...,
        h: int = ...,
        one_sided: bool = ...,
        harvey_correction: bool = ...,
        variance_estimator: Literal["acf", "bartlett"]) -> Tuple[float, float]: ...
