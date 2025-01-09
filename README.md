# dieboldmariano

This package provides a simple, pure-Python implementation of the Diebold-Mariano statistical test. It has no dependencies outside the Python standard library.

## What is the Diebold-Mariano test?

Assume we have a real-valued timeseries $T_{n \ge 0}$, and two forecasters 
$F$, $G$ who each produce predictions $F_{n \ge 0}$, $G_{n \ge 0}$. Which 
one is better? Is the difference statistically significant?

The cost function for errors can be specified arbitrarily and does not
need to be symmetric (for example, one might want to penalize errors in one
direction more).

An usual choice (the default in this implementation if no cost function
is passed explicitly) is the square error.

The test produces the DM statistic and the corresponding p-value.

A more detailed and very accessible description is available at the link [3]
in the references section.

## Code example

```python
from dieboldmariano import dm_test
T = [10, 20, 30, 40, 50]
F = [11, 21, 29, 42, 53]
G = [13, 26, 24, 40, 59]

dm_test(T, F, G, one_sided=True)
# (-2.2229922805746782, 0.04515565862099125)
```

## Usage

The public interface of the package is the `dm_test` function.

It outputs a tuple of two values, the DM statistic and the p-value, in that order.

It takes the following parameters:

**V** (ordinal): `Sequence[float]`  
The actual timeseries.

**P1** (ordinal): `Sequence[float]`  
First prediction series.

**P2** (ordinal): `Sequence[float]`  
Second prediction series.

**loss** (keyword): `Callable[[float, float], float]`  
Loss function used to charge each prediction at each timestep. Takes actual and predicted values as arguments in that order. 
Default: squared error `lambda u, v: (u - v) ** 2`

**h** (keyword): `int`  
Forecast horizon.
Default: 1

**one_sided** (keyword): `bool`  
If true, tests if P2 has at least as much predictive accuracy as P1. If false, tests if accuracies are equal.
Default: false

**harvey_correction** (keyword): `bool`  
If true, uses modified test statistics per [1].
Default: true

**variance_estimator** (keyword): `Literal["acf", "bartlett"]`  
Long-run variance estimator type:
- `"acf"`: autocorrelation method
- `"bartlett"`: Bartlett weights method

Both methods are discussed in [2].

## References
[1] Diebold, F.X. and Mariano, R.S. (1995) Comparing predictive accuracy. Journal of Business and Economic Statistics, 13, 253-263.

[2] Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of forecasting, 13(2), 281-291.

[3] https://www.real-statistics.com/time-series-analysis/forecasting-accuracy/diebold-mariano-test/
