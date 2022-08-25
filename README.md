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

A more detailed and very accessible description is available at the link
in the references section.

## Code example

```
>>> from dieboldmariano import dm_test
>>> T = [10, 20, 30, 40, 50]
>>> F = [11, 21, 29, 42, 53]
>>> G = [13, 26, 24, 40, 59]

>>> dm_test(T, F, G, one_sided=True)
(-2.2229922805746782, 0.04515565862099125)
```

## References
Diebold, F.X. and Mariano, R.S. (1995) Comparing predictive accuracy. Journal of Business and Economic Statistics, 13, 253-263.

Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of forecasting, 13(2), 281-291.

https://www.real-statistics.com/time-series-analysis/forecasting-accuracy/diebold-mariano-test/
