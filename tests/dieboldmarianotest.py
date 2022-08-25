from unittest import TestCase, main
from src.dieboldmariano import (
    regularized_incomplete_beta,
    dm_test,
    autocovariance,
)


class TestRegularizedBeta(TestCase):
    def test_regbeta(self):
        test_cases = [
            (0.5, 0.9, 0.1, 0.0772608),
            (0.3, 0.1, 0.9, 0.874676),
            (0.41, 0.17, 0.19, 0.5022324),
            (0.5, 0.5, 0.5, 0.5),
        ]

        for x, a, b, exp in test_cases:
            self.assertAlmostEqual(regularized_incomplete_beta(x, a, b), exp, places=4)


class TestAutocovariance(TestCase):
    def test_autocovariance(self):
        V = [0, 2, 3, 2, 1, 8, 3, 2]
        mean = sum(x for x in V) / sum(1 for _ in V)
        self.assertAlmostEqual(autocovariance(V, 3, mean), 0.6816, places=4)


class TestDieboldMariano(TestCase):
    def test_diebold_mariano_1(self):
        V = [0, 0, 0, 0, 0, 0]
        P1 = [0, 1, 2, 3, 4, 5]
        P2 = [0, 2, 3, 3, 5, 6]
        stat, pvalue = dm_test(V, P1, P2)

        self.assertAlmostEqual(stat, -2.4905, places=4)
        self.assertAlmostEqual(pvalue, 0.05513, places=4)

    def test_diebold_mariano_2(self):
        V = [0] * 15
        P1 = [
            0.3675225224615024,
            0.5450608127223211,
            0.22044990720021074,
            0.8688743471040006,
            0.6072512228948467,
            0.8582283746501538,
            0.41662295816718187,
            0.3812100211114714,
            0.7185133356116706,
            0.30827323290318875,
            0.4297093624074402,
            0.8615974488577858,
            0.7240514514808826,
            0.3450839595261038,
            0.19675301374013598,
        ]
        P2 = [
            0.12794132529988733,
            0.3679643204341153,
            0.768715420996091,
            0.09972248964114683,
            0.27197653636654284,
            0.28797629010039505,
            0.9160213458582482,
            0.6112628048698798,
            0.5452873235576241,
            0.06785907231746158,
            0.6072666485541124,
            0.6241575753989782,
            0.39998026828867217,
            0.19895665605941748,
            0.8094941215215619,
        ]
        stat, pvalue = dm_test(V, P1, P2, h=3, loss = lambda x, y: abs(x - y), one_sided=True)

        self.assertAlmostEqual(stat, 1.2109, places=4)
        self.assertAlmostEqual(pvalue, 0.877, places=4)


if __name__ == "__main__":
    main()
