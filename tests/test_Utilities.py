from numpy import array, ones, identity, zeros, full, exp
from numpy.random import normal
from numpy.ma.testutils import assert_array_equal, assert_array_almost_equal

from sstspack.Utilities import block_diag, identity_fn, jacobian, hessian, identity_fn


def test_block_diag():
    """"""
    expected = identity(4)

    assert_array_equal(block_diag([identity(2), identity(2)]), expected)


def test_jacobian():
    """"""

    def test_func(x):
        return full((1, 1), x[0] ** 2 + 3 * x[1])

    expected = zeros((1, 2))
    expected[0, 0] = 6
    expected[0, 1] = 3

    assert_array_almost_equal(
        jacobian(test_func, 3 * ones(2), relative=False), expected
    )

    def test_func(x):
        return full((1, 1), x[0] + exp(-1 - 1 * x[0]) * x[1])

    expected = zeros((1, 2))
    expected[0, 0] = 1 - exp(-4) * 2
    expected[0, 1] = exp(-4)

    assert_array_almost_equal(
        jacobian(test_func, array([3, 2]), relative=False), expected
    )

    c_0 = 0.01
    c_mu = 0.1

    def test_exp_func(x):
        return full((1, 1), x[0] + exp(c_0 + c_mu * x[0]) * x[2])

    x_values = 3
    exp_term = exp(c_0 + c_mu * x_values)
    expected = zeros((1, 4))
    expected[0, 0] = 1 + c_mu * exp_term * x_values
    expected[0, 2] = exp_term

    assert_array_almost_equal(jacobian(test_exp_func, full(4, 3), h=1e-6), expected)


def test_hessian():
    """"""

    def test_func(x):
        return 3 * x[0] ** 2 * x[1]

    expected = zeros((2, 2))
    expected[0, 0] = 18
    expected[1, 1] = 0
    expected[0, 1] = expected[1, 0] = 18

    assert_array_almost_equal(
        hessian(test_func, 3 * ones(2), relative=False), expected, decimal=4
    )


def test_identity_fn():
    """"""
    expected = normal(1)
    result = identity_fn(expected)

    assert result == expected
