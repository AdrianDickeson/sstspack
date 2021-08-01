from pytest import approx
import numpy as np
from numpy.ma.testutils import assert_array_equal, assert_array_almost_equal

import sstspack.fitting as fit


def test_parameter_transform_function():
    func1 = fit.parameter_transform_function((-np.inf, np.inf))
    func2 = fit.parameter_transform_function((1, np.inf))
    func3 = fit.parameter_transform_function((-np.inf, 1))
    func4 = fit.parameter_transform_function((-3, 4))

    assert func1(3) == 3
    assert func1(-3) == -3
    assert func1(0) == 0

    assert func2(0) == approx(2)
    assert func2(1) == approx(1 + np.exp(1))
    assert func2(-1) == approx(1 + np.exp(-1))

    assert func3(0) == approx(0)
    assert func3(-1) == approx(1 - np.exp(1))
    assert func3(1) == approx(1 - np.exp(-1))

    assert func4(-1e10) == approx(-3)
    assert func4(0) == approx(0.5)
    assert func4(1e10) == approx(4)


def test_inverse_parameter_transform_function():
    func1 = fit.parameter_transform_function((-np.inf, np.inf))
    func2 = fit.parameter_transform_function((1, np.inf))
    func3 = fit.parameter_transform_function((-np.inf, 1))
    func4 = fit.parameter_transform_function((-3, 4))

    ifunc1 = fit.inverse_parameter_transform_function((-np.inf, np.inf))
    ifunc2 = fit.inverse_parameter_transform_function((1, np.inf))
    ifunc3 = fit.inverse_parameter_transform_function((-np.inf, 1))
    ifunc4 = fit.inverse_parameter_transform_function((-3, 4))

    assert func1(ifunc1(3)) == approx(3)
    assert func2(ifunc2(3)) == approx(3)
    assert func3(ifunc3(-3)) == approx(-3)
    assert func4(ifunc4(3)) == approx(3)


def test_fit_model_max_likelihood():
    pass
