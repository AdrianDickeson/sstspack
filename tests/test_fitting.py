"""
Created on 4 Sep 2020

@author: adriandickeson
"""
import unittest

import numpy as np
from numpy.ma.testutils import assert_array_equal, assert_array_almost_equal

import sstspack.fitting as fit


class TestFitting(unittest.TestCase):
    def test_parameter_transform_function(self):
        func1 = fit.parameter_transform_function((-np.inf, np.inf))
        func2 = fit.parameter_transform_function((1, np.inf))
        func3 = fit.parameter_transform_function((-np.inf, 1))
        func4 = fit.parameter_transform_function((-3, 4))

        self.assertEqual(func1(3), 3)
        self.assertEqual(func1(-3), -3)
        self.assertEqual(func1(0), 0)

        self.assertAlmostEqual(func2(0), 2)
        self.assertAlmostEqual(func2(1), 1 + np.exp(1))
        self.assertAlmostEqual(func2(-1), 1 + np.exp(-1))

        self.assertAlmostEqual(func3(0), 0)
        self.assertAlmostEqual(func3(-1), 1 - np.exp(1))
        self.assertAlmostEqual(func3(1), 1 - np.exp(-1))

        self.assertAlmostEqual(func4(-1e10), -3)
        self.assertAlmostEqual(func4(0), 0.5)
        self.assertAlmostEqual(func4(1e10), 4)

    def test_inverse_parameter_transform_function(self):
        func1 = fit.parameter_transform_function((-np.inf, np.inf))
        func2 = fit.parameter_transform_function((1, np.inf))
        func3 = fit.parameter_transform_function((-np.inf, 1))
        func4 = fit.parameter_transform_function((-3, 4))

        ifunc1 = fit.inverse_parameter_transform_function((-np.inf, np.inf))
        ifunc2 = fit.inverse_parameter_transform_function((1, np.inf))
        ifunc3 = fit.inverse_parameter_transform_function((-np.inf, 1))
        ifunc4 = fit.inverse_parameter_transform_function((-3, 4))

        self.assertAlmostEqual(func1(ifunc1(3)), 3)
        self.assertAlmostEqual(func2(ifunc2(3)), 3)
        self.assertAlmostEqual(func3(ifunc3(-3)), -3)
        self.assertAlmostEqual(func4(ifunc4(3)), 3)

    def test_fit_model_max_likelihood(self):
        pass

    def test_jacobian(self):
        def testfunc(params):
            return params[0] ** 2 + params[1] ** 2

        x = np.array([2, 3])
        expected_result = np.array([4, 6])
        assert_array_almost_equal(fit.jacobian(testfunc, x), expected_result)

    def test_hessian(self):
        def testfunc1(params):
            return params[0] ** 2 * params[1] ** 2

        x = np.array([2, 3])
        expected_result = np.array([[18.0, 24.0], [24.0, 8.0]])
        assert_array_almost_equal(fit.hessian(testfunc1, x), expected_result)

        def testfunc2(params):
            return params[0] ** 2 + params[1] ** 2

        x = np.array([1, 2])
        expected_result = np.array([[2, 0], [0, 2]])
        assert_array_almost_equal(fit.hessian(testfunc2, x, 1, False), expected_result)

        def testfunc3(params):
            return np.exp(params[0] + 0.5 * params[1])

        x = np.array([2, 3])
        xexp = np.exp(3.5)
        expected_result = np.array([[xexp, 0.5 * xexp], [0.5 * xexp, 0.25 * xexp]])
        assert_array_almost_equal(fit.hessian(testfunc3, x), expected_result, h=1e-10)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()