'''
Created on 15 May 2020

@author: adriandickeson
'''
import unittest

from numpy.ma.testutils import assert_array_equal, assert_array_almost_equal
from numpy import ones, zeros, full, identity, hstack, vstack, pi as PI

import sstspack.modeldata as md

class Test_modeldata(unittest.TestCase):


    def setUp(self):
        self.model_columns = ['Z', 'd', 'H', 'T', 'c', 'R', 'Q']
        self.short_model_rows = 1
        self.long_model_rows = 100

    def test_get_local_level_model_data(self):
        sigma2_eta = 2
        sigma2_epsilon = 1.1

        H = full((1,1), sigma2_epsilon)
        Q = full((1,1), sigma2_eta)

        data_df = md.get_local_level_model_data(self.short_model_rows, sigma2_eta, sigma2_epsilon)

        self.assertTrue(all([x in data_df.columns for x in self.model_columns]))
        self.assertEqual(len(data_df), self.short_model_rows)
        assert_array_equal(data_df.loc[0, 'Z'], ones((1,1)))
        assert_array_equal(data_df.loc[0, 'd'], zeros((1,1)))
        assert_array_equal(data_df.loc[0, 'H'], H)
        assert_array_equal(data_df.loc[0, 'T'], ones((1,1)))
        assert_array_equal(data_df.loc[0, 'c'], zeros((1,1)))
        assert_array_equal(data_df.loc[0, 'R'], ones((1,1)))
        assert_array_equal(data_df.loc[0, 'Q'], Q)

        data_df = md.get_local_level_model_data(self.short_model_rows, Q, H)
        assert_array_equal(data_df.loc[0, 'H'], H)
        assert_array_equal(data_df.loc[0, 'Q'], Q)

        H = sigma2_epsilon * identity(2)
        data_df = md.get_local_level_model_data(self.short_model_rows, Q, H)
        assert_array_equal(data_df.loc[0, 'Z'], ones((2,1)))
        assert_array_equal(data_df.loc[0, 'd'], zeros((2,1)))
        assert_array_equal(data_df.loc[0, 'H'], H)

    def test_get_local_linear_trend_model_data(self):
        H = ones((1,1))
        Q = ones((2,2))
        data_df = md.get_local_linear_trend_model_data(self.short_model_rows, Q, H)

        self.assertEqual(len(data_df), self.short_model_rows)
        Z = zeros((1,2))
        Z[0,0] = 1
        assert_array_equal(data_df.loc[0,'Z'], Z)
        assert_array_equal(data_df.loc[0,'d'], zeros((1,1)))
        assert_array_equal(data_df.loc[0,'H'], H)

        T = ones((2,2))
        T[1,0] = 0
        assert_array_equal(data_df.loc[0,'T'], T)
        assert_array_equal(data_df.loc[0,'c'], zeros((2,1)))
        assert_array_equal(data_df.loc[0,'R'], identity(2))
        assert_array_equal(data_df.loc[0,'Q'], Q)

        H = identity(2)
        data_df = md.get_local_linear_trend_model_data(self.short_model_rows, Q, H)
        assert_array_equal(data_df.loc[0, 'Z'], hstack([ones((2,1)), zeros((2,1))]))
        assert_array_equal(data_df.loc[0, 'd'], zeros((2,1)))
        assert_array_equal(data_df.loc[0, 'H'], H)

    def test_get_time_domain_seasonal_model_data(self):
        s = 3
        H = 5
        sigma2_omega = 2
        data_df = md.get_time_domain_seasonal_model_data(self.short_model_rows, s, sigma2_omega, H)

        Z = zeros((1,s))
        Z[0,0] = 1
        assert_array_equal(data_df.loc[0,'Z'], Z)
        assert_array_equal(data_df.loc[0,'d'], zeros((1,1)))
        assert_array_equal(data_df.loc[0,'H'], full((1,1), H))

        T = zeros((3,3))
        T[0,1] = T[0,0] = -1
        T[1,0] = T[2,1] = 1
        R = zeros((s,1))
        R[0,0] = 1
        assert_array_equal(data_df.loc[0,'T'], T)
        assert_array_equal(data_df.loc[0,'c'], zeros((3,1)))
        assert_array_equal(data_df.loc[0,'R'], R)
        assert_array_equal(data_df.loc[0,'Q'], full((1,1), sigma2_omega))

        H = identity(2)
        data_df = md.get_time_domain_seasonal_model_data(self.short_model_rows, s, sigma2_omega, H)
        assert_array_equal(data_df.loc[0, 'Z'], vstack([Z, Z]))
        assert_array_equal(data_df.loc[0, 'd'], zeros((2,1)))
        assert_array_equal(data_df.loc[0, 'H'], H)

    def test_get_static_model_df(self):
        a, b, c = (2, 3, 4)
        data_df = md.get_static_model_df(self.long_model_rows, a=a, b=b, c=c)
        self.assertEqual(len(data_df), self.long_model_rows)
        self.assertTrue(all(data_df.a == a))
        self.assertTrue(all(data_df.b == b))
        self.assertTrue(all(data_df.c == c))

    def test_combine_model_data(self):
        H1 = 3
        Q1 = 2
        model1 = md.get_local_level_model_data(self.short_model_rows, Q1, H1)
        sigma2_omega = 4
        s = 3
        H = 5
        model2 = md.get_time_domain_seasonal_model_data(self.short_model_rows, s,
                                                        sigma2_omega, H)
        combined_model = md.combine_model_data([model1, model2])
        Z = zeros((1, 4))
        Z[0,0] = Z[0,1] = 1
        assert_array_equal(combined_model.loc[0, 'Z'], Z)
        assert_array_equal(combined_model.loc[0, 'd'], zeros((1,1)))
        assert_array_equal(combined_model.loc[0, 'H'], full((1,1), 8))

        T = zeros((4,4))
        T[0,0] = T[2,1] = T[3,2] = 1
        T[1,1] = T[1,2] = -1        
        assert_array_equal(combined_model.loc[0, 'T'], T)
        assert_array_equal(combined_model.loc[0, 'c'], zeros((4,1)))
        R = zeros((4,2))
        R[0,0] = R[1,1] = 1
        assert_array_equal(combined_model.loc[0, 'R'], R)
        Q = zeros((2,2))
        Q[0,0] = 2
        Q[1,1] = 4
        assert_array_equal(combined_model.loc[0, 'Q'], Q)

    def test_get_frequency_domain_seasonal_model_data(self):
        H = 2
        sigma2_omega = [3, 4]
        s = 4
        data_df = md.get_frequency_domain_seasonal_model_data(self.short_model_rows, s,
                                                              sigma2_omega, H)
        Z = ones((1,3))
        Z[0,1] = 0
        assert_array_equal(data_df.loc[0, 'Z'], Z)
        assert_array_equal(data_df.loc[0, 'd'], zeros((1,1)))
        assert_array_equal(data_df.loc[0, 'H'], full((1,1), 2))

        T = zeros((3,3))
        T[2,2] = T[1,0] = -1
        T[0,1] = 1
        assert_array_almost_equal(data_df.loc[0, 'T'], T)
        assert_array_equal(data_df.loc[0, 'c'], zeros((3,1)))
        R = zeros((3,2))
        R[0,0] = R[2,1] = 1
        assert_array_equal(data_df.loc[0, 'R'], R)
        Q = zeros((2,2))
        Q[0,0] = 3
        Q[1,1] = 4
        assert_array_equal(data_df.loc[0, 'Q'], Q)

    def test_frequency_domain_model_terms(self):
        i = 2
        s = 4
        omega = 2 * PI / s
        sigma2 = 4
        Z, T, c, R, Q = md.frequency_domain_model_terms(i, s, omega, sigma2)
        assert_array_equal(Z, ones((1,1)))
        assert_array_equal(T, full((1,1), -1))
        assert_array_equal(c, zeros((1,1)))
        assert_array_equal(R, ones((1,1)))
        assert_array_equal(Q, full((1,1), sigma2))

        i = 1
        Z, T, c, R, Q = md.frequency_domain_model_terms(i, s, omega, sigma2)
        Z_expected = zeros((1,2))
        Z_expected[0,0] = 1
        assert_array_equal(Z, Z_expected)
        T_expected = zeros((2,2))
        T_expected[0,1] = 1
        T_expected[1,0] = -1
        assert_array_almost_equal(T, T_expected)
        assert_array_equal(c, zeros((2,1)))
        R_expected = zeros((2,1))
        R_expected[0,0] = 1
        assert_array_equal(R, R_expected)
        assert_array_equal(Q, full((1,1), sigma2))

    def test_get_ARMA_model_data(self):
        phi_terms = [0.25, -0.25]
        theta_terms = [0.4]
        Q = full((1,1), 0.5)
        data_df = md.get_ARMA_model_data(self.short_model_rows, phi_terms, theta_terms, Q)

        Z = zeros((1,2))
        Z[0,0] = 1
        assert_array_equal(data_df.loc[0, 'Z'], Z)
        assert_array_equal(data_df.loc[0, 'd'], zeros((1,1)))
        assert_array_equal(data_df.loc[0, 'H'], zeros((1,1)))

        T = zeros((2,2))
        T[:,0] = phi_terms
        T[0,1] = 1
        assert_array_equal(data_df.loc[0, 'T'], T)
        assert_array_equal(data_df.loc[0, 'c'], zeros((2,1)))
        R = ones((2,1))
        R[1:,0] = theta_terms
        assert_array_equal(data_df.loc[0, 'R'], R)
        assert_array_equal(data_df.loc[0, 'Q'], Q)

    def test_get_ARIMA_model_data(self):
        phi_terms = [0.25, -0.25]
        theta_terms = [0.4]
        Q = full((1,1), 0.5)
        d = 1
        data_df = md.get_ARIMA_model_data(self.short_model_rows, phi_terms, d, theta_terms, Q)

        Z = zeros((1,3))
        Z[0,0] = Z[0,1] = 1
        assert_array_equal(data_df.loc[0, 'Z'], Z)
        assert_array_equal(data_df.loc[0, 'd'], zeros((1,1)))
        assert_array_equal(data_df.loc[0, 'H'], zeros((1,1)))

        T = zeros((3,3))
        T[1:,1] = phi_terms
        T[0,0] = T[0,1] = T[1,2] = 1
        assert_array_equal(data_df.loc[0, 'T'], T)
        assert_array_equal(data_df.loc[0, 'c'], zeros((3,1)))
        R = ones((3,1))
        R[0,0] = 0
        R[2:,0] = theta_terms
        assert_array_equal(data_df.loc[0, 'R'], R)
        assert_array_equal(data_df.loc[0, 'Q'], Q)

    def test_observation_terms(self):
        Z_input = ones((1,1))
        H_input = 5
        Z, d, H = md.observation_terms(H_input, Z_input)
        assert_array_equal(Z, Z_input)
        assert_array_equal(d, zeros((1,1)))
        assert_array_equal(H, full((1,1), H_input))

        H_input = full((1,1), H_input)
        Z, d, H = md.observation_terms(H_input, Z_input)
        assert_array_equal(Z, Z_input)
        assert_array_equal(d, zeros((1,1)))
        assert_array_equal(H, H_input)

        H_input = 5 * identity(2)
        Z, d, H = md.observation_terms(H_input, Z_input)
        assert_array_equal(Z, ones((2,1)))
        assert_array_equal(d, zeros((2,1)))
        assert_array_equal(H, H_input)


if __name__ == "__main__":
    unittest.main()
