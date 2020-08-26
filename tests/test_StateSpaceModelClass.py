'''
Created on 18 Feb 2020

@author: adriandickeson
'''

import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from sstspack import StateSpaceModel as SSM, modeldata as md


class TestStateSpaceModel(unittest.TestCase):


    def SetupLocalModel(self, series_length, sigma2_eta, H):
        self.local_model_data = md.get_local_level_model_data(series_length, sigma2_eta, H)

        self.a0 = np.zeros((1, 1))
        self.P0 = np.ones((1, 1))

        Y_data = pd.read_csv('data/noisy_sin_data.csv')
        self.Y = Y_data['Observed']

    def GetLocalModel(self):
        return SSM(self.Y, self.local_model_data, self.a0, self.P0)

    def GetDiffuseLocalModel(self):
        return SSM(self.Y, self.local_model_data, self.a0, self.P0, [True])

    def setUp(self):
        # Setup local-level mpdel
        self.SetupLocalModel(series_length = 100, sigma2_eta = 0.002, H = 0.01)

    def test__init__(self):
        local_model = self.GetLocalModel()

    def test_filter(self):
        local_model = self.GetLocalModel()
        local_model.filter()
        self.assertTrue(local_model.filter_run)
        self.assertAlmostEqual(local_model.a_prior_final.ravel()[0], -0.2022203)
        self.assertAlmostEqual(local_model.P_prior_final.ravel()[0], 0.00558258)

        local_model = self.GetDiffuseLocalModel()
        local_model.filter()
        self.assertAlmostEqual(local_model.a_prior_final.ravel()[0], -0.2022203)
        self.assertAlmostEqual(local_model.P_prior_final.ravel()[0], 0.00558258)

    def test_filter_missing(self):
        local_model = self.GetLocalModel()
        local_model.y[20:22] = pd.NA
        local_model.y[22:24] = np.nan
        local_model.y[24:26] = None
        local_model.y[26:28] = [[pd.NA],[pd.NA]]
        local_model.y[28:30] = [[np.nan],[np.nan]]
        local_model.y[30:32] = [[None],[None]]
        local_model.y[32:34] = [[[pd.NA]],[[pd.NA]]]
        local_model.y[34:36] = [[[np.nan]],[[np.nan]]]
        local_model.y[36:38] = [[[None]],[[None]]]
        local_model.filter()
        self.assertAlmostEqual(local_model.a_prior_final.ravel()[0], -0.2022203)
        self.assertAlmostEqual(local_model.P_prior_final.ravel()[0], 0.00558258)

    def test_smoother(self):
        local_model = self.GetLocalModel()
        local_model.smoother()
        self.assertTrue(local_model.smoother_run)
        self.assertAlmostEqual(local_model.a_hat[0].ravel()[0], 0.15147618024595688)
        self.assertAlmostEqual(local_model.V[0].ravel()[0], 0.00356978664017194)

        local_model = self.GetDiffuseLocalModel()
        local_model.smoother()
        self.assertTrue(local_model.smoother_run)
        self.assertAlmostEqual(local_model.a_hat[0].ravel()[0], 0.15201885512767077)
        self.assertAlmostEqual(local_model.V[0].ravel()[0], 0.0035825756949558396)

    def test_simulate_smoother(self):
        local_model = self.GetLocalModel()
        sim_df = local_model.simulate_smoother()
        self.assertAlmostEqual(sim_df['alpha'][0].ravel()[0] + sim_df['epsilon'][0].ravel()[0],
                               local_model.y[0])

        local_model = self.GetDiffuseLocalModel()
        sim_df = local_model.simulate_smoother()
        self.assertAlmostEqual(sim_df['alpha'][0].ravel()[0] + sim_df['epsilon'][0].ravel()[0],
                               local_model.y[0])


    def test_simulate_model(self):
        SSM.simulate_model(self.local_model_data, self.a0, self.P0)

    def test_log_likelihood(self):
        local_model = self.GetLocalModel()
        log_lik = local_model.log_likelihood()
        self.assertTrue(log_lik, 36.032667773295756)

    def test_disturbance_smoother(self):
        local_model = self.GetLocalModel()
        local_model.disturbance_smoother()
        self.assertAlmostEqual(local_model.epsilon_hat[0].ravel()[0], -0.06554295424595682)
        self.assertAlmostEqual(local_model.epsilon_hat_sigma2[0].ravel()[0], 0.0035697866640171582)
        self.assertAlmostEqual(local_model.eta_hat[0].ravel()[0], 0.01341154320968328)
        self.assertAlmostEqual(local_model.eta_hat_sigma2[0].ravel()[0], 0.001741661575038556)

        local_model = self.GetDiffuseLocalModel()
        local_model.disturbance_smoother()
        self.assertAlmostEqual(local_model.epsilon_hat[0].ravel()[0], -0.06608562912767078)
        self.assertAlmostEqual(local_model.epsilon_hat_sigma2[0].ravel()[0], 0.0035825756949558396)
        self.assertAlmostEqual(local_model.eta_hat[0].ravel()[0], 0.013217125825534158)
        self.assertAlmostEqual(local_model.eta_hat_sigma2[0].ravel()[0], 0.0017433030277982336)

    def test_is_all_missing(self):
        self.assertFalse(SSM.is_all_missing(0))
        self.assertTrue(SSM.is_all_missing(pd.NA))
        self.assertTrue(SSM.is_all_missing(None))
        self.assertTrue(SSM.is_all_missing(np.nan))
        self.assertTrue(SSM.is_all_missing(np.NaN))
        self.assertTrue(SSM.is_all_missing(np.NAN))
        self.assertFalse(SSM.is_all_missing([0]))
        self.assertFalse(SSM.is_all_missing([0, 0]))
        self.assertFalse(SSM.is_all_missing([0, pd.NA]))
        self.assertTrue(SSM.is_all_missing([pd.NA]))
        self.assertTrue(SSM.is_all_missing([pd.NA, pd.NA]))
        self.assertFalse(SSM.is_all_missing([[0]]))
        self.assertFalse(SSM.is_all_missing([[0], [0]]))
        self.assertFalse(SSM.is_all_missing([[0], [pd.NA]]))
        self.assertTrue(SSM.is_all_missing([[pd.NA]]))
        self.assertTrue(SSM.is_all_missing([[pd.NA], [pd.NA]]))
 
    def test_is_partial_missing(self):
        self.assertFalse(SSM.is_partial_missing(0))
        self.assertFalse(SSM.is_partial_missing(pd.NA))
        self.assertFalse(SSM.is_partial_missing([0,0]))
        self.assertTrue(SSM.is_partial_missing([0,pd.NA]))
        self.assertFalse(SSM.is_partial_missing([[0],[0]]))
        self.assertTrue(SSM.is_partial_missing([[0],[pd.NA]]))

    def test_remove_missing_rows(self):
        assert_array_equal(SSM.remove_missing_rows([[1,0],[0,1]], [1,pd.NA]), [[1,0]])
        assert_array_equal(SSM.remove_missing_rows([[1,0],[0,1]], [[1],[pd.NA]]), [[1,0]])
        self.assertEqual(SSM.remove_missing_rows([1,pd.NA], [1,pd.NA]), [1])
        self.assertEqual(SSM.remove_missing_rows([[1],[pd.NA]], [[1],[pd.NA]]), [[1]])

    def test_copy_missing(self):
        assert_array_equal(SSM.copy_missing([[0],[0]], [[1],[1]]), [[0],[0]])
        assert_array_equal(np.array(SSM.copy_missing([[0],[0]], [[1],[pd.NA]])).shape, (2,1))
        self.assertEqual(SSM.copy_missing([[0],[0]], [[1],[pd.NA]])[0][0], 0)
        assert SSM.copy_missing([[0],[0]], [[1],[pd.NA]])[1][0] is pd.NA
        assert_array_equal(SSM.copy_missing([0,0], [1,1]), [0,0])
        assert_array_equal(np.array(SSM.copy_missing([0,0], [1,pd.NA])).shape, (2,))
        self.assertEqual(SSM.copy_missing([0,0], [1,pd.NA])[0], 0)
        assert SSM.copy_missing([0,0], [1,pd.NA])[1] is pd.NA
        assert_array_equal(SSM.copy_missing([0], [1]), [0])
        assert_array_equal(np.array(SSM.copy_missing([0], [pd.NA])).shape, (1,))
        assert SSM.copy_missing([0], [pd.NA])[0] is pd.NA
        self.assertEqual(SSM.copy_missing(0, 1), 0)
        assert SSM.copy_missing(0, pd.NA) is pd.NA

    def test_adapt_row_to_any_missing_data(self):
        y_val = 2
        y_data_uv = pd.Series(np.full(1, y_val))
        y_missing_uv = pd.Series(np.full(1, pd.NA))
        y_val_mv = np.full((2,1), None)
        y_val_mv[0,0] = y_val
        y_missing_mv = pd.Series([y_val_mv])
        model_uv = md.get_local_level_model_data(1, 1, 1)
        model_mv = md.get_local_level_model_data(1, 1, np.identity(2))

        a0 = np.zeros((1,1))
        P0 = np.ones((1,1))

        ssm_data_uv = SSM(y_data_uv, model_uv, a0, P0)
        ssm_data_uv.adapt_row_to_any_missing_data(0,0)
        assert_array_equal(ssm_data_uv.Z[0], np.ones((1,1)))
        assert_array_equal(ssm_data_uv.v[0], np.full((1,1), y_val))

        ssm_missing_uv = SSM(y_missing_uv, model_uv, a0, P0)
        ssm_missing_uv.adapt_row_to_any_missing_data(0,0)
        assert_array_equal(ssm_missing_uv.Z[0], np.zeros((1,1)))
        assert_array_equal(ssm_missing_uv.v[0], np.zeros((1,1)))

        ssm_missing_mv = SSM(y_missing_mv, model_mv, a0, P0)
        ssm_missing_mv.adapt_row_to_any_missing_data(0,0)
        assert_array_equal(ssm_missing_mv.Z[0], np.ones((1,1)))
        assert_array_equal(ssm_missing_mv.d[0], np.zeros((1,1)))
        assert_array_equal(ssm_missing_mv.H[0], np.ones((1,1)))
        assert_array_equal(ssm_missing_mv.v[0], np.full((1,1), y_val))

    def test_set_up_initial_terms(self):
        y_val = 2
        y_data = pd.Series(np.full(1, y_val))
        model = md.get_local_level_model_data(1, 1, 1)
        a0 = np.zeros((1,1))
        P0 = np.ones((1,1))
        ssm_data = SSM(y_data, model, a0, P0)

        a0 = 3 * np.ones((3,1))
        P0 = 2 * np.identity(3)
        ssm_data.set_up_initial_terms(a0, P0, None)
        assert_array_equal(ssm_data.a_prior[0], a0)
        assert_array_equal(ssm_data.P_prior[0], P0)
        self.assertEqual(ssm_data.d_diffuse, -1)

        a0 = 4 * np.ones((3,1))
        P0 = 5 * np.identity(3)
        ssm_data.set_up_initial_terms(a0, P0, [False, False, False])
        assert_array_equal(ssm_data.a_prior[0], a0)
        assert_array_equal(ssm_data.P_prior[0], P0)
        self.assertEqual(ssm_data.d_diffuse, -1)

        expected_a0 = np.zeros((3,1))
        expected_a0[2,0] = a0[2,0]
        expected_P_infinity_prior = np.identity(3)
        expected_P_infinity_prior[2,2] = 0
        expected_P_star_prior = np.zeros((3,3))
        expected_P_star_prior[2,2] = P0[2,2]
        expected_P_prior_row = np.array([0,0,P0[2,2]])
        ssm_data.Z[ssm_data.initial_index] = np.ones((1, 3))
        ssm_data.set_up_initial_terms(a0, P0, [True, True, False])
        assert_array_equal(ssm_data.a_prior[0], expected_a0)
        assert_array_equal(ssm_data.P_infinity_prior[0], expected_P_infinity_prior)
        assert_array_equal(ssm_data.P_star_prior[0], expected_P_star_prior)
        self.assertTrue(np.isinf(ssm_data.P_prior[0][0,0]))
        self.assertTrue(np.isinf(ssm_data.P_prior[0][1,1]))
        assert_array_equal(ssm_data.P_prior[0][2,:], expected_P_prior_row)
        assert_array_equal(ssm_data.P_prior[0][:,2], expected_P_prior_row)
        self.assertEqual(ssm_data.d_diffuse, ssm_data.n)

        expected_a0 = a0.copy()
        expected_a0[0,0] = 0
        expected_P_infinity_prior = np.zeros((3,3))
        expected_P_infinity_prior[0,0] = 1
        expected_P_star_prior = np.zeros((3,3))
        expected_P_star_prior[1:,1:] = P0[1:,1:]
        expected_P_prior_row = np.array([0,P0[1,2],P0[2,2]])
        ssm_data.set_up_initial_terms(a0, P0, [True, False, False])
        assert_array_equal(ssm_data.a_prior[0], expected_a0)
        assert_array_equal(ssm_data.P_infinity_prior[0], expected_P_infinity_prior)
        assert_array_equal(ssm_data.P_star_prior[0], expected_P_star_prior)
        self.assertTrue(np.isinf(ssm_data.P_prior[0][0,0]))
        assert_array_equal(ssm_data.P_prior[0][2,:], expected_P_prior_row)
        assert_array_equal(ssm_data.P_prior[0][:,2], expected_P_prior_row)
        self.assertEqual(ssm_data.d_diffuse, ssm_data.n)

        expected_a0 = np.zeros((3,1))
        expected_P_infinity_prior = np.identity(3)
        expected_P_star_prior = np.zeros((3,3))
        ssm_data.set_up_initial_terms(a0, P0, [True, True, True])
        assert_array_equal(ssm_data.a_prior[0], expected_a0)
        assert_array_equal(ssm_data.P_infinity_prior[0], expected_P_infinity_prior)
        assert_array_equal(ssm_data.P_star_prior[0], expected_P_star_prior)
        self.assertTrue(np.isinf(ssm_data.P_prior[0][0,0]))
        self.assertTrue(np.isinf(ssm_data.P_prior[0][1,1]))
        self.assertTrue(np.isinf(ssm_data.P_prior[0][2,2]))
        self.assertEqual(ssm_data.d_diffuse, ssm_data.n)

    def test_filter_row(self):
        y_data = pd.Series([np.ones((1,1))])
        y_missing = pd.Series([np.full((1,1), pd.NA)])
        model_data = md.get_local_level_model_data(1, 1, 1)
        a0 = np.zeros((1,1))
        P0 = np.ones((1,1))

        ssm_data = SSM(y_data, model_data, a0, P0)
        ssm_data.filter()
        assert_array_equal(ssm_data.v[0], np.ones((1,1)))
        assert_array_equal(ssm_data.F[0], np.full((1,1), 2))
        assert_array_equal(ssm_data.F_inverse[0], np.full((1,1), 0.5))
        assert_array_equal(ssm_data.a_posterior[0], np.full((1,1), 0.5))
        assert_array_equal(ssm_data.P_posterior[0], np.full((1,1), 0.5))
        assert_array_equal(ssm_data.a_prior_final, np.full((1,1), 0.5))
        assert_array_equal(ssm_data.P_prior_final, np.full((1,1), 1.5))

        ssm_data = SSM(y_missing, model_data, a0, P0)
        ssm_data.filter()
        assert_array_equal(ssm_data.v[0], np.zeros((1,1)))
        assert_array_equal(ssm_data.F[0], np.full((1,1), 2))
        assert_array_equal(ssm_data.F_inverse[0], np.full((1,1), 1))
        assert_array_equal(ssm_data.a_posterior[0], np.full((1,1), 0))
        assert_array_equal(ssm_data.P_posterior[0], np.full((1,1), 1))
        assert_array_equal(ssm_data.a_prior_final, np.full((1,1), 0))
        assert_array_equal(ssm_data.P_prior_final, np.full((1,1), 2))

        ssm_data = SSM(y_data, model_data, a0, P0, [True])
        ssm_data.filter()
        assert_array_equal(ssm_data.v[0], np.ones((1,1)))
        assert_array_equal(ssm_data.F[0].shape, (1,1))
        self.assertTrue(np.isinf(ssm_data.F[0][0,0]))
        assert_array_equal(ssm_data.F_infinity[0], np.ones((1,1)))
        assert_array_equal(ssm_data.F_star[0], np.ones((1,1)))
        assert_array_equal(ssm_data.M_infinity[0], np.ones((1,1)))
        assert_array_equal(ssm_data.M_star[0], np.zeros((1,1)))
        assert_array_equal(ssm_data.F1[0], np.ones((1,1)))
        assert_array_equal(ssm_data.F2[0], np.full((1,1), -1))
        assert_array_equal(ssm_data.K0[0], np.ones((1,1)))
        assert_array_equal(ssm_data.K1[0], np.full((1,1), -1))
        assert_array_equal(ssm_data.L0[0], np.zeros((1,1)))
        assert_array_equal(ssm_data.L1[0], np.ones((1,1)))

        assert_array_equal(ssm_data.a_posterior[0], np.ones((1,1)))
        assert_array_equal(ssm_data.a_prior_final, np.ones((1,1)))
        assert_array_equal(ssm_data.P_infinity_posterior[0], np.zeros((1,1)))
        assert_array_equal(ssm_data.P_star_posterior[0], np.ones((1,1)))
        assert_array_equal(ssm_data.P_posterior[0], np.ones((1,1)))

        assert_array_equal(ssm_data.a_prior_final, np.full((1,1), 1))
        assert_array_equal(ssm_data.P_prior_final, np.full((1,1), 2))
        self.assertEqual(ssm_data.d_diffuse, 0)

        ssm_data = SSM(y_missing, model_data, a0, P0, [True])
        ssm_data.filter()
        assert_array_equal(ssm_data.a_prior[0], np.zeros((1,1)))
        assert_array_equal(ssm_data.P_infinity_prior[0], np.ones((1,1)))
        assert_array_equal(ssm_data.P_star_prior[0], np.zeros((1,1)))
        assert_array_equal(ssm_data.P_prior[0].shape, (1,1))
        self.assertTrue(np.isinf(ssm_data.P_prior[0][0,0]))
        assert_array_equal(ssm_data.v[0], np.zeros((1,1)))
        assert_array_equal(ssm_data.Z[0], np.zeros((1,1)))
        assert_array_equal(ssm_data.F[0].shape, (1,1))
        self.assertTrue(np.isinf(ssm_data.F[0][0,0]))
        assert_array_equal(ssm_data.F_inverse[0], np.full((1,1), 1))
        assert_array_equal(ssm_data.a_posterior[0], np.full((1,1), 0))
        assert_array_equal(ssm_data.P_infinity_posterior[0], ssm_data.P_infinity_prior[0])
        assert_array_equal(ssm_data.P_star_posterior[0], ssm_data.P_star_prior[0])
        assert_array_equal(ssm_data.P_posterior[0].shape, (1,1))
        self.assertTrue(np.isinf(ssm_data.P_posterior[0][0,0]))
        assert_array_equal(ssm_data.a_prior_final, np.full((1,1), 0))
        assert_array_equal(ssm_data.P_prior_final.shape, (1,1))
        self.assertTrue(np.isinf(ssm_data.P_prior_final[0,0]))
        self.assertEqual(ssm_data.d_diffuse, 1)

    def test_diffuse_P(self):
        P_star = np.full((1,1), 2.)
        result = SSM.diffuse_P(P_star, np.zeros((1,1)))
        assert_array_equal(result, P_star)

        result = SSM.diffuse_P(P_star, np.ones((1,1)))
        assert_array_equal(result.shape, (1,1))
        self.assertTrue(np.isinf(result[0,0]))

        P_star = np.identity(2)
        P_star[1,1] = 0
        P_infinity = np.identity(2)
        result = SSM.diffuse_P(P_star, P_infinity)
        self.assertTrue(np.isinf(result[0,0]))
        self.assertTrue(np.isinf(result[1,1]))


if __name__ == "__main__":
    unittest.main()
