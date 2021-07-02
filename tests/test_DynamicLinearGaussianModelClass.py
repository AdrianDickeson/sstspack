import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from sstspack import DynamicLinearGaussianModel as DLGM, GaussianModelDesign as md


class TestDynamicLinearGaussianModel(unittest.TestCase):
    def SetupLocalModel(self, series_length, sigma2_eta, H):
        self.local_model_design = md.get_local_level_model_design(
            series_length, sigma2_eta, H
        )

        self.a0 = np.zeros((1, 1))
        self.P0 = np.ones((1, 1))

        y_data = [
            0.085933226,
            0.054878689,
            0.080292212,
            0.321316458,
            0.307564319,
            0.532068456,
            0.263077943,
            0.231693705,
            0.427390255,
            0.569944428,
            0.658612762,
            0.690223242,
            0.711405893,
            0.818019523,
            0.646691109,
            0.840838002,
            0.943208824,
            0.683327325,
            1.011936804,
            0.980985747,
            0.94511906,
            0.939887808,
            0.893678285,
            0.839005343,
            0.891895152,
            1.131433676,
            1.064615366,
            1.051954086,
            0.81818596,
            0.908056985,
            0.865080444,
            0.826108072,
            0.995082652,
            0.822238422,
            0.801745378,
            0.906782053,
            0.870757007,
            0.690115185,
            0.567819417,
            0.557593299,
            0.708726663,
            0.535870638,
            0.527855858,
            0.45499549,
            0.333009208,
            0.300976424,
            0.344860468,
            0.28674572,
            -0.010297699,
            0.021895886,
            -0.042692693,
            -0.01879136,
            -0.234571879,
            -0.14654362,
            -0.217044867,
            -0.398849978,
            -0.399375493,
            -0.582486571,
            -0.476228583,
            -0.731831066,
            -0.625440522,
            -0.741851001,
            -0.786738307,
            -0.72425126,
            -0.620512675,
            -0.839570503,
            -1.083981301,
            -0.85990807,
            -0.773437416,
            -0.876128193,
            -0.965106566,
            -0.894352694,
            -1.05967849,
            -0.981618505,
            -0.817546424,
            -0.974246431,
            -1.039899116,
            -1.130202407,
            -1.0852465,
            -1.006398124,
            -0.906507315,
            -0.981829895,
            -0.728688095,
            -0.835400226,
            -0.706902319,
            -0.951300565,
            -0.765329634,
            -0.788774808,
            -0.560263498,
            -0.642565372,
            -0.703309488,
            -0.598771592,
            -0.51597972,
            -0.529250463,
            -0.363115832,
            -0.291532849,
            -0.134092451,
            -0.147029655,
            -0.183369639,
            -0.148558182,
        ]
        self.y = pd.Series(y_data, name="Observed")

    def GetLocalModel(self):
        return DLGM(self.y, self.local_model_design, self.a0, self.P0)

    def GetDiffuseLocalModel(self):
        return DLGM(self.y, self.local_model_design, self.a0, self.P0, [True])

    def setUp(self):
        # Setup local-level mpdel
        self.SetupLocalModel(series_length=100, sigma2_eta=0.002, H=0.01)

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
        local_model.y[26:28] = [[pd.NA], [pd.NA]]
        local_model.y[28:30] = [[np.nan], [np.nan]]
        local_model.y[30:32] = [[None], [None]]
        local_model.y[32:34] = [[[pd.NA]], [[pd.NA]]]
        local_model.y[34:36] = [[[np.nan]], [[np.nan]]]
        local_model.y[36:38] = [[[None]], [[None]]]
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
        self.assertAlmostEqual(
            sim_df["alpha"][0].ravel()[0] + sim_df["epsilon"][0].ravel()[0],
            local_model.y[0].ravel()[0],
        )

        local_model = self.GetDiffuseLocalModel()
        sim_df = local_model.simulate_smoother()
        self.assertAlmostEqual(
            sim_df["alpha"][0].ravel()[0] + sim_df["epsilon"][0].ravel()[0],
            local_model.y[0].ravel()[0],
        )

    def test_simulate_model(self):
        DLGM.simulate_model(self.local_model_design, self.a0, self.P0)

    def test_log_likelihood(self):
        local_model = self.GetLocalModel()
        log_lik = local_model.log_likelihood()
        self.assertAlmostEqual(log_lik, 36.032667773295756)

        local_model = self.GetDiffuseLocalModel()
        diffuse_log_lik = local_model.log_likelihood()
        self.assertAlmostEqual(diffuse_log_lik, 36.04596947782409)

        missing_y = self.y.copy()
        missing_y[20:30] = None
        local_model = DLGM(missing_y, self.local_model_design, self.a0, self.P0)
        log_lik = local_model.log_likelihood()
        self.assertAlmostEqual(log_lik, 28.047420202120488)

    def test_disturbance_smoother(self):
        local_model = self.GetLocalModel()
        local_model.disturbance_smoother()
        self.assertAlmostEqual(
            local_model.epsilon_hat[0].ravel()[0], -0.06554295424595682
        )
        self.assertAlmostEqual(
            local_model.epsilon_hat_sigma2[0].ravel()[0], 0.0035697866640171582
        )
        self.assertAlmostEqual(local_model.eta_hat[0].ravel()[0], 0.01341154320968328)
        self.assertAlmostEqual(
            local_model.eta_hat_sigma2[0].ravel()[0], 0.001741661575038556
        )

        local_model = self.GetDiffuseLocalModel()
        local_model.disturbance_smoother()
        self.assertAlmostEqual(
            local_model.epsilon_hat[0].ravel()[0], -0.06608562912767078
        )
        self.assertAlmostEqual(
            local_model.epsilon_hat_sigma2[0].ravel()[0], 0.0035825756949558396
        )
        self.assertAlmostEqual(local_model.eta_hat[0].ravel()[0], 0.013217125825534158)
        self.assertAlmostEqual(
            local_model.eta_hat_sigma2[0].ravel()[0], 0.0017433030277982336
        )

    def test_is_all_missing(self):
        self.assertFalse(DLGM.is_all_missing(0))
        self.assertTrue(DLGM.is_all_missing(pd.NA))
        self.assertTrue(DLGM.is_all_missing(None))
        self.assertTrue(DLGM.is_all_missing(np.nan))
        self.assertTrue(DLGM.is_all_missing(np.NaN))
        self.assertTrue(DLGM.is_all_missing(np.NAN))
        self.assertFalse(DLGM.is_all_missing([0]))
        self.assertFalse(DLGM.is_all_missing([0, 0]))
        self.assertFalse(DLGM.is_all_missing([0, pd.NA]))
        self.assertTrue(DLGM.is_all_missing([pd.NA]))
        self.assertTrue(DLGM.is_all_missing([pd.NA, pd.NA]))
        self.assertFalse(DLGM.is_all_missing([[0]]))
        self.assertFalse(DLGM.is_all_missing([[0], [0]]))
        self.assertFalse(DLGM.is_all_missing([[0], [pd.NA]]))
        self.assertTrue(DLGM.is_all_missing([[pd.NA]]))
        self.assertTrue(DLGM.is_all_missing([[pd.NA], [pd.NA]]))

    def test_is_partial_missing(self):
        self.assertFalse(DLGM.is_partial_missing(0))
        self.assertFalse(DLGM.is_partial_missing(pd.NA))
        self.assertFalse(DLGM.is_partial_missing([0, 0]))
        self.assertTrue(DLGM.is_partial_missing([0, pd.NA]))
        self.assertFalse(DLGM.is_partial_missing([[0], [0]]))
        self.assertTrue(DLGM.is_partial_missing([[0], [pd.NA]]))

    def test_remove_missing_rows(self):
        assert_array_equal(
            DLGM.remove_missing_rows([[1, 0], [0, 1]], [1, pd.NA]), [[1, 0]]
        )
        assert_array_equal(
            DLGM.remove_missing_rows([[1, 0], [0, 1]], [[1], [pd.NA]]), [[1, 0]]
        )
        self.assertEqual(DLGM.remove_missing_rows([1, pd.NA], [1, pd.NA]), [1])
        self.assertEqual(
            DLGM.remove_missing_rows([[1], [pd.NA]], [[1], [pd.NA]]), [[1]]
        )

    def test_copy_missing(self):
        assert_array_equal(DLGM.copy_missing([[0], [0]], [[1], [1]]), [[0], [0]])
        assert_array_equal(
            np.array(DLGM.copy_missing([[0], [0]], [[1], [pd.NA]])).shape, (2, 1)
        )
        self.assertEqual(DLGM.copy_missing([[0], [0]], [[1], [pd.NA]])[0][0], 0)
        assert DLGM.copy_missing([[0], [0]], [[1], [pd.NA]])[1][0] is pd.NA
        assert_array_equal(DLGM.copy_missing([0, 0], [1, 1]), [0, 0])
        assert_array_equal(np.array(DLGM.copy_missing([0, 0], [1, pd.NA])).shape, (2,))
        self.assertEqual(DLGM.copy_missing([0, 0], [1, pd.NA])[0], 0)
        assert DLGM.copy_missing([0, 0], [1, pd.NA])[1] is pd.NA
        assert_array_equal(DLGM.copy_missing([0], [1]), [0])
        assert_array_equal(np.array(DLGM.copy_missing([0], [pd.NA])).shape, (1,))
        assert DLGM.copy_missing([0], [pd.NA])[0] is pd.NA
        self.assertEqual(DLGM.copy_missing(0, 1), 0)
        assert DLGM.copy_missing(0, pd.NA) is pd.NA

    def test_adapt_row_to_any_missing_data(self):
        y_val = 2
        y_data_uv = pd.Series(np.full(1, y_val))
        y_missing_uv = pd.Series(np.full(1, pd.NA))
        y_val_mv = np.full((2, 1), None)
        y_val_mv[0, 0] = y_val
        y_missing_mv = pd.Series([y_val_mv])
        model_uv = md.get_local_level_model_design(1, 1, 1)
        model_mv = md.get_local_level_model_design(1, 1, np.identity(2))

        a0 = np.zeros((1, 1))
        P0 = np.ones((1, 1))

        ssm_data_uv = DLGM(y_data_uv, model_uv, a0, P0)
        ssm_data_uv.adapt_row_to_any_missing_data(0)
        assert_array_equal(ssm_data_uv.Z[0], np.ones((1, 1)))
        assert_array_equal(ssm_data_uv.v[0], np.full((1, 1), y_val))

        ssm_missing_uv = DLGM(y_missing_uv, model_uv, a0, P0)
        ssm_missing_uv.adapt_row_to_any_missing_data(0)
        assert_array_equal(ssm_missing_uv.Z[0], np.zeros((1, 1)))
        assert_array_equal(ssm_missing_uv.v[0], np.zeros((1, 1)))

        ssm_missing_mv = DLGM(y_missing_mv, model_mv, a0, P0)
        ssm_missing_mv.adapt_row_to_any_missing_data(0)
        assert_array_equal(ssm_missing_mv.Z[0], np.ones((1, 1)))
        assert_array_equal(ssm_missing_mv.d[0], np.zeros((1, 1)))
        assert_array_equal(ssm_missing_mv.H[0], np.ones((1, 1)))
        assert_array_equal(ssm_missing_mv.v[0], np.full((1, 1), y_val))

    def test_set_up_initial_terms(self):
        y_val = 2
        y_data = pd.Series(np.full(1, y_val))
        model = md.get_local_level_model_design(1, 1, 1)
        a0 = np.zeros((1, 1))
        P0 = np.ones((1, 1))
        ssm_data = DLGM(y_data, model, a0, P0)

        a0 = 3 * np.ones((3, 1))
        P0 = 2 * np.identity(3)
        ssm_data._set_up_initial_terms(a0, P0, None)
        assert_array_equal(ssm_data.a_prior[0], a0)
        assert_array_equal(ssm_data.P_prior[0], P0)
        self.assertEqual(ssm_data.d_diffuse, -1)

        a0 = 4 * np.ones((3, 1))
        P0 = 5 * np.identity(3)
        ssm_data._set_up_initial_terms(a0, P0, [False, False, False])
        assert_array_equal(ssm_data.a_prior[0], a0)
        assert_array_equal(ssm_data.P_prior[0], P0)
        self.assertEqual(ssm_data.d_diffuse, -1)

        expected_a0 = np.zeros((3, 1))
        expected_a0[2, 0] = a0[2, 0]
        expected_P_infinity_prior = np.identity(3)
        expected_P_infinity_prior[2, 2] = 0
        expected_P_star_prior = np.zeros((3, 3))
        expected_P_star_prior[2, 2] = P0[2, 2]
        expected_P_prior_row = np.array([0, 0, P0[2, 2]])
        ssm_data.Z[ssm_data.initial_index] = np.ones((1, 3))
        ssm_data._set_up_initial_terms(a0, P0, [True, True, False])
        assert_array_equal(ssm_data.a_prior[0], expected_a0)
        assert_array_equal(ssm_data.P_infinity_prior[0], expected_P_infinity_prior)
        assert_array_equal(ssm_data.P_star_prior[0], expected_P_star_prior)
        self.assertTrue(np.isinf(ssm_data.P_prior[0][0, 0]))
        self.assertTrue(np.isinf(ssm_data.P_prior[0][1, 1]))
        assert_array_equal(ssm_data.P_prior[0][2, :], expected_P_prior_row)
        assert_array_equal(ssm_data.P_prior[0][:, 2], expected_P_prior_row)
        self.assertEqual(ssm_data.d_diffuse, ssm_data.n)

        expected_a0 = a0.copy()
        expected_a0[0, 0] = 0
        expected_P_infinity_prior = np.zeros((3, 3))
        expected_P_infinity_prior[0, 0] = 1
        expected_P_star_prior = np.zeros((3, 3))
        expected_P_star_prior[1:, 1:] = P0[1:, 1:]
        expected_P_prior_row = np.array([0, P0[1, 2], P0[2, 2]])
        ssm_data._set_up_initial_terms(a0, P0, [True, False, False])
        assert_array_equal(ssm_data.a_prior[0], expected_a0)
        assert_array_equal(ssm_data.P_infinity_prior[0], expected_P_infinity_prior)
        assert_array_equal(ssm_data.P_star_prior[0], expected_P_star_prior)
        self.assertTrue(np.isinf(ssm_data.P_prior[0][0, 0]))
        assert_array_equal(ssm_data.P_prior[0][2, :], expected_P_prior_row)
        assert_array_equal(ssm_data.P_prior[0][:, 2], expected_P_prior_row)
        self.assertEqual(ssm_data.d_diffuse, ssm_data.n)

        expected_a0 = np.zeros((3, 1))
        expected_P_infinity_prior = np.identity(3)
        expected_P_star_prior = np.zeros((3, 3))
        ssm_data._set_up_initial_terms(a0, P0, [True, True, True])
        assert_array_equal(ssm_data.a_prior[0], expected_a0)
        assert_array_equal(ssm_data.P_infinity_prior[0], expected_P_infinity_prior)
        assert_array_equal(ssm_data.P_star_prior[0], expected_P_star_prior)
        self.assertTrue(np.isinf(ssm_data.P_prior[0][0, 0]))
        self.assertTrue(np.isinf(ssm_data.P_prior[0][1, 1]))
        self.assertTrue(np.isinf(ssm_data.P_prior[0][2, 2]))
        self.assertEqual(ssm_data.d_diffuse, ssm_data.n)

    def test_filter_row(self):
        y_data = pd.Series([np.ones((1, 1))])
        y_missing = pd.Series([np.full((1, 1), pd.NA)])
        model_data = md.get_local_level_model_design(1, 1, 1)
        a0 = np.zeros((1, 1))
        P0 = np.ones((1, 1))

        ssm_data = DLGM(y_data, model_data, a0, P0)
        ssm_data.filter()
        assert_array_equal(ssm_data.v[0], np.ones((1, 1)))
        assert_array_equal(ssm_data.F[0], np.full((1, 1), 2))
        assert_array_equal(ssm_data.F_inverse[0], np.full((1, 1), 0.5))
        assert_array_equal(ssm_data.a_posterior[0], np.full((1, 1), 0.5))
        assert_array_equal(ssm_data.P_posterior[0], np.full((1, 1), 0.5))
        assert_array_equal(ssm_data.a_prior_final, np.full((1, 1), 0.5))
        assert_array_equal(ssm_data.P_prior_final, np.full((1, 1), 1.5))

        ssm_data = DLGM(y_missing, model_data, a0, P0)
        ssm_data.filter()
        assert_array_equal(ssm_data.v[0], np.zeros((1, 1)))
        assert_array_equal(ssm_data.F[0], np.full((1, 1), 2))
        assert_array_equal(ssm_data.F_inverse[0], np.full((1, 1), 1))
        assert_array_equal(ssm_data.a_posterior[0], np.full((1, 1), 0))
        assert_array_equal(ssm_data.P_posterior[0], np.full((1, 1), 1))
        assert_array_equal(ssm_data.a_prior_final, np.full((1, 1), 0))
        assert_array_equal(ssm_data.P_prior_final, np.full((1, 1), 2))

        ssm_data = DLGM(y_data, model_data, a0, P0, [True])
        ssm_data.filter()
        assert_array_equal(ssm_data.v[0], np.ones((1, 1)))
        assert_array_equal(ssm_data.F[0].shape, (1, 1))
        self.assertTrue(np.isinf(ssm_data.F[0][0, 0]))
        assert_array_equal(ssm_data.F_infinity[0], np.ones((1, 1)))
        assert_array_equal(ssm_data.F_star[0], np.ones((1, 1)))
        assert_array_equal(ssm_data.M_infinity[0], np.ones((1, 1)))
        assert_array_equal(ssm_data.M_star[0], np.zeros((1, 1)))
        assert_array_equal(ssm_data.F1[0], np.ones((1, 1)))
        assert_array_equal(ssm_data.F2[0], np.full((1, 1), -1))
        assert_array_equal(ssm_data.K0[0], np.ones((1, 1)))
        assert_array_equal(ssm_data.K1[0], np.full((1, 1), -1))
        assert_array_equal(ssm_data.L0[0], np.zeros((1, 1)))
        assert_array_equal(ssm_data.L1[0], np.ones((1, 1)))

        assert_array_equal(ssm_data.a_posterior[0], np.ones((1, 1)))
        assert_array_equal(ssm_data.a_prior_final, np.ones((1, 1)))
        assert_array_equal(ssm_data.P_infinity_posterior[0], np.zeros((1, 1)))
        assert_array_equal(ssm_data.P_star_posterior[0], np.ones((1, 1)))
        assert_array_equal(ssm_data.P_posterior[0], np.ones((1, 1)))

        assert_array_equal(ssm_data.a_prior_final, np.full((1, 1), 1))
        assert_array_equal(ssm_data.P_prior_final, np.full((1, 1), 2))
        self.assertEqual(ssm_data.d_diffuse, 0)

        ssm_data = DLGM(y_missing, model_data, a0, P0, [True])
        ssm_data.filter()
        assert_array_equal(ssm_data.a_prior[0], np.zeros((1, 1)))
        assert_array_equal(ssm_data.P_infinity_prior[0], np.ones((1, 1)))
        assert_array_equal(ssm_data.P_star_prior[0], np.zeros((1, 1)))
        assert_array_equal(ssm_data.P_prior[0].shape, (1, 1))
        self.assertTrue(np.isinf(ssm_data.P_prior[0][0, 0]))
        assert_array_equal(ssm_data.v[0], np.zeros((1, 1)))
        assert_array_equal(ssm_data.Z[0], np.zeros((1, 1)))
        assert_array_equal(ssm_data.F[0].shape, (1, 1))
        self.assertTrue(np.isinf(ssm_data.F[0][0, 0]))
        assert_array_equal(ssm_data.F_inverse[0], np.full((1, 1), 1))
        assert_array_equal(ssm_data.a_posterior[0], np.full((1, 1), 0))
        assert_array_equal(
            ssm_data.P_infinity_posterior[0], ssm_data.P_infinity_prior[0]
        )
        assert_array_equal(ssm_data.P_star_posterior[0], ssm_data.P_star_prior[0])
        assert_array_equal(ssm_data.P_posterior[0].shape, (1, 1))
        self.assertTrue(np.isinf(ssm_data.P_posterior[0][0, 0]))
        assert_array_equal(ssm_data.a_prior_final, np.full((1, 1), 0))
        assert_array_equal(ssm_data.P_prior_final.shape, (1, 1))
        self.assertTrue(np.isinf(ssm_data.P_prior_final[0, 0]))
        self.assertEqual(ssm_data.d_diffuse, 1)

    def test_diffuse_P(self):
        P_star = np.full((1, 1), 2.0)
        result = DLGM.diffuse_P(P_star, np.zeros((1, 1)))
        assert_array_equal(result, P_star)

        result = DLGM.diffuse_P(P_star, np.ones((1, 1)))
        assert_array_equal(result.shape, (1, 1))
        self.assertTrue(np.isinf(result[0, 0]))

        P_star = np.identity(2)
        P_star[1, 1] = 0
        P_infinity = np.identity(2)
        result = DLGM.diffuse_P(P_star, P_infinity)
        self.assertTrue(np.isinf(result[0, 0]))
        self.assertTrue(np.isinf(result[1, 1]))


if __name__ == "__main__":
    unittest.main()
