from numpy import identity, zeros, array, ones, full
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas import Series

from sstspack import ExtendedDynamicModel as EKF, DynamicLinearGaussianModel as DLGM
import sstspack.ExtendedModelDesign as emd
import sstspack.GaussianModelDesign as md
from sstspack.Utilities import identity_fn

expected_columns = array(
    [
        "y",
        "Z_fn",
        "H_fn",
        "T_fn",
        "R_fn",
        "Q_fn",
        "Z",
        "Z_prime",
        "H",
        "T",
        "T_prime",
        "R",
        "Q",
        "a_hat_initial",
        "V_initial",
        "Z_hat",
        "Z_hat_prime",
        "H_hat",
        "T_hat",
        "T_hat_prime",
        "R_hat",
        "Q_hat",
        "a_hat_prior",
        "a_hat_posterior",
        "P_hat_prior",
        "P_hat_posterior",
        "v_hat",
        "F_hat_inverse",
        "K_hat",
        "L_hat",
        "r_hat",
        "N_hat",
        "r0_hat",
        "r1_hat",
        "N0_hat",
        "N1_hat",
        "N2_hat",
        "y_original",
        "Z_original",
        "d_original",
        "H_original",
        "a_prior",
        "a_posterior",
        "P_prior",
        "P_posterior",
        "v",
        "F",
        "F_inverse",
        "K",
        "L",
        "a_hat",
        "r",
        "N",
        "V",
        "epsilon_hat",
        "epsilon_hat_sigma2",
        "eta_hat",
        "eta_hat_sigma2",
        "u",
        "D",
        "P_star_prior",
        "P_infinity_prior",
        "M_star",
        "M_infinity",
        "F1",
        "F2",
        "K0",
        "K1",
        "L0",
        "L1",
        "P_star_posterior",
        "P_infinity_posterior",
        "F_star",
        "F_infinity",
        "r0",
        "r1",
        "N0",
        "N1",
        "N2",
        "p",
    ]
)

index_standard = [1, 2, 3]
y_standard = Series([1, 2, 1], index=index_standard)


def get_standard_model_design():
    """"""
    a_initial = zeros((1, 1))
    P_initial = identity(1)

    def identity_matrix(state):
        return identity(1)

    Z_fn = T_fn = identity_fn
    H_fn = R_fn = Q_fn = identity_matrix

    model_design = emd.get_nonlinear_model_design(
        index_standard, Z_fn, T_fn, R_fn, Q_fn, H_fn
    )

    return model_design, a_initial, P_initial


def test___init___standard():
    """"""
    model_design, a_initial, P_initial = get_standard_model_design()

    model = EKF(y_standard, model_design, a_initial, P_initial)

    assert_array_equal(model.model_data_df.columns, expected_columns)


def test_filter():
    """"""
    expected_a_prior = Series(
        [zeros((1, 1)), full((1, 1), 0.5), full((1, 1), 1.4)], index=index_standard
    )
    expected_a_prior_final = full((1, 1), 1.153846)
    expected_P_prior = Series(
        [ones((1, 1)), full((1, 1), 1.5), full((1, 1), 1.6)], index=index_standard
    )
    expected_P_prior_final = full((1, 1), 1.615385)
    expected_a_posterior = Series(
        [full((1, 1), 0.5), full((1, 1), 1.4), full((1, 1), 1.153846)],
        index=index_standard,
    )
    expected_P_posterior = Series(
        [full((1, 1), 0.5), full((1, 1), 0.6), full((1, 1), 0.615385)],
        index=index_standard,
    )

    model_design, a_initial, P_initial = get_standard_model_design()

    model = EKF(y_standard, model_design, a_initial, P_initial)

    model.filter()

    dlgm_model_design = md.get_local_level_model_design(y_standard, 1, 1)
    dlgm_model = DLGM(y_standard, dlgm_model_design, a_initial, P_initial)
    dlgm_model.filter()

    assert model.filter_run
    for idx in model.index:
        assert_array_almost_equal(model.a_prior[idx], expected_a_prior[idx])
        assert_array_almost_equal(model.P_prior[idx], expected_P_prior[idx])
        assert_array_almost_equal(model.a_posterior[idx], expected_a_posterior[idx])
        assert_array_almost_equal(model.P_posterior[idx], expected_P_posterior[idx])

        assert_array_almost_equal(model.a_prior[idx], dlgm_model.a_prior[idx])
        assert_array_almost_equal(model.P_prior[idx], dlgm_model.P_prior[idx])
        assert_array_almost_equal(model.a_posterior[idx], dlgm_model.a_posterior[idx])
        assert_array_almost_equal(model.P_posterior[idx], dlgm_model.P_posterior[idx])

    assert_array_almost_equal(model.a_prior_final, expected_a_prior_final)
    assert_array_almost_equal(model.P_prior_final, expected_P_prior_final)

    assert_array_almost_equal(model.a_prior_final, dlgm_model.a_prior_final)
    assert_array_almost_equal(model.P_prior_final, dlgm_model.P_prior_final)
