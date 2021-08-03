from pandas import Series, NA
from numpy import full, inf, ones, zeros, identity, isinf, NAN, nan, NaN, array
from numpy.testing import assert_array_equal, assert_array_almost_equal

import sstspack.GaussianModelDesign as md
from sstspack.DynamicLinearGaussianModelClass import DynamicLinearGaussianModel as DLGM

expected_columns = (
    "y",
    "Z",
    "d",
    "H",
    "T",
    "c",
    "R",
    "Q",
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
)
index_standard = [1, 2, 3]
y_standard = Series([1, 2, 1], index_standard)


def get_standard_local_model_parameters():
    """"""
    Q = full((1, 1), 0.5)
    H = ones((1, 1))
    model_design = md.get_local_level_model_design(y_standard, Q, H)

    a_initial = zeros((1, 1))
    P_initial = identity(1)

    return model_design, a_initial, P_initial


def test___init___standard():
    """"""
    model_design, a_initial, P_initial = get_standard_local_model_parameters()

    model = DLGM(y_standard, model_design, a_initial, P_initial)

    assert all(col in model.model_data_df.columns for col in expected_columns)
    assert_array_equal(model.index, index_standard)


def test___init___diffuse():
    """"""
    model_design, _, _ = get_standard_local_model_parameters()

    model = DLGM(y_standard, model_design, diffuse_states=[True])

    assert all(col in model.model_data_df.columns for col in expected_columns)
    assert_array_equal(model.index, index_standard)


def test_filter():
    """"""
    expected_a_prior = Series(
        [zeros((1, 1)), full((1, 1), 0.5), full((1, 1), 1.25)], index=index_standard
    )
    expected_a_prior_final = full((1, 1), 1.125)
    expected_P_prior = Series(
        [ones((1, 1)), ones((1, 1)), ones((1, 1))], index=index_standard
    )
    expected_P_prior_final = ones((1, 1))
    expected_a_posterior = Series(
        [full((1, 1), 0.5), full((1, 1), 1.25), full((1, 1), 1.125)],
        index=index_standard,
    )
    expected_P_posterior = Series(
        [full((1, 1), 0.5), full((1, 1), 0.5), full((1, 1), 0.5)], index=index_standard
    )

    model_design, a_initial, P_initial = get_standard_local_model_parameters()
    model = DLGM(y_standard, model_design, a_initial, P_initial)

    model.filter()

    assert model.filter_run
    for idx in model.index:
        assert_array_almost_equal(model.a_prior[idx], expected_a_prior[idx])
        assert_array_almost_equal(model.P_prior[idx], expected_P_prior[idx])
        assert_array_almost_equal(model.a_posterior[idx], expected_a_posterior[idx])
        assert_array_almost_equal(model.P_posterior[idx], expected_P_posterior[idx])

    assert_array_almost_equal(model.a_prior_final, expected_a_prior_final)
    assert_array_almost_equal(model.P_prior_final, expected_P_prior_final)


def test_filter_missing():
    """"""
    expected_a_prior = Series(
        [zeros((1, 1)), zeros((1, 1)), full((1, 1), 1.2)], index=index_standard
    )
    expected_a_prior_final = full((1, 1), 1.095238)
    expected_P_prior = Series(
        [ones((1, 1)), full((1, 1), 1.5), full((1, 1), 1.1)], index=index_standard
    )
    expected_P_prior_final = full((1, 1), 1.02381)
    expected_a_posterior = Series(
        [zeros((1, 1)), full((1, 1), 1.2), full((1, 1), 1.095238)],
        index=index_standard,
    )
    expected_P_posterior = Series(
        [full((1, 1), 1), full((1, 1), 0.6), full((1, 1), 0.52381)],
        index=index_standard,
    )

    y_missing = y_standard.copy()
    y_missing[1] = NA

    model_design, a_initial, P_initial = get_standard_local_model_parameters()
    model = DLGM(y_missing, model_design, a_initial, P_initial)

    model.filter()

    for idx in model.index:
        assert_array_almost_equal(model.a_prior[idx], expected_a_prior[idx])
        assert_array_almost_equal(model.P_prior[idx], expected_P_prior[idx])
        assert_array_almost_equal(model.a_posterior[idx], expected_a_posterior[idx])
        assert_array_almost_equal(model.P_posterior[idx], expected_P_posterior[idx])

    assert_array_almost_equal(model.a_prior_final, expected_a_prior_final)
    assert_array_almost_equal(model.P_prior_final, expected_P_prior_final)


def test_filter_diffuse():
    """"""
    expected_a_prior = Series(
        [ones((1, 1)), full((1, 1), 1.6)], index=index_standard[1:]
    )
    expected_a_prior_final = full((1, 1), 1.285714)
    expected_P_prior = Series(
        [full((1, 1), 1.5), full((1, 1), 1.1)], index=index_standard[1:]
    )
    expected_P_prior_final = full((1, 1), 1.02381)
    expected_a_posterior = Series(
        [ones((1, 1)), full((1, 1), 1.6), full((1, 1), 1.285714)],
        index=index_standard,
    )
    expected_P_posterior = Series(
        [full((1, 1), 1), full((1, 1), 0.6), full((1, 1), 0.52381)],
        index=index_standard,
    )

    model_design, _, _ = get_standard_local_model_parameters()
    model = DLGM(y_standard, model_design, diffuse_states=[True])

    model.filter()

    assert_array_almost_equal(
        model.a_posterior[model.initial_index],
        expected_a_posterior[model.initial_index],
    )
    assert_array_almost_equal(
        model.P_posterior[model.initial_index],
        expected_P_posterior[model.initial_index],
    )
    for idx in model.non_diffuse_index:
        assert_array_almost_equal(model.a_prior[idx], expected_a_prior[idx])
        assert_array_almost_equal(model.P_prior[idx], expected_P_prior[idx])
        assert_array_almost_equal(model.a_posterior[idx], expected_a_posterior[idx])
        assert_array_almost_equal(model.P_posterior[idx], expected_P_posterior[idx])

    assert_array_almost_equal(model.a_prior_final, expected_a_prior_final)
    assert_array_almost_equal(model.P_prior_final, expected_P_prior_final)


def test_smoother():
    """"""
    expected_a_hat = Series(
        [full((1, 1), 0.84375), full((1, 1), 1.1875), full((1, 1), 1.125)],
        index=index_standard,
    )
    expected_V = Series(
        [full((1, 1), 0.34375), full((1, 1), 0.375), full((1, 1), 0.5)],
        index=index_standard,
    )

    model_design, a_initial, P_initial = get_standard_local_model_parameters()
    model = DLGM(y_standard, model_design, a_initial, P_initial)

    model.smoother()

    assert model.smoother_run
    for idx in model.index:
        assert_array_almost_equal(model.a_hat[idx], expected_a_hat[idx])
        assert_array_almost_equal(model.V[idx], expected_V[idx])


def test_smoother_missing():
    """"""
    expected_a_hat = Series(
        [full((1, 1), 0.761905), full((1, 1), 1.142857), full((1, 1), 1.095238)],
        index=index_standard,
    )
    expected_V = Series(
        [full((1, 1), 0.52381), full((1, 1), 0.428571), full((1, 1), 0.52381)],
        index=index_standard,
    )

    y_missing = y_standard.copy()
    y_missing[1] = NA

    model_design, a_initial, P_initial = get_standard_local_model_parameters()
    model = DLGM(y_missing, model_design, a_initial, P_initial)

    model.smoother()

    for idx in model.index:
        assert_array_almost_equal(model.a_hat[idx], expected_a_hat[idx])
        assert_array_almost_equal(model.V[idx], expected_V[idx])


def test_smoother_diffuse():
    """"""
    expected_a_hat = Series(
        [full((1, 1), 1.285714), full((1, 1), 1.428571), full((1, 1), 1.285714)],
        index=index_standard,
    )
    expected_V = Series(
        [full((1, 1), 0.52381), full((1, 1), 0.428571), full((1, 1), 0.52381)],
        index=index_standard,
    )

    model_design, _, _ = get_standard_local_model_parameters()
    model = DLGM(y_standard, model_design, diffuse_states=[True])

    model.smoother()

    assert model.smoother_run
    for idx in model.index:
        assert_array_almost_equal(model.a_hat[idx], expected_a_hat[idx])
        assert_array_almost_equal(model.V[idx], expected_V[idx])


def test_log_likelihood():
    """"""
    expected_log_likelihood = -4.6246613704539365

    model_design, a_initial, P_initial = get_standard_local_model_parameters()
    model = DLGM(y_standard, model_design, a_initial, P_initial)

    assert model.log_likelihood() == expected_log_likelihood


def test_disturbance_smoother():
    """"""
    expected_epsilon_hat = Series(
        [full((1, 1), 0.15625), full((1, 1), 0.8125), full((1, 1), -0.125)],
        index=index_standard,
    )
    expected_epsilon_hat_sigma2 = Series(
        [full((1, 1), 0.34375), full((1, 1), 0.375), full((1, 1), 0.5)],
        index=index_standard,
    )
    expected_eta_hat = Series(
        [full((1, 1), 0.34375), full((1, 1), -0.0625), full((1, 1), 0.0)],
        index=index_standard,
    )
    expected_eta_hat_sigma2 = Series(
        [full((1, 1), 0.34375), full((1, 1), 0.375), full((1, 1), 0.5)],
        index=index_standard,
    )

    model_design, a_initial, P_initial = get_standard_local_model_parameters()
    model = DLGM(y_standard, model_design, a_initial, P_initial)

    model.disturbance_smoother()

    assert model.disturbance_smoother_run
    for idx in model.index:
        assert_array_almost_equal(model.epsilon_hat[idx], expected_epsilon_hat[idx])
        assert_array_almost_equal(
            model.epsilon_hat_sigma2[idx], expected_epsilon_hat_sigma2[idx]
        )
        assert_array_almost_equal(model.eta_hat[idx], expected_eta_hat[idx])
        assert_array_almost_equal(
            model.eta_hat_sigma2[idx], expected_eta_hat_sigma2[idx]
        )


def test_is_all_missing():
    assert not DLGM.is_all_missing(0)
    assert DLGM.is_all_missing(NA)
    assert DLGM.is_all_missing(None)
    assert DLGM.is_all_missing(nan)
    assert DLGM.is_all_missing(NaN)
    assert DLGM.is_all_missing(NAN)
    assert not DLGM.is_all_missing([0])
    assert not DLGM.is_all_missing([0, 0])
    assert not DLGM.is_all_missing([0, NA])
    assert DLGM.is_all_missing([NA])
    assert DLGM.is_all_missing([NA, NA])
    assert not DLGM.is_all_missing([[0]])
    assert not DLGM.is_all_missing([[0], [0]])
    assert not DLGM.is_all_missing([[0], [NA]])
    assert DLGM.is_all_missing([[NA]])
    assert DLGM.is_all_missing([[NA], [NA]])


def test_is_partial_missing():
    assert not DLGM.is_partial_missing(0)
    assert not DLGM.is_partial_missing(NA)
    assert not DLGM.is_partial_missing([0, 0])
    assert DLGM.is_partial_missing([0, NA])
    assert not DLGM.is_partial_missing([[0], [0]])
    assert DLGM.is_partial_missing([[0], [NA]])


def test_remove_missing_rows():
    assert_array_equal(DLGM.remove_missing_rows([[1, 0], [0, 1]], [1, NA]), [[1, 0]])
    assert_array_equal(
        DLGM.remove_missing_rows([[1, 0], [0, 1]], [[1], [NA]]), [[1, 0]]
    )
    assert DLGM.remove_missing_rows([1, NA], [1, NA]) == [1]
    assert DLGM.remove_missing_rows([[1], [NA]], [[1], [NA]]) == [[1]]


def test_copy_missing():
    assert_array_equal(DLGM.copy_missing([[0], [0]], [[1], [1]]), [[0], [0]])
    assert_array_equal(array(DLGM.copy_missing([[0], [0]], [[1], [NA]])).shape, (2, 1))
    assert DLGM.copy_missing([[0], [0]], [[1], [NA]])[0][0] == 0
    assert DLGM.copy_missing([[0], [0]], [[1], [NA]])[1][0] is NA
    assert_array_equal(DLGM.copy_missing([0, 0], [1, 1]), [0, 0])
    assert_array_equal(array(DLGM.copy_missing([0, 0], [1, NA])).shape, (2,))
    assert DLGM.copy_missing([0, 0], [1, NA])[0] == 0
    assert DLGM.copy_missing([0, 0], [1, NA])[1] is NA
    assert_array_equal(DLGM.copy_missing([0], [1]), [0])
    assert_array_equal(array(DLGM.copy_missing([0], [NA])).shape, (1,))
    assert DLGM.copy_missing([0], [NA])[0] is NA
    assert DLGM.copy_missing(0, 1) == 0
    assert DLGM.copy_missing(0, NA) is NA


def test_adapt_row_to_any_missing_data():
    y_val = 2
    y_data_uv = Series(full(1, y_val))
    y_missing_uv = Series(full(1, NA))
    y_val_mv = full((2, 1), None)
    y_val_mv[0, 0] = y_val
    y_missing_mv = Series([y_val_mv])
    model_uv = md.get_local_level_model_design(1, 1, 1)
    model_mv = md.get_local_level_model_design(1, 1, identity(2))

    a0 = zeros((1, 1))
    P0 = ones((1, 1))

    ssm_data_uv = DLGM(y_data_uv, model_uv, a0, P0)
    ssm_data_uv.adapt_row_to_any_missing_data(0)
    assert_array_equal(ssm_data_uv.Z[0], ones((1, 1)))
    assert_array_equal(ssm_data_uv.y[0], full((1, 1), y_val))

    ssm_missing_uv = DLGM(y_missing_uv, model_uv, a0, P0)
    ssm_missing_uv.adapt_row_to_any_missing_data(0)
    assert_array_equal(ssm_missing_uv.Z[0], zeros((1, 1)))
    assert ssm_missing_uv.y[0][0, 0] is NA

    ssm_missing_mv = DLGM(y_missing_mv, model_mv, a0, P0)
    ssm_missing_mv.adapt_row_to_any_missing_data(0)
    assert_array_equal(ssm_missing_mv.Z[0], ones((1, 1)))
    assert_array_equal(ssm_missing_mv.d[0], zeros((1, 1)))
    assert_array_equal(ssm_missing_mv.H[0], ones((1, 1)))
    assert_array_equal(ssm_missing_mv.y[0], full((1, 1), y_val))


def test_set_up_initial_terms():
    y_val = 2
    y_data = Series(full(1, y_val))
    model = md.get_local_level_model_design(1, 1, 1)
    a0 = zeros((1, 1))
    P0 = ones((1, 1))
    ssm_data = DLGM(y_data, model, a0, P0)

    a0 = 3 * ones((3, 1))
    P0 = 2 * identity(3)
    ssm_data._set_up_initial_terms(a0, P0, None)
    assert_array_equal(ssm_data.a_prior[0], a0)
    assert_array_equal(ssm_data.P_prior[0], P0)
    assert ssm_data.d_diffuse == -1

    a0 = 4 * ones((3, 1))
    P0 = 5 * identity(3)
    ssm_data._set_up_initial_terms(a0, P0, [False, False, False])
    assert_array_equal(ssm_data.a_prior[0], a0)
    assert_array_equal(ssm_data.P_prior[0], P0)
    assert ssm_data.d_diffuse == -1

    expected_a0 = zeros((3, 1))
    expected_a0[2, 0] = a0[2, 0]
    expected_P_infinity_prior = identity(3)
    expected_P_infinity_prior[2, 2] = 0
    expected_P_star_prior = zeros((3, 3))
    expected_P_star_prior[2, 2] = P0[2, 2]
    expected_P_prior_row = array([0, 0, P0[2, 2]])
    ssm_data.Z[ssm_data.initial_index] = ones((1, 3))
    ssm_data._set_up_initial_terms(a0, P0, [True, True, False])
    assert_array_equal(ssm_data.a_prior[0], expected_a0)
    assert_array_equal(ssm_data.P_infinity_prior[0], expected_P_infinity_prior)
    assert_array_equal(ssm_data.P_star_prior[0], expected_P_star_prior)
    assert isinf(ssm_data.P_prior[0][0, 0])
    assert isinf(ssm_data.P_prior[0][1, 1])
    assert_array_equal(ssm_data.P_prior[0][2, :], expected_P_prior_row)
    assert_array_equal(ssm_data.P_prior[0][:, 2], expected_P_prior_row)
    assert ssm_data.d_diffuse == ssm_data.n

    expected_a0 = a0.copy()
    expected_a0[0, 0] = 0
    expected_P_infinity_prior = zeros((3, 3))
    expected_P_infinity_prior[0, 0] = 1
    expected_P_star_prior = zeros((3, 3))
    expected_P_star_prior[1:, 1:] = P0[1:, 1:]
    expected_P_prior_row = array([0, P0[1, 2], P0[2, 2]])
    ssm_data._set_up_initial_terms(a0, P0, [True, False, False])
    assert_array_equal(ssm_data.a_prior[0], expected_a0)
    assert_array_equal(ssm_data.P_infinity_prior[0], expected_P_infinity_prior)
    assert_array_equal(ssm_data.P_star_prior[0], expected_P_star_prior)
    assert isinf(ssm_data.P_prior[0][0, 0])
    assert_array_equal(ssm_data.P_prior[0][2, :], expected_P_prior_row)
    assert_array_equal(ssm_data.P_prior[0][:, 2], expected_P_prior_row)
    assert ssm_data.d_diffuse == ssm_data.n

    expected_a0 = zeros((3, 1))
    expected_P_infinity_prior = identity(3)
    expected_P_star_prior = zeros((3, 3))
    ssm_data._set_up_initial_terms(a0, P0, [True, True, True])
    assert_array_equal(ssm_data.a_prior[0], expected_a0)
    assert_array_equal(ssm_data.P_infinity_prior[0], expected_P_infinity_prior)
    assert_array_equal(ssm_data.P_star_prior[0], expected_P_star_prior)
    assert isinf(ssm_data.P_prior[0][0, 0])
    assert isinf(ssm_data.P_prior[0][1, 1])
    assert isinf(ssm_data.P_prior[0][2, 2])
    assert ssm_data.d_diffuse == ssm_data.n


def test_filter_row():
    y_data = Series([ones((1, 1))])
    y_missing = Series([full((1, 1), NA)])
    model_data = md.get_local_level_model_design(1, 1, 1)
    a0 = zeros((1, 1))
    P0 = ones((1, 1))

    ssm_data = DLGM(y_data, model_data, a0, P0)
    ssm_data.filter()
    assert_array_equal(ssm_data.v[0], ones((1, 1)))
    assert_array_equal(ssm_data.F[0], full((1, 1), 2))
    assert_array_equal(ssm_data.F_inverse[0], full((1, 1), 0.5))
    assert_array_equal(ssm_data.a_posterior[0], full((1, 1), 0.5))
    assert_array_equal(ssm_data.P_posterior[0], full((1, 1), 0.5))
    assert_array_equal(ssm_data.a_prior_final, full((1, 1), 0.5))
    assert_array_equal(ssm_data.P_prior_final, full((1, 1), 1.5))

    ssm_data = DLGM(y_missing, model_data, a0, P0)
    ssm_data.filter()
    assert_array_equal(ssm_data.v[0], zeros((1, 1)))
    assert_array_equal(ssm_data.F[0], full((1, 1), 2))
    assert_array_equal(ssm_data.F_inverse[0], full((1, 1), 1))
    assert_array_equal(ssm_data.a_posterior[0], full((1, 1), 0))
    assert_array_equal(ssm_data.P_posterior[0], full((1, 1), 1))
    assert_array_equal(ssm_data.a_prior_final, full((1, 1), 0))
    assert_array_equal(ssm_data.P_prior_final, full((1, 1), 2))

    ssm_data = DLGM(y_data, model_data, a0, P0, [True])
    ssm_data.filter()
    assert_array_equal(ssm_data.v[0], ones((1, 1)))
    assert_array_equal(ssm_data.F[0].shape, (1, 1))
    assert isinf(ssm_data.F[0][0, 0])
    assert_array_equal(ssm_data.F_infinity[0], ones((1, 1)))
    assert_array_equal(ssm_data.F_star[0], ones((1, 1)))
    assert_array_equal(ssm_data.M_infinity[0], ones((1, 1)))
    assert_array_equal(ssm_data.M_star[0], zeros((1, 1)))
    assert_array_equal(ssm_data.F1[0], ones((1, 1)))
    assert_array_equal(ssm_data.F2[0], full((1, 1), -1))
    assert_array_equal(ssm_data.K0[0], ones((1, 1)))
    assert_array_equal(ssm_data.K1[0], full((1, 1), -1))
    assert_array_equal(ssm_data.L0[0], zeros((1, 1)))
    assert_array_equal(ssm_data.L1[0], ones((1, 1)))

    assert_array_equal(ssm_data.a_posterior[0], ones((1, 1)))
    assert_array_equal(ssm_data.a_prior_final, ones((1, 1)))
    assert_array_equal(ssm_data.P_infinity_posterior[0], zeros((1, 1)))
    assert_array_equal(ssm_data.P_star_posterior[0], ones((1, 1)))
    assert_array_equal(ssm_data.P_posterior[0], ones((1, 1)))

    assert_array_equal(ssm_data.a_prior_final, full((1, 1), 1))
    assert_array_equal(ssm_data.P_prior_final, full((1, 1), 2))
    assert ssm_data.d_diffuse == 0

    ssm_data = DLGM(y_missing, model_data, a0, P0, [True])
    ssm_data.filter()
    assert_array_equal(ssm_data.a_prior[0], zeros((1, 1)))
    assert_array_equal(ssm_data.P_infinity_prior[0], ones((1, 1)))
    assert_array_equal(ssm_data.P_star_prior[0], zeros((1, 1)))
    assert_array_equal(ssm_data.P_prior[0].shape, (1, 1))
    assert isinf(ssm_data.P_prior[0][0, 0])
    assert_array_equal(ssm_data.v[0], zeros((1, 1)))
    assert_array_equal(ssm_data.Z[0], zeros((1, 1)))
    assert_array_equal(ssm_data.F[0].shape, (1, 1))
    assert isinf(ssm_data.F[0][0, 0])
    assert_array_equal(ssm_data.F_inverse[0], full((1, 1), 1))
    assert_array_equal(ssm_data.a_posterior[0], full((1, 1), 0))
    assert_array_equal(ssm_data.P_infinity_posterior[0], ssm_data.P_infinity_prior[0])
    assert_array_equal(ssm_data.P_star_posterior[0], ssm_data.P_star_prior[0])
    assert_array_equal(ssm_data.P_posterior[0].shape, (1, 1))
    assert isinf(ssm_data.P_posterior[0][0, 0])
    assert_array_equal(ssm_data.a_prior_final, full((1, 1), 0))
    assert_array_equal(ssm_data.P_prior_final.shape, (1, 1))
    assert isinf(ssm_data.P_prior_final[0, 0])
    assert ssm_data.d_diffuse == 1


def test_diffuse_P():
    P_star = full((1, 1), 2.0)
    result = DLGM.diffuse_P(P_star, zeros((1, 1)))
    assert_array_equal(result, P_star)

    result = DLGM.diffuse_P(P_star, ones((1, 1)))
    assert_array_equal(result.shape, (1, 1))
    assert isinf(result[0, 0])

    P_star = identity(2)
    P_star[1, 1] = 0
    P_infinity = identity(2)
    result = DLGM.diffuse_P(P_star, P_infinity)
    assert isinf(result[0, 0])
    assert isinf(result[1, 1])
