from pandas import Series, NA
from numpy import full, ones, zeros, identity
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


def test_smoother():
    """"""
    pass
