from math import cos, sin
from copy import copy

from numpy import full, ones, zeros, identity, hstack, vstack, pi as PI, diag, block
import pandas as pd
from scipy.linalg.special_matrices import block_diag


# TODO: enable input of parameters as arrays
def get_local_level_model_design(length_index, Q, H, dt=None):
    """"""
    Z = ones((1, 1))
    d = zeros((1, 1))
    T = ones((1, 1))
    R = ones((1, 1))
    c = zeros((1, 1))

    Z, d, H, T, c, R, Q = process_terms(H, Z, d, Q, T, c, R)
    result = get_static_model_df(length_index, Z=Z, d=d, H=H, T=T, c=c, R=R, Q=Q)

    if not dt is None:
        for i, idx in enumerate(result.index):
            result.Q[idx] = result.Q[idx] * dt[i]

    return result


def get_local_linear_trend_model_design(length_index, Q, H, dt=None):
    """"""
    Z = ones((1, 2))
    Z[0, 1] = 0
    d = zeros((1, 1))
    T = identity(2)
    T[0, 1] = 1
    R = identity(2)
    c = zeros((2, 1))

    Z, d, H, T, c, R, Q = process_terms(H, Z, d, Q, T, c, R)
    result = get_static_model_df(length_index, Z=Z, d=d, H=H, T=T, c=c, R=R, Q=Q)

    if not dt is None:
        for i, idx in enumerate(result.index):
            T = result.loc[idx, "T"]
            T[0, 1] = dt[i]
            result.loc[idx, "T"] = T

            sigma2_zeta_delta = result.Q[idx][1, 1] * dt[i]
            Q = zeros((2, 2))
            Q[0, 0] = 0.25 * dt[i] * dt[i]
            Q[0, 1] = Q[1, 0] = 0.5 * dt[i]
            Q[1, 1] = 1
            Q = sigma2_zeta_delta * Q
            result.Q[idx] = Q

    return result


def get_time_domain_seasonal_model_design(length_index, s, sigma2_omega, H):
    """"""
    Z = zeros((1, s))
    Z[0, 0] = 1
    d = zeros((1, 1))
    T = zeros((s, s))
    T[0, :-1] = -1
    for idx in range(1, s):
        T[idx, idx - 1] = 1
    Q = full((1, 1), sigma2_omega)
    R = zeros((s, 1))
    R[0, 0] = 1
    c = zeros((s, 1))

    Z, d, H, T, c, R, Q = process_terms(H, Z, d, Q, T, c, R)
    result = get_static_model_df(length_index, Z=Z, d=d, H=H, T=T, c=c, R=R, Q=Q)
    return result


def get_frequency_domain_seasonal_model_design(length_index, s, Q_list, H):
    """"""
    omega = 2 * PI / s
    submodels = []

    d = zeros((1, 1))
    try:
        H = full((1, 1), H)
    except ValueError:
        pass
    p = H.shape[1]
    i = 1
    while i <= s / 2:
        Z, T, c, R, Q = frequency_domain_model_terms(i, s, omega, Q_list[i - 1])
        Z, d_i, H, T, c, R, Q = process_terms(H, Z, d, Q, T, c, R)
        submodel = get_static_model_df(
            length_index, Z=Z, d=d_i, H=H, T=T, c=c, R=R, Q=Q
        )
        submodels.append(submodel)
        H = zeros((p, p))
        i += 1

    result = combine_model_design(submodels)
    return result


def frequency_domain_model_terms(i, s, omega, Q):
    """"""
    if i == s / 2:
        Q_i = copy(Q)
        Z = ones((1, 1))
        T = full((1, 1), -1)
        c = zeros((1, 1))
        R = ones((1, 1))
    else:
        try:
            _ = Q[0, 0]
        except (TypeError, IndexError):
            Q_i = Q * identity(2)
        else:
            p = Q.shape[1]
            Q_i = zeros((2 * p, 2 * p))
            for idx in range(p):
                Q_i[2 * idx, 2 * idx] = Q[idx, idx]
                Q_i[2 * idx + 1, 2 * idx + 1] = Q[idx, idx]
                for idx2 in range(idx + 1, p):
                    Q_i[2 * idx, 2 * idx2] = Q[idx, idx2]
                    Q_i[2 * idx2, 2 * idx] = Q[idx2, idx]
                    Q_i[2 * idx + 1, 2 * idx2 + 1] = Q[idx, idx2]
                    Q_i[2 * idx2 + 1, 2 * idx + 1] = Q[idx2, idx]
        Z = zeros((1, 2))
        Z[0, 0] = 1
        cos_term = cos(i * omega)
        sin_term = sin(i * omega)
        T = zeros((2, 2))
        T[0, :] = [cos_term, sin_term]
        T[1, :] = [-sin_term, cos_term]
        c = zeros((2, 1))
        R = identity(2)

    return Z, T, c, R, Q_i


def get_ARMA_model_design(series_length_index, phi_terms, theta_terms, Q):
    """"""
    return get_ARIMA_model_design(series_length_index, phi_terms, 0, theta_terms, Q)


def get_SARMA_model_design(series_length_index, s, PHI_terms, THETA_terms, Q):
    """"""
    return get_ARMA_x_SARMA_model_design(
        series_length_index, [], [], s, PHI_terms, THETA_terms, Q
    )


def get_ARMA_x_SARMA_model_design(
    series_length_index, phi_terms, theta_terms, s, PHI_terms, THETA_terms, Q
):
    """"""
    return get_ARIMA_x_SARIMA_model_design(
        series_length_index, phi_terms, 0, theta_terms, s, PHI_terms, 0, THETA_terms, Q
    )


def get_ARIMA_model_design(series_length_index, phi_terms, d, theta_terms, Q):
    """"""
    return get_ARIMA_x_SARIMA_model_design(
        series_length_index, phi_terms, d, theta_terms, 1, [], 0, [], Q
    )


def get_SARIMA_model_design(series_length_index, s, PHI_terms, D, THETA_terms, Q):
    """"""
    return get_ARIMA_x_SARIMA_model_design(
        series_length_index, [], 0, [], s, PHI_terms, D, THETA_terms, Q
    )


def get_ARIMA_x_SARIMA_model_design(
    series_length_index, phi_terms, d, theta_terms, s, PHI_terms, D, THETA_terms, Q
):
    """"""
    full_phi_terms = model_product(phi_terms, s, PHI_terms)
    full_theta_terms = model_product(theta_terms, s, THETA_terms)
    r = max(len(full_phi_terms), len(full_theta_terms) + 1)

    difference_length_index = d + s * D
    state_length_index = difference_length_index + r
    H = zeros((1, 1))
    d_term = zeros((1, 1))
    Z = zeros((1, state_length_index))

    phi_r = zeros(r)
    phi_r[: len(full_phi_terms)] = full_phi_terms
    theta_r = zeros(r)
    theta_r[0] = 1
    theta_r[1 : (len(full_theta_terms) + 1)] = full_theta_terms

    T = zeros((state_length_index, state_length_index))

    difference_terms = zeros(state_length_index)
    difference_terms[difference_length_index] = 1
    for idx in range(D):
        colidx = difference_length_index - s * idx - 1
        difference_terms[colidx] = 1
        for _ in range(s - 1):
            T[colidx, colidx - 1] = 1
            colidx -= 1
        T[colidx, :] = difference_terms.copy()
    for idx in reversed(range(d)):
        difference_terms[idx] = 1
        T[idx, :] = difference_terms.copy()

    T[difference_length_index:, difference_length_index] = phi_r
    for idx in range(1, r):
        T[difference_length_index + idx - 1, difference_length_index + idx] = 1
    c = zeros((state_length_index, 1))
    R = zeros((state_length_index, 1))
    R[difference_length_index:, 0] = theta_r
    Q = full((1, 1), Q)

    Z[0, :] = difference_terms.copy()

    Z, d, H, T, c, R, Q = process_terms(H, Z, d, Q, T, c, R)
    result = get_static_model_df(
        series_length_index, Z=Z, d=d_term, H=H, T=T, c=c, R=R, Q=Q
    )
    return result


def get_intervention_model_design(length_index, intervention_point, Q=0, H=0):
    """"""
    try:
        H = full((1, 1), H)
    except ValueError:
        pass
    p = H.shape[1]

    try:
        Q = full((1, 1), Q)
    except ValueError:
        pass
    m = Q.shape[1]

    Z = zeros((p, m))
    d = zeros((p, 1))
    T = identity(m)
    c = zeros((m, 1))
    R = zeros((m, m))
    Q1 = zeros((m, m))

    try:
        _ = iter(length_index)
    except TypeError:
        prior_index = list(range(intervention_point))
        post_index = list(range(intervention_point, length_index))
    else:
        intervention_idx = list(length_index).index(intervention_point)
        prior_index = [
            value for idx, value in enumerate(length_index) if idx < intervention_idx
        ]
        post_index = [
            value for idx, value in enumerate(length_index) if idx >= intervention_idx
        ]

    result_prior = get_static_model_df(prior_index, Z=Z, d=d, H=H, T=T, c=c, R=R, Q=Q1)

    Z = identity(p)
    R = identity(m)

    result_post = get_static_model_df(post_index, Z=Z, d=d, H=H, T=T, c=c, R=R, Q=Q)

    return pd.concat([result_prior, result_post])


def get_time_varying_regression_model_design(length_index, regressors_df, Q, H):
    """"""
    try:
        _ = iter(length_index)
    except TypeError:
        index = range(length_index)
    else:
        index = length_index

    m = len(regressors_df.columns)
    Z_master = zeros((1, m))
    c_master = zeros((m, 1))
    T_master = identity(m)
    d_master = zeros((1, 1))
    R_master = identity(m)

    submodels = []
    for idx in index:
        Z = Z_master.copy()
        Z[0, :] = regressors_df.loc[idx, :]
        Z, d, H, T, c, R, Q = process_terms(
            H, Z, d_master, Q, T_master, c_master, R_master
        )
        submodels.append(get_static_model_df([idx], Z=Z, d=d, H=H, T=T, c=c, R=R, Q=Q))

    result = pd.concat(submodels)
    return result


def get_static_model_df(length_index, **kwargs):
    """"""
    y_timeseries = None
    try:
        if type(length_index) != list:
            y_timeseries = length_index
            length_index = y_timeseries.index
    except AttributeError:
        y_timeseries = None

    try:
        _ = iter(length_index)
    except TypeError:
        index = range(length_index)
        length = length_index
    else:
        index = length_index
        length = len(length_index)

    result = pd.DataFrame(index=index)
    for key in kwargs:
        result[key] = [kwargs[key]] * length

    adjustment_columns = ["Z", "d", "H"]
    if (not y_timeseries is None) and all(
        [col in result.columns for col in adjustment_columns]
    ):
        for idx in index:
            p_model = result.Z[idx].shape[0]
            p_series = len(y_timeseries[idx])
            p_diff = p_series - p_model

            if p_series > p_model:
                result.Z[idx] = vstack([result.Z[idx]] + [result.Z[idx][0, :]] * p_diff)
                result.d[idx] = vstack([result.d[idx]] + [zeros((1, 1))] * p_diff)

                H = zeros((p_series, p_series))
                H[:p_model, :p_model] = result.H[idx]
                for diag_idx in range(p_model, p_series):
                    H[diag_idx, diag_idx] = result.H[idx][0, 0]
                result.H[idx] = H

    return result


def combine_model_design(model_data_list):
    """"""
    result = model_data_list[0].copy()

    for row in result.index:
        result.Z[row] = hstack([df.Z[row] for df in model_data_list])
        result.d[row] = sum([df.d[row] for df in model_data_list])
        result.H[row] = sum([df.H[row] for df in model_data_list])
        result.loc[row, "T"] = block_diag(*[df.loc[row, "T"] for df in model_data_list])
        result.c[row] = vstack([df.c[row] for df in model_data_list])
        result.R[row] = block_diag(*[df.R[row] for df in model_data_list])
        result.Q[row] = block_diag(*[df.Q[row] for df in model_data_list])

        retain_rv = [val != 0 for val in diag(result.Q[row])]
        m = result.Z[row].shape[1]
        if all(not val for val in retain_rv):
            result.R[row] = zeros((m, 1))
            result.Q[row] = zeros((1, 1))
        elif any(not val for val in retain_rv):
            result.R[row] = result.R[row][:, retain_rv]
            result.Q[row] = result.Q[row][retain_rv, :]
            result.Q[row] = result.Q[row][:, retain_rv]

    return result


def process_terms(H, Z, d, Q, T, c, R):
    """"""
    try:
        H = full((1, 1), H)
    except ValueError:
        pass
    p = H.shape[1]

    try:
        Q = full((1, 1), Q)
    except ValueError:
        pass
    q_len = Q.shape[1] // R.shape[1]

    if q_len == 1:
        Z = vstack([Z] * p)
    else:
        Z = block_diag(*[Z] * p)
        T = block_diag(*[T] * q_len)
        R = block_diag(*[R] * q_len)

    d = vstack([d] * p)
    c = vstack([c] * q_len)

    return Z, d, H, T, c, R, Q


def model_product(standard_terms, s, seasonal_terms):
    """"""
    result = zeros(s * len(seasonal_terms) + len(standard_terms))
    result[: len(standard_terms)] = standard_terms
    for idx, term in enumerate(seasonal_terms):
        result[s * (idx + 1) - 1] += term

    for idx1, standard_term in enumerate(standard_terms):
        for idx2, seasonal_term in enumerate(seasonal_terms):
            full_idx = s * (idx2 + 1) + idx1
            full_term = standard_term * seasonal_term
            result[full_idx] += full_term

    return result


def get_spline_smoothing_model_design(length_index, lambda_term, H, dt=None):
    """"""
    if dt is None:
        Z = zeros((1, 2))
        Z[0, 0] = 1
        d = zeros((1, 1))
        T = zeros((2, 2))
        T[0, 0] = 2
        T[0, 1] = -1
        T[1, 0] = 1
        c = zeros((1, 1))
        R = zeros((2, 1))
        R[0, 0] = 1
        Q = H / lambda_term

        Z, d, H, T, c, R, Q = process_terms(H, Z, d, Q, T, c, R)
        result = get_static_model_df(length_index, Z=Z, d=d, H=H, T=T, c=c, R=R, Q=Q)

        return result

    try:
        H = full((1, 1), H)
    except ValueError:
        pass

    Q = zeros((2, 2))
    Q[1, 1] = H[0, 0] / lambda_term
    result = get_local_linear_trend_model_design(length_index, Q, H, dt)

    return result
