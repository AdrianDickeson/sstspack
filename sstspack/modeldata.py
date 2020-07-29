'''
Created on 14 May 2020

@author: adriandickeson
'''

from math import cos, sin

from numpy import full, ones, zeros, identity, hstack, sum, vstack, pi as PI
import pandas as pd
from scipy.linalg.special_matrices import block_diag

def get_local_level_model_data(series_length, Q, H):
    '''
    '''
    Z = ones((1,1))
    Z, d, H = observation_terms(H, Z)
    T = ones((1,1))
    R = ones((1,1))
    c = zeros((1,1))
    Q = full((1,1), Q)

    result = get_static_model_df(series_length, Z=Z, d=d, H=H, T=T, c=c, R=R, Q=Q)
    return result

def get_local_linear_trend_model_data(series_length, Q, H):
    '''
    '''
    Z = ones((1,2))
    Z[0,1] = 0
    Z, d, H = observation_terms(H, Z)
    T = identity(2)
    T[0,1] = 1
    R = identity(2)
    c = zeros((2,1))

    result = get_static_model_df(series_length, Z=Z, d=d, H=H, T=T, c=c, R=R, Q=Q)
    return result

def get_time_domain_seasonal_model_data(series_length, s, sigma2_omega, H):
    '''
    '''
    Z = zeros((1,s))
    Z[0,0] = 1
    Z, d, H = observation_terms(H, Z)
    T = zeros((s,s))
    T[0,:-1] = -1
    for idx in range(1,s):
        T[idx,idx-1] = 1
    Q = full((1,1), sigma2_omega)
    R = zeros((s,1))
    R[0,0] = 1
    c = zeros((s,1))

    result = get_static_model_df(series_length, Z=Z, d=d, H=H, T=T, c=c, R=R, Q=Q)
    return result

def get_frequency_domain_seasonal_model_data(series_length, s, sigma2_omega, H):
    '''
    '''
    omega = 2 * PI / s
    submodels = []

    d = zeros((1,1))
    H = full((1, 1), H)
    i = 1
    while i <= s/2:
        Z, T, c, R, Q = frequency_domain_model_terms(i, s, omega, sigma2_omega[i-1])
        submodel = get_static_model_df(series_length, Z=Z, d=d, H=H, T=T, c=c, R=R, Q=Q)
        submodels.append(submodel)
        H = zeros((1,1))
        i += 1

    result = combine_model_data(submodels)
    return result

def frequency_domain_model_terms(i, s, omega, sigma2):
    '''
    '''
    Q = full((1, 1), sigma2)

    if i == s/2:
        Z = ones((1, 1))
        T = full((1, 1), -1)
        c = zeros((1, 1))
        R = ones((1, 1))
    else:
        Z = zeros((1, 2))
        Z[0,0] = 1
        cos_term = cos(i*omega)
        sin_term = sin(i*omega)
        T = zeros((2,2))
        T[0,:] = [cos_term, sin_term]
        T[1,:] = [-sin_term, cos_term]
        c = zeros((2,1))
        R = zeros((2,1))
        R[0,0] = 1

    return Z, T, c, R, Q

def get_ARMA_model_data(series_length, phi_terms, theta_terms, Q):
    '''
    '''
    return get_ARIMA_model_data(series_length, phi_terms, 0, theta_terms, Q)

def get_SARMA_model_data(series_length, s, PHI_terms, THETA_terms, Q):
    '''
    '''
    return get_ARMA_x_SARMA_model_data(series_length, [], [], s, PHI_terms, THETA_terms, Q)

def get_ARMA_x_SARMA_model_data(series_length, phi_terms, theta_terms,
                                s, PHI_terms, THETA_terms, Q):
    '''
    '''
    full_phi_terms = model_product(phi_terms, s, PHI_terms)
    full_theta_terms = model_product(theta_terms, s, THETA_terms)
    return get_ARIMA_model_data(series_length, full_phi_terms, 0, full_theta_terms, Q)

def get_ARIMA_model_data(series_length, phi_terms, d, theta_terms, Q):
    '''
    '''
    r = max(len(phi_terms), len(theta_terms)+1)

    H = zeros((1,1))
    d_term = zeros((1,1))
    Z = zeros((1,d+r))
    Z[0,:(d+1)] = 1

    phi_r = zeros(r)
    phi_r[:len(phi_terms)] = phi_terms
    theta_r = zeros(r)
    theta_r[0] = 1
    theta_r[1:(len(theta_terms)+1)] = theta_terms

    T = zeros((d+r,d+r))
    for idx in range(d):
        T[idx,idx:(d+1)] = 1
    T[d:,d] = phi_r
    for idx in range(1, r):
        T[d+idx-1, d+idx] = 1
    c = zeros((d+r,1))
    R = zeros((d+r,1))
    R[d:,0] = theta_r
    Q = full((1,1), Q)

    result = get_static_model_df(series_length, Z=Z, d=d_term, H=H, T=T, c=c, R=R, Q=Q)
    return result

def get_static_model_df(length, **kwargs):
    '''
    '''
    index = range(length)
    result = pd.DataFrame(index=index)

    for key in kwargs:
        result[key] = [kwargs[key]] * length

    return result

def combine_model_data(model_data_list):
    '''
    '''
    result = model_data_list[0].copy()

    for row in result.index:
        result.Z[row] = hstack([df.Z[row] for df in model_data_list])
        result.d[row] = sum([df.d[row] for df in model_data_list], keepdims=True)[0]
        result.H[row] = sum([df.H[row] for df in model_data_list], keepdims=True)[0]
        result.loc[row, 'T'] = block_diag(*[df.loc[row, 'T'] for df in model_data_list])
        result.c[row] = vstack([df.c[row] for df in model_data_list])
        result.R[row] = block_diag(*[df.R[row] for df in model_data_list])
        result.Q[row] = block_diag(*[df.Q[row] for df in model_data_list])

    return result

def observation_terms(H, Z):
    '''
    '''
    try:
        H = full((1,1), H)
    except ValueError:
        pass
    p = H.shape[1]
    Z = vstack([Z]*p)
    d = zeros((p,1))
    return Z, d, H

def model_product(standard_terms, s, seasonal_terms):
    '''
    '''
    result = zeros(s * len(seasonal_terms) + len(standard_terms))
    result[:len(standard_terms)] = standard_terms
    for idx, term in enumerate(seasonal_terms):
        result[s * (idx + 1) - 1] += term

    for idx1, standard_term in enumerate(standard_terms):
        for idx2, seasonal_term in enumerate(seasonal_terms):
            full_idx = s * (idx2 + 1) + idx1
            full_term = standard_term * seasonal_term
            result[full_idx] += full_term

    return result
