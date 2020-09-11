'''
Created on 18 May 2020

@author: adriandickeson
'''
from numpy import zeros, ones, identity, hstack, dot
import matplotlib.pyplot as plt

from sstspack import StateSpaceModel as SSM, modeldata as md

def plot_sim(model_sim, title, plot_idx, model_df):
    '''
    '''
    fig, axs = plt.subplots(2, 1)
    ax = axs[0]

    model_sim['Z'] = model_df['Z']
    y_hat = hstack(model_sim.apply(lambda x: dot(x['Z'], x['alpha']), axis=1))

    ax.plot(model_sim.index, y_hat[0,:], c='black')
    ax.scatter(model_sim.index, model_sim['y'])
    ax.set_title(title)

    ax = axs[1]

    alpha = hstack(model_sim['alpha'])
    ax.plot(model_sim.index, alpha[plot_idx,:], c='black')

if __name__ == '__main__':
    a0 = zeros((1,1))
    P0 = ones((1,1))
    local_level_df = md.get_local_level_model_data(100, 10, 100)
    local_level_sim = SSM.simulate_model(local_level_df, a0, P0)

    plot_sim(local_level_sim, 'Local level model', 0, local_level_df)


    a0 = zeros((2,1))
    P0 = identity(2)
    H = 100
    Q = 0.01 * identity(2)
    local_linear_trend_df = md.get_local_linear_trend_model_data(100, Q, H)
    local_linear_trend_sim = SSM.simulate_model(local_linear_trend_df, a0, P0)

    plot_sim(local_linear_trend_sim, 'Local linear trend model', 1, local_linear_trend_df)


    s = 10
    a0 = zeros((s,1))
    a0[:,0] = [-5*(i - 0.5*(s - 1)) for i in range(s)]
    P0 = identity(s)
    H = 100
    sigma2_omega = 1
    time_domain_seasonal_df = md.get_time_domain_seasonal_model_data(100, s, sigma2_omega, H)
    time_domain_seasonal_sim = SSM.simulate_model(time_domain_seasonal_df, a0, P0)

    plot_sim(time_domain_seasonal_sim, 'Time domain seasonal model', 0, time_domain_seasonal_df)


    a0 = zeros((s+1,1))
    a0[1:,0] = [-5*(i - 0.5*(s - 1)) for i in range(s)]
    P0 = identity(s+1)
    combined_df = md.combine_model_data([local_level_df, time_domain_seasonal_df])
    combined_sim = SSM.simulate_model(combined_df, a0, P0)

    plot_sim(combined_sim, 'Combined model', 0, combined_df)


    s = 10
    a0 = zeros((s-1,1))
    a0[:,0] = [-5*(i - 0.5*(s - 1)) for i in range(s-1)]
    P0 = identity(s-1)
    H = 100
    sigma2_omega = ones(5)
    frequency_domain_seasonal_df = md.get_frequency_domain_seasonal_model_data(100, s, sigma2_omega,
                                                                               H)
    frequency_domain_seasonal_sim = SSM.simulate_model(frequency_domain_seasonal_df, a0, P0)

    plot_sim(frequency_domain_seasonal_sim, 'Frequency domain seasonal model', 0,
             frequency_domain_seasonal_df)


    a0 = zeros((1,1))
    P0 = identity(1)
    Q = 1
    phi_terms = [0.9]
    theta_terms = []
    ARMA_df = md.get_ARMA_model_data(100, phi_terms, theta_terms, Q)
    ARMA_sim = SSM.simulate_model(ARMA_df, a0, P0)

    plot_sim(ARMA_sim, 'ARMA model', 0, ARMA_df)


    phi_terms = [0.4]
    a0 = zeros((3,1))
    P0 = identity(3)
    ARIMA_df = md.get_ARIMA_model_data(100, phi_terms, 2, theta_terms, Q)
    ARIMA_sim = SSM.simulate_model(ARIMA_df, a0, P0)

    plot_sim(ARIMA_sim, 'ARIMA model', 1, ARIMA_df)


    a0 = zeros((s,1))
    a0[:,0] = [-5*(i - 0.5*s) for i in range(s)]
    P0 = identity(s)
    SARMA_df = md.get_SARMA_model_data(100, s, phi_terms, theta_terms, Q)
    SARMA_sim = SSM.simulate_model(SARMA_df, a0, P0)

    plot_sim(SARMA_sim, 'SARMA model', 0, SARMA_df)


    a0 = zeros((s+1, 1))
    a0[:,0] = [-5*(i - 0.5*s) for i in range(s+1)]
    P0 = identity(s+1)
    ARMA_x_SARMA_df = md.get_ARMA_x_SARMA_model_data(100, phi_terms, theta_terms,
                                                     s, phi_terms, theta_terms, Q)
    ARMA_x_SARMA_sim = SSM.simulate_model(ARMA_x_SARMA_df, a0, P0)

    plot_sim(ARMA_x_SARMA_sim, 'ARMA x SARMA model', 0, ARMA_x_SARMA_df)


    D = 1
    a0 = zeros((2*s,1))
    a0[:,0] = [-5*(i - 0.5*s) for i in range(2*s)]
    P0 = identity(2*s)
    SARIMA_df = md.get_SARIMA_model_data(100, s, phi_terms, D, theta_terms, Q)
    SARIMA_sim = SSM.simulate_model(SARIMA_df, a0, P0)

    plot_sim(SARIMA_sim, 'SARIMA model', 0, SARIMA_df)


    d = D = 1
    a0 = zeros((2 * s+2, 1))
    a0[:,0] = [-5*(i - 0.5*s) for i in range(2 * s+2)]
    P0 = identity(2 * s+2)
    ARIMA_x_SARIMA_df = md.get_ARIMA_x_SARIMA_model_data(100, phi_terms, d, theta_terms,
                                                         s, phi_terms, D, theta_terms, Q)
    ARIMA_x_SARIMA_sim = SSM.simulate_model(ARIMA_x_SARIMA_df, a0, P0)

    plot_sim(ARIMA_x_SARIMA_sim, 'ARIMA x SARiMA model', 0, ARIMA_x_SARIMA_df)


    plt.show()
