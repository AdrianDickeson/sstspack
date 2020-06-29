'''
Created on 28 Apr 2020

@author: adriandickeson
'''
import pandas as pd
import matplotlib.pyplot as plt
from numpy import sqrt, linspace, array
from scipy.stats import norm, gaussian_kde
from numpy.random import normal

from sstspack import StateSpaceModel as SSM

TITLE = 'Volume of Nile river at Aswan 1871-1970'
XLIM = (1868, 1973)
XLABEL = 'year'
FIG_LAYOUT = [0, 0.03, 1, 0.95]

def get_fig_data(ssmodel, state_col, error_col, confidence = 0.9):
    '''
    ''' 
    percentile = 0.5 + 0.5 * confidence
    quantile = norm.ppf(percentile)
    state_error = ssmodel.model_data_df[error_col].apply(lambda x: quantile * sqrt(x.ravel()[0]))
    est_state = ssmodel.model_data_df[state_col].apply(lambda x: x.ravel()[0])

    x_vals = [0] + list(ssmodel.y.index) + [2000]
    upper_state = list(est_state + state_error)
    upper_state = [upper_state[0]] + upper_state + [upper_state[-1]]
    lower_state = list(est_state - state_error)
    lower_state = [lower_state[0]] + lower_state + [lower_state[-1]]
    est_state = list(est_state)
    est_state = [est_state[0]] + est_state + [est_state[-1]]

    data_dict = {'est_state': est_state,
                 'lower_state': lower_state,
                 'upper_state': upper_state}
    result = pd.DataFrame(data_dict, index=x_vals)
    return result

def plot_state(ax, ssmodel, state_data_df, legend_text, missing_mask = None, xlim = XLIM,
               confidence = 0.9):
    '''
    '''
    if missing_mask is None:
        missing_mask = ssmodel.y.apply(lambda _: True)

    d1 = ax.scatter(x=ssmodel.y.index[missing_mask], y=ssmodel.y[missing_mask], marker='x', s=50.)
    d2, = ax.plot(state_data_df['est_state'], 'r', linewidth=2., c='red')
    ax.plot(state_data_df['upper_state'], '--', c='orange')
    ax.plot(state_data_df['lower_state'], '--', c='orange')
    d3 = ax.fill_between(state_data_df.index, state_data_df['upper_state'],
                         state_data_df['lower_state'], alpha=0.5)
    ax.set_title('Nile volume data')
    ax.set_xlim(xlim)
    ax.set_ylim(500, 1400)
    ax.set_ylabel('volume')
    ax.legend((d1, d2, d3), ('Observed', legend_text, '{:.0f}% conf. int.'.format(100*confidence)),
              loc='upper right', fontsize=5)

def plot_line(ax, data_series, title, ylim, ylabel, xlabel = None, xlim = XLIM):
    '''
    '''
    ax.plot(data_series)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_scatter_line(ax, data_series, title, ylim, ylabel, xlabel = None):
    '''
    '''
    ax.scatter(x=data_series.index, y=data_series, marker='x', s=50.)
    ax.plot(data_series, '--', c='blue')
    ax.set_title(title)
    ax.set_xlim(XLIM)
    ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_line_and_scatter(ax, line_data, scatter_data, title, ylim, ylabel, xlabel = None):
    '''
    '''
    ax.plot(line_data, '-', c='black')
    ax.scatter(x=scatter_data.index, y=scatter_data, marker='x', s=25.)
    ax.set_title(title)
    ax.set_xlim(XLIM)
    ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_histogram(ax, data_series, title, xlim, ylabel):
    '''
    '''
    ax.hist(data_series, bins = 13, density = True)
    ax.set_xlim(xlim)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    density = gaussian_kde(data_series)
    density.covariance_factor = lambda : 0.5
    density._compute_covariance()
    x_vals = linspace(-3.5, 3.0, 200)
    ax.plot(x_vals, density(x_vals), c='red')

def plot_qq(ax, data_series, title, limit):
    '''
    '''
    confidence_data = normal(size=(len(data_series), 10000))
    for i in range(10000):
        confidence_data[:,i] = sorted(confidence_data[:,i])
    for i in range(len(data_series)):
        confidence_data[i,:] = sorted(confidence_data[i,:])
    lower_bound = confidence_data[:,49]
    upper_bound = confidence_data[:,9949]

    ordered_data = sorted(data_series)
    percentiles = (1 + array(range(len(data_series)))) / (len(data_series) + 1)
    quantiles = norm.ppf(percentiles)

    d2 = ax.fill_between(quantiles, upper_bound, lower_bound, alpha=0.5, color='orange')
    ax.plot(quantiles, upper_bound, c='red')
    ax.plot(quantiles, lower_bound, c='red')
    d1 = ax.scatter(y=ordered_data, x=quantiles, marker='x', s=25, c='blue')
    ax.plot(limit, limit, c='black')

    ax.set_title(title)
    ax.set_xlim(quantiles[0], quantiles[-1])
    ax.set_ylim(limit)
    ax.set_xlabel('expected')
    ax.set_ylabel('observed')
    ax.legend((d1, d2), ('Observed', '{:.0f}% conf. int.'.format(99)), loc='upper left', fontsize=5)

def plot_correlogram(ax, correl_data, variance, title):
    '''
    '''
    xlim = (0, len(correl_data)+1)
    bound = array([norm.ppf(0.995) * sqrt(variance), norm.ppf(0.995) * sqrt(variance)])

    d2 = ax.fill_between(xlim, bound, -bound, alpha=0.5, color='orange')
    ax.plot(xlim, bound, c='red')
    ax.plot(xlim, -bound, c='red')

    ax.bar(range(1, len(correl_data)+1), correl_data)
    ax.set_ylim(-1, 1)
    ax.set_xlim(xlim)
    ax.set_title(title)
    ax.legend((d2,), ('{:.0f}% conf. int.'.format(99),), loc='upper right', fontsize=5)

def auto_covariance(data_series, lag):
    '''
    '''
    shifted_data = data_series.shift(periods=lag)
    result = (data_series * shifted_data).iloc[lag:].sum() / len(data_series)
    return result

def auto_correlation(data_series, lag):
    '''
    '''
    n = len(data_series)
    variance = data_series.apply(lambda x: x * x).sum() / n
    result = auto_covariance(data_series, lag) / variance
    return result

def plot_fig21(ssmodel):
    '''
    '''
    state_data_df = get_fig_data(ssmodel, 'a_prior', 'P_prior')

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('{} - Fig. 2.1'.format(TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_state(ax, ssmodel, state_data_df, 'Filtered mean')

    ax = axs[0, 1]
    plot_line(ax, ssmodel.P_prior, 'P_prior (P_t)', (5000, 17500), 'variance')

    ax = axs[1, 0]
    data_series = ssmodel.y - ssmodel.a_prior
    plot_scatter_line(ax, data_series, 'Forecast error', (-450, 450), 'error')

    ax = axs[1, 1]
    plot_line(ax, ssmodel.F, 'F (F_t)', (20000, 32500), 'variance', XLABEL)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig('figures/fig2.1.pdf')

def plot_fig22(ssmodel):
    '''
    '''
    state_data_df = get_fig_data(ssmodel, 'a_hat', 'V')

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('{} - Fig. 2.2'.format(TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_state(ax, ssmodel, state_data_df, 'Smoothed mean')

    ax = axs[0, 1]
    plot_line(ax, ssmodel.V, 'V (V_t)', (2200, 4100), 'variance')

    ax = axs[1, 0]
    plot_scatter_line(ax, ssmodel.r, 'Smoothing cumulant r (r_t)', (-0.04, 0.024),  'error', XLABEL)

    ax = axs[1, 1]
    plot_line(ax, ssmodel.N, 'N (N_t)', (0.000048, 0.00011), 'variance', XLABEL)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig('figures/fig2.2.pdf')

def plot_fig23(ssmodel):
    '''
    '''
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('{} - Fig. 2.3'.format(TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_scatter_line(ax, ssmodel.epsilon_hat, 'epsilon_hat', (-380, 300),  'error')

    ax = axs[0, 1]
    plot_line(ax, ssmodel.epsilon_hat_sigma2, 'epsilon_hat_sigma2', None, 'variance')

    ax = axs[1, 0]
    plot_scatter_line(ax, ssmodel.eta_hat, 'eta_hat', (-55, 35),  'error', XLABEL)

    ax = axs[1, 1]
    plot_line(ax, ssmodel.eta_hat_sigma2, 'eta_hat_sigma2', None, 'variance', XLABEL)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig('figures/fig2.3.pdf')

def plot_fig24(ssmodel, sim):
    '''
    '''
    initial_key = sim.index[0]
    a0 = ssmodel.a_hat[initial_key]
    P0 = ssmodel.V[initial_key]
    model_fields = ['Z', 'd', 'H', 'T', 'c', 'R', 'Q']
    model_df = ssmodel.model_data_df[model_fields].copy()
    new_sim = SSM.simulate_model(model_df, a0, P0)
    plot_ssm = SSM(new_sim.y, model_df, a0, P0)
    plot_ssm.smoother()
    plot_sim = sim.alpha - ssmodel.a_hat + plot_ssm.a_hat

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('{} - Fig. 2.4'.format(TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_line_and_scatter(ax, ssmodel.a_hat, plot_sim, 'a_hat simulation',
                          None, 'volume')

    ax = axs[0, 1]
    plot_line_and_scatter(ax, ssmodel.a_hat, sim.alpha, 'a_hat conditional simulation',
                          None, 'volume')

    ax = axs[1, 0]
    plot_line_and_scatter(ax, ssmodel.epsilon_hat, sim.epsilon,
                          'epsilon_hat conditional simulation', None,  'error', XLABEL)

    ax = axs[1, 1]
    plot_line_and_scatter(ax, ssmodel.eta_hat[:-1], sim.eta[:-1], 'eta_hat conditional simulation',
                          None, 'error', XLABEL)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig('figures/fig2.4.pdf')

def plot_fig25(ssmodel):
    '''
    '''
    missing_mask = ssmodel.y.apply(lambda x: not SSM.is_all_missing(x))
    state_data_df = get_fig_data(ssmodel, 'a_prior', 'P_prior')

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('{} - Fig. 2.5'.format(TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_state(ax, ssmodel, state_data_df, 'Filtered mean', missing_mask)

    ax = axs[0, 1]
    plot_line(ax, ssmodel.P_prior, 'P_prior (P_t)', (5000, 36000), 'variance')

    state_data_df = get_fig_data(ssmodel, 'a_hat', 'V')

    ax = axs[1, 0]
    plot_state(ax, ssmodel, state_data_df, 'Smoothed mean', missing_mask)

    ax = axs[1, 1]
    plot_line(ax, ssmodel.V, 'V (V_t)', (2200, 10000), 'variance', XLABEL)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig('figures/fig2.5.pdf')

def plot_fig26(ssmodel):
    '''
    '''
    missing_mask = ssmodel.y.apply(lambda x: not SSM.is_all_missing(x))
    state_data_df = get_fig_data(ssmodel, 'a_prior', 'P_prior', confidence = 0.5)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('{} - Fig. 2.6'.format(TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_state(ax, ssmodel, state_data_df, 'Filtered mean', missing_mask, xlim = (1868,2003),
               confidence = 0.5)

    ax = axs[0, 1]
    plot_line(ax, ssmodel.P_prior, 'P_prior (P_t)', (5000, 50000), 'variance', xlim = (1868,2003))

    ax = axs[1, 0]
    plot_line(ax, ssmodel.a_prior, 'a_prior (a_t)', (700, 1250), 'volume',
              XLABEL, xlim = (1868,2003))

    ax = axs[1, 1]
    plot_line(ax, ssmodel.F, 'F (F_t)', (20000, 65000), 'variance', XLABEL,
              xlim = (1868,2003))

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig('figures/fig2.6.pdf')

def plot_fig27(ssmodel):
    '''
    '''
    data_series = ssmodel.model_data_df.apply(lambda x: x['v'].ravel()[0] / sqrt(x['F'].ravel()[0]),
                                              axis=1)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('{} - Fig. 2.7'.format(TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_scatter_line(ax, data_series, 'Standardised residuals - epsilon', (-3.5, 3), 'error')

    ax = axs[0, 1]
    plot_histogram(ax, data_series, 'Histogram', (-3.5, 3), 'density')

    ax = axs[1, 0]
    plot_qq(ax, data_series, 'QQ plot', (-3.5, 3))

    ax = axs[1, 1]
    mean = data_series.sum() / len(data_series)
    acf = [auto_correlation(data_series - mean, i) for i in range(1, 11)]
    variance = 1 / len(data_series)
    plot_correlogram(ax, acf, variance, 'Correlogram')

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig('figures/fig2.7.pdf')

def plot_fig28(ssmodel):
    '''
    '''
    data_series = ssmodel.model_data_df.apply(lambda x: x['u'].ravel()[0] / sqrt(x['D'].ravel()[0]),
                                              axis=1)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('{} - Fig. 2.8'.format(TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_scatter_line(ax, data_series, 'Standardised residuals - epsilon', (-3.5, 3), 'error')

    ax = axs[0, 1]
    plot_histogram(ax, data_series, 'Histogram', (-3.5, 3), 'density')

    data_series = ssmodel.model_data_df.apply(lambda x: x['r'].ravel()[0] / sqrt(x['N'].ravel()[0]),
                                              axis=1)

    ax = axs[1, 0]
    plot_scatter_line(ax, data_series, 'Standardised residuals - eta', (-3.5, 3), 'error')

    ax = axs[1, 1]
    plot_histogram(ax, data_series, 'Histogram', (-3.5, 3), 'density')

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig('figures/fig2.8.pdf')

def run_diagnostics(ssmodel):
    '''
    '''
    forecast_errors = ssmodel.model_data_df.apply(lambda x:
                                                  x['v'].ravel()[0] / sqrt(x['F'].ravel()[0]),
                                                  axis=1)
    n = len(forecast_errors)

    m_1 = forecast_errors.sum() / n
    forecast_errors = forecast_errors - m_1

    m_dict = {i: forecast_errors.apply(lambda x: x ** i).sum() / n
              for i in [2, 3, 4]}
    S = m_dict[3] / sqrt(m_dict[2] ** 3)
    K = m_dict[4] / m_dict[2] ** 2 - 3
    N = n * (S ** 2 / 6 + K ** 2 / 24)

    h = 33
    h_series = forecast_errors.apply(lambda x: x ** 2)
    H = h_series.iloc[:h].sum() / h_series.iloc[h:].sum()

    q = 9
    q_dict = {i: auto_correlation(forecast_errors, i) for i in range(1,q+1)}
    Q = n * (n + 2) * sum([q_dict[i] ** 2 / (n - i) for i in range(1,q+1)])

    print('Diagnostic checks')
    print('=================\n')
    print('S: {:.2f}, K: {:.2f}, N: {:.2f}, H({}): {:.2f}, Q({}): {:.2f}'.format(S, K, N, h,
                                                                                 H, q, Q))
