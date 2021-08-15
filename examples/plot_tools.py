from numpy import sqrt, linspace, array
import pandas as pd
from scipy.stats import gaussian_kde, norm
from numpy.random import normal

XLIM = (1868, 1973)


def get_fig_data(ssmodel, state_col, error_col, confidence=0.9):
    """"""
    percentile = 0.5 + 0.5 * confidence
    quantile = norm.ppf(percentile)
    state_error = ssmodel.model_data_df[error_col].apply(
        lambda x: quantile * sqrt(x.ravel()[0])
    )
    est_state = ssmodel.model_data_df[state_col].apply(lambda x: x.ravel()[0])

    x_vals = [0] + list(ssmodel.y.index) + [2000]
    upper_state = list(est_state + state_error)
    upper_state = [upper_state[0]] + upper_state + [upper_state[-1]]
    lower_state = list(est_state - state_error)
    lower_state = [lower_state[0]] + lower_state + [lower_state[-1]]
    est_state = list(est_state)
    est_state = [est_state[0]] + est_state + [est_state[-1]]

    data_dict = {
        "est_state": est_state,
        "lower_state": lower_state,
        "upper_state": upper_state,
    }
    return pd.DataFrame(data_dict, index=x_vals)


def plot_state(
    ax,
    ssmodel,
    state_data_df,
    legend_text,
    missing_mask=None,
    xlim=XLIM,
    confidence=0.9,
):
    """"""
    if missing_mask is None:
        missing_mask = ssmodel.y.apply(lambda _: True)

    d1 = ax.scatter(
        x=ssmodel.y.index[missing_mask], y=ssmodel.y[missing_mask], marker="x", s=50.0
    )
    (d2,) = ax.plot(state_data_df["est_state"], linewidth=2.0, c="red")
    ax.plot(state_data_df["upper_state"], "--", c="orange")
    ax.plot(state_data_df["lower_state"], "--", c="orange")
    d3 = ax.fill_between(
        state_data_df.index,
        state_data_df["upper_state"],
        state_data_df["lower_state"],
        alpha=0.5,
    )
    ax.set_title("Nile volume data")
    ax.set_xlim(xlim)
    ax.set_ylim(500, 1400)
    ax.set_ylabel("volume")
    ax.legend(
        (d1, d2, d3),
        ("Observed", legend_text, "{:.0f}% conf. int.".format(100 * confidence)),
        loc="upper right",
        fontsize=5,
    )


def plot_line(ax, data_series, title, ylim, ylabel, xlabel=None, xlim=XLIM):
    """"""
    ax.plot(data_series)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_scatter_line(ax, data_series, title, ylim, ylabel, xlabel=None):
    """"""
    ax.scatter(x=data_series.index, y=data_series, marker="x", s=50.0)
    ax.plot(data_series, "--", c="blue")
    ax.set_title(title)
    ax.set_xlim(XLIM)
    ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_line_and_scatter(
    ax, line_data, scatter_data, title, ylim, ylabel, xlabel=None
):
    """"""
    ax.plot(line_data, "-", c="black")
    ax.scatter(x=scatter_data.index, y=scatter_data, marker="x", s=25.0)
    ax.set_title(title)
    ax.set_xlim(XLIM)
    ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_histogram(ax, data_series, title, xlim, ylabel):
    """"""
    ax.hist(data_series, bins=13, density=True)
    ax.set_xlim(xlim)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    density = gaussian_kde(data_series)
    density.covariance_factor = lambda: 0.5
    density._compute_covariance()
    x_vals = linspace(-3.5, 3.0, 200)
    ax.plot(x_vals, density(x_vals), c="red")


def plot_qq(ax, data_series, title, limit):
    """"""
    confidence_data = normal(size=(len(data_series), 10000))
    for i in range(10000):
        confidence_data[:, i] = sorted(confidence_data[:, i])
    for i in range(len(data_series)):
        confidence_data[i, :] = sorted(confidence_data[i, :])
    lower_bound = confidence_data[:, 49]
    upper_bound = confidence_data[:, 9949]

    ordered_data = sorted(data_series)
    percentiles = (1 + array(range(len(data_series)))) / (len(data_series) + 1)
    quantiles = norm.ppf(percentiles)

    d2 = ax.fill_between(quantiles, upper_bound, lower_bound, alpha=0.5, color="orange")
    ax.plot(quantiles, upper_bound, c="red")
    ax.plot(quantiles, lower_bound, c="red")
    d1 = ax.scatter(y=ordered_data, x=quantiles, marker="x", s=25, c="blue")
    ax.plot(limit, limit, c="black")

    ax.set_title(title)
    ax.set_xlim(quantiles[0], quantiles[-1])
    ax.set_ylim(limit)
    ax.set_xlabel("expected")
    ax.set_ylabel("observed")
    ax.legend(
        (d1, d2),
        ("Observed", "{:.0f}% conf. int.".format(99)),
        loc="upper left",
        fontsize=5,
    )


def plot_correlogram(ax, correl_data, variance, title):
    """"""
    xlim = (0, len(correl_data) + 1)
    bound = array([norm.ppf(0.995) * sqrt(variance), norm.ppf(0.995) * sqrt(variance)])

    d2 = ax.fill_between(xlim, bound, -1 * bound, alpha=0.5, color="orange")
    ax.plot(xlim, bound, c="red")
    ax.plot(xlim, -1 * bound, c="red")

    ax.bar(range(1, len(correl_data) + 1), correl_data)
    ax.set_ylim(-1, 1)
    ax.set_xlim(xlim)
    ax.set_title(title)
    ax.legend((d2,), ("{:.0f}% conf. int.".format(99),), loc="upper right", fontsize=5)


def auto_covariance(data_series, lag):
    """"""
    shifted_data = data_series.shift(periods=lag)
    return (data_series * shifted_data).iloc[lag:].sum() / len(data_series)


def auto_correlation(data_series, lag):
    """"""
    n = len(data_series)
    variance = data_series.apply(lambda x: x * x).sum() / n
    return auto_covariance(data_series, lag) / variance
