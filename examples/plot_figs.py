import pandas as pd
import matplotlib.pyplot as plt
from numpy import sqrt, linspace, array, hstack, dot, ravel
from scipy.stats import norm, gaussian_kde
from numpy.random import normal

from sstspack import DynamicLinearGaussianModel as DLGM

NILE_DATA_TITLE = "Volume of Nile river at Aswan 1871-1970"
SEATBELT_DATA_TITLE = "Great Britain Road Accident Casulaties 1969-1984"
BOX_JENKINS_DATA_TITLE = "Box-Jenkins modeling of internet user data"
MOTORCYCLE_DATA_TITLE = "Simulated Motorcycle Acceleration Data"
XLIM = (1868, 1973)
XLABEL = "year"
FIG_LAYOUT = [0, 0.03, 1, 0.95]


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
    result = pd.DataFrame(data_dict, index=x_vals)
    return result


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
    (d2,) = ax.plot(state_data_df["est_state"], "r", linewidth=2.0, c="red")
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
    result = (data_series * shifted_data).iloc[lag:].sum() / len(data_series)
    return result


def auto_correlation(data_series, lag):
    """"""
    n = len(data_series)
    variance = data_series.apply(lambda x: x * x).sum() / n
    result = auto_covariance(data_series, lag) / variance
    return result


def plot_fig21(ssmodel):
    """"""
    state_data_df = get_fig_data(ssmodel, "a_prior", "P_prior")

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.1".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_state(ax, ssmodel, state_data_df, "Filtered mean")

    ax = axs[0, 1]
    plot_line(ax, ssmodel.P_prior, "P_prior (P_t)", (5000, 17500), "variance")

    ax = axs[1, 0]
    data_series = ssmodel.y - ssmodel.a_prior
    plot_scatter_line(ax, data_series, "Forecast error", (-450, 450), "error")

    ax = axs[1, 1]
    plot_line(ax, ssmodel.F, "F (F_t)", (20000, 32500), "variance", XLABEL)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.1.pdf")


def plot_fig22(ssmodel):
    """"""
    state_data_df = get_fig_data(ssmodel, "a_hat", "V")

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.2".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_state(ax, ssmodel, state_data_df, "Smoothed mean")

    ax = axs[0, 1]
    plot_line(ax, ssmodel.V, "V (V_t)", (2200, 4100), "variance")

    ax = axs[1, 0]
    plot_scatter_line(
        ax, ssmodel.r, "Smoothing cumulant r (r_t)", (-0.04, 0.024), "error", XLABEL
    )

    ax = axs[1, 1]
    plot_line(ax, ssmodel.N, "N (N_t)", (0.000048, 0.00011), "variance", XLABEL)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.2.pdf")


def plot_fig23(ssmodel):
    """"""
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.3".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_scatter_line(ax, ssmodel.epsilon_hat, "epsilon_hat", (-380, 300), "error")

    ax = axs[0, 1]
    plot_line(ax, ssmodel.epsilon_hat_sigma2, "epsilon_hat_sigma2", None, "variance")

    ax = axs[1, 0]
    plot_scatter_line(ax, ssmodel.eta_hat, "eta_hat", (-55, 35), "error", XLABEL)

    ax = axs[1, 1]
    plot_line(ax, ssmodel.eta_hat_sigma2, "eta_hat_sigma2", None, "variance", XLABEL)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.3.pdf")


def plot_fig24(ssmodel, sim):
    """"""
    initial_key = sim.index[0]
    a0 = ssmodel.a_hat[initial_key]
    P0 = ssmodel.V[initial_key]
    model_fields = ["Z", "d", "H", "T", "c", "R", "Q"]
    model_df = ssmodel.model_data_df[model_fields].copy()
    new_sim = DLGM.simulate_model(model_df, a0, P0)
    plot_ssm = DLGM(new_sim.y, model_df, a0, P0)
    plot_ssm.smoother()
    plot_sim = sim.alpha - ssmodel.a_hat + plot_ssm.a_hat

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.4".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_line_and_scatter(
        ax, ssmodel.a_hat, plot_sim, "a_hat simulation", None, "volume"
    )

    ax = axs[0, 1]
    plot_line_and_scatter(
        ax, ssmodel.a_hat, sim.alpha, "a_hat conditional simulation", None, "volume"
    )

    ax = axs[1, 0]
    plot_line_and_scatter(
        ax,
        ssmodel.epsilon_hat,
        sim.epsilon,
        "epsilon_hat conditional simulation",
        None,
        "error",
        XLABEL,
    )

    ax = axs[1, 1]
    plot_line_and_scatter(
        ax,
        ssmodel.eta_hat[:-1],
        sim.eta[:-1],
        "eta_hat conditional simulation",
        None,
        "error",
        XLABEL,
    )

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.4.pdf")


def plot_fig25(ssmodel):
    """"""
    missing_mask = ssmodel.y.apply(lambda x: not DLGM.is_all_missing(x))
    state_data_df = get_fig_data(ssmodel, "a_prior", "P_prior")

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.5".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_state(ax, ssmodel, state_data_df, "Filtered mean", missing_mask)

    ax = axs[0, 1]
    plot_line(ax, ssmodel.P_prior, "P_prior (P_t)", (5000, 36000), "variance")

    state_data_df = get_fig_data(ssmodel, "a_hat", "V")

    ax = axs[1, 0]
    plot_state(ax, ssmodel, state_data_df, "Smoothed mean", missing_mask)

    ax = axs[1, 1]
    plot_line(ax, ssmodel.V, "V (V_t)", (2200, 10000), "variance", XLABEL)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.5.pdf")


def plot_fig26(ssmodel):
    """"""
    missing_mask = ssmodel.y.apply(lambda x: not DLGM.is_all_missing(x))
    state_data_df = get_fig_data(ssmodel, "a_prior", "P_prior", confidence=0.5)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.6".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_state(
        ax,
        ssmodel,
        state_data_df,
        "Filtered mean",
        missing_mask,
        xlim=(1868, 2003),
        confidence=0.5,
    )

    ax = axs[0, 1]
    plot_line(
        ax,
        ssmodel.P_prior,
        "P_prior (P_t)",
        (5000, 50000),
        "variance",
        xlim=(1868, 2003),
    )

    ax = axs[1, 0]
    plot_line(
        ax,
        ssmodel.a_prior,
        "a_prior (a_t)",
        (700, 1250),
        "volume",
        XLABEL,
        xlim=(1868, 2003),
    )

    ax = axs[1, 1]
    plot_line(
        ax, ssmodel.F, "F (F_t)", (20000, 65000), "variance", XLABEL, xlim=(1868, 2003)
    )

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.6.pdf")


def plot_fig27(ssmodel):
    """"""
    data_series = ssmodel.model_data_df.apply(
        lambda x: x["v"].ravel()[0] / sqrt(x["F"].ravel()[0]), axis=1
    )

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.7".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_scatter_line(
        ax, data_series, "Standardised residuals - epsilon", (-3.5, 3), "error"
    )

    ax = axs[0, 1]
    plot_histogram(ax, data_series, "Histogram", (-3.5, 3), "density")

    ax = axs[1, 0]
    plot_qq(ax, data_series, "QQ plot", (-3.5, 3))

    ax = axs[1, 1]
    mean = data_series.sum() / len(data_series)
    acf = [auto_correlation(data_series - mean, i) for i in range(1, 11)]
    variance = 1 / len(data_series)
    plot_correlogram(ax, acf, variance, "Correlogram")

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.7.pdf")


def plot_fig28(ssmodel):
    """"""
    data_series = ssmodel.model_data_df.apply(
        lambda x: x["u"].ravel()[0] / sqrt(x["D"].ravel()[0]), axis=1
    )

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 2.8".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    plot_scatter_line(
        ax, data_series, "Standardised residuals - epsilon", (-3.5, 3), "error"
    )

    ax = axs[0, 1]
    plot_histogram(ax, data_series, "Histogram", (-3.5, 3), "density")

    data_series = ssmodel.model_data_df.apply(
        lambda x: x["r"].ravel()[0] / sqrt(x["N"].ravel()[0]), axis=1
    )

    ax = axs[1, 0]
    plot_scatter_line(
        ax, data_series, "Standardised residuals - eta", (-3.5, 3), "error"
    )

    ax = axs[1, 1]
    plot_histogram(ax, data_series, "Histogram", (-3.5, 3), "density")

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig2.8.pdf")


def run_diagnostics(ssmodel):
    """"""
    forecast_errors = ssmodel.model_data_df.apply(
        lambda x: x["v"].ravel()[0] / sqrt(x["F"].ravel()[0]), axis=1
    )
    n = len(forecast_errors)

    m_1 = forecast_errors.sum() / n
    forecast_errors = forecast_errors - m_1

    m_dict = {i: forecast_errors.apply(lambda x: x ** i).sum() / n for i in [2, 3, 4]}
    S = m_dict[3] / sqrt(m_dict[2] ** 3)
    K = m_dict[4] / m_dict[2] ** 2 - 3
    N = n * (S ** 2 / 6 + K ** 2 / 24)

    h = 33
    h_series = forecast_errors.apply(lambda x: x ** 2)
    H = h_series.iloc[:h].sum() / h_series.iloc[h:].sum()

    q = 9
    q_dict = {i: auto_correlation(forecast_errors, i) for i in range(1, q + 1)}
    Q = n * (n + 2) * sum([q_dict[i] ** 2 / (n - i) for i in range(1, q + 1)])

    print("Diagnostic checks")
    print("=================\n")
    print(
        "S: {:.2f}, K: {:.2f}, N: {:.2f}, H({}): {:.2f}, Q({}): {:.2f}".format(
            S, K, N, h, H, q, Q
        )
    )


def plot_fig81(seatbelt_df):
    """"""
    fig, ax = plt.subplots()
    fig.suptitle("{} - Fig. 8.1".format(SEATBELT_DATA_TITLE), fontsize=14)

    ax.plot(seatbelt_df.index, seatbelt_df.iloc[:, 0], "-bx")
    ax.set_ylabel("Log KSI")
    ax.set_xlabel("Year")

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.1.pdf")


def plot_fig82(model):
    """"""
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("{} - Fig. 8.2".format(SEATBELT_DATA_TITLE), fontsize=14)

    plot_seasonal_breakdown(axs, model, [0], range(1, 12))

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.2.pdf")


def plot_seasonal_breakdown(axs, model, level_indexes, seasonal_indexes):
    """"""
    level_data = array(
        [
            dot(model.Z[idx][:, level_indexes], model.a_hat[idx][level_indexes, :])
            for idx in model.index
        ]
    ).T
    seasonal_data = array(
        [
            dot(
                model.Z[idx][:, seasonal_indexes], model.a_hat[idx][seasonal_indexes, :]
            )
            for idx in model.index
        ]
    ).T
    ax = axs[0]
    ax.plot(model.index, model.y, "g-.")
    ax.plot(model.index, ravel(level_data), "r-")
    ax.set_title("Level")

    ax = axs[1]
    ax.plot(model.index, ravel(seasonal_data))
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.set_title("Seasonality")

    ax = axs[2]
    ax.plot(model.index, model.epsilon_hat)
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.set_title("Irregular")


def plot_fig83(model):
    a_hat = hstack(model.a_hat)
    a_prior = hstack(model.a_prior)
    fig, ax = plt.subplots(1)
    fig.suptitle("{} - Fig. 8.3".format(SEATBELT_DATA_TITLE), fontsize=14)

    ax.scatter(x=model.index, y=model.y, marker=".", label="Data")
    ax.plot(model.index, a_hat[0, :], "r--", label="Smoothed")
    ax.plot(model.index[12:], a_prior[0, 12:], "g--", label="Predicted")
    ax.set_ylim((6.9, 7.9))
    ax.legend()

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.3.pdf")


def plot_lines(data_series, ax):
    """"""
    for idx in data_series.index:
        try:
            value = data_series[idx][0, 0]
        except:
            value = data_series[idx]
        ax.plot((idx, idx), (0, value), c="k")


def plot_fig84(model, confidence=0.9):
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("{} - Fig. 8.4".format(SEATBELT_DATA_TITLE), fontsize=14)

    percentile = 0.5 + 0.5 * confidence
    quantile = norm.ppf(percentile)
    nondiffuse_index = model.index[12:]
    prediction_residuals = model.v[nondiffuse_index]
    residuals_std = ravel(sqrt(model.F[nondiffuse_index].to_list()))
    standardise_residuals = prediction_residuals / residuals_std
    auxillary_residuals_u = model.u[nondiffuse_index] / ravel(
        sqrt(model.D[nondiffuse_index].to_list())
    )
    level_idx = [idx for idx in model.index if model.r[idx] is not pd.NA]
    level_data = [model.r[idx][0, 0] for idx in level_idx]
    level_series = pd.Series(level_data, index=level_idx)
    level_std = array([sqrt(model.N[idx][0, 0]) for idx in level_idx])
    aux_level_resid = level_series / level_std

    ax = axs[0]
    ax.plot(standardise_residuals.index, standardise_residuals)
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.axhline(y=quantile, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.axhline(y=-quantile, color="k", ls="--", alpha=0.75, lw=0.5)

    ax = axs[1]
    plot_lines(auxillary_residuals_u, ax)
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.axhline(y=quantile, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.axhline(y=-quantile, color="k", ls="--", alpha=0.75, lw=0.5)

    ax = axs[2]
    plot_lines(aux_level_resid, ax)
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.axhline(y=quantile, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.axhline(y=-quantile, color="k", ls="--", alpha=0.75, lw=0.5)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.4.pdf")


def plot_fig85(model):
    """"""
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("{} - Fig. 8.5".format(SEATBELT_DATA_TITLE), fontsize=14)

    plot_seasonal_breakdown(axs, model, [0, 12, 13], range(1, 12))

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.5.pdf")


def plot_fig86(data_df):
    """"""
    fig, ax = plt.subplots(1)
    fig.suptitle("{} - Fig. 8.6".format(SEATBELT_DATA_TITLE), fontsize=14)

    ax.plot(data_df.index, data_df.Front, "-.r", label="Front Seat")
    ax.plot(data_df.index, data_df.Rear, "--xg", label="Rear Seat")
    ax.legend()
    ax.set_xlabel("Year")
    ax.set_ylabel("Log KSI")

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.6.pdf")


def plot_fig87(model):
    """"""
    fig, axs = plt.subplots(2, 1)
    fig.suptitle("{} - Fig. 8.7".format(SEATBELT_DATA_TITLE), fontsize=14)

    confidence = 0.95
    percentile = 0.5 + 0.5 * confidence
    quantile = norm.ppf(percentile)

    data = hstack([model.y[idx] for idx in model.index])
    seasonal = [False] * 2 + [True] * 22 + [False] * 6
    front_level = []
    front_sd = []
    rear_level = []
    rear_sd = []
    for idx in model.index:
        Z = model.Z[idx].copy()
        Z[:, seasonal] = 0
        var_mat = dot(dot(Z, model.V[idx]), Z.T)
        front_level.append(dot(Z[0, :], model.a_hat[idx]))
        front_sd.append(sqrt(var_mat[0, 0]))
        rear_level.append(dot(Z[1, :], model.a_hat[idx]))
        rear_sd.append(sqrt(var_mat[1, 1]))

    front_level = ravel(array(front_level))
    front_sd = ravel(array(front_sd))
    front_upper = ravel(front_level + quantile * front_sd)
    front_lower = ravel(front_level - quantile * front_sd)

    rear_level = ravel(array(rear_level))
    rear_sd = ravel(array(rear_sd))
    rear_upper = ravel(rear_level + quantile * rear_sd)
    rear_lower = ravel(rear_level - quantile * rear_sd)

    ax = axs[0]
    ax.plot(model.index, data[0, :], "--xb", label="Front Seat")
    ax.plot(model.index, ravel(front_level), "-r", label="Level")
    ax.fill_between(model.index, front_upper, front_lower, alpha=0.5, label="95% Conf.")
    ax.legend()

    ax = axs[1]
    ax.plot(model.index, data[1, :], "--xb", label="Rear Seat")
    ax.plot(model.index, ravel(rear_level), "-r", label="Level")
    ax.fill_between(model.index, rear_upper, rear_lower, alpha=0.5, label="95% Conf.")
    ax.legend()

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.7.pdf")


def plot_with_missing_data(data, ax):
    """"""
    start_idx = None
    curr_idx = 1
    data_idx_list = []
    while curr_idx <= len(data):
        if start_idx is None and not data[curr_idx] is pd.NA:
            start_idx = curr_idx
        if data[curr_idx] is pd.NA and start_idx is not None:
            data_idx_list.append(list(range(start_idx, curr_idx)))
            start_idx = None
        curr_idx += 1

    for plot_range in data_idx_list:
        ax.plot(plot_range, data[plot_range], "-b")


def plot_fig88(data, missing_data):
    """"""
    fig, axs = plt.subplots(2, 1)
    fig.suptitle("{} - Fig. 8.8".format(BOX_JENKINS_DATA_TITLE), fontsize=14)

    ax = axs[0]
    ax.plot(data)
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.set_ylabel("Change")

    ax = axs[1]
    plot_with_missing_data(missing_data, ax)
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.set_ylabel("Change")
    ax.set_xlabel("Minutes")

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.8.pdf")


def plot_fig89(model):
    """"""
    fig, ax = plt.subplots(1)
    ax.scatter(
        model.index[:99], model.y[model.index[:99]], label="Change in internet users"
    )
    forecast_data = pd.Series(
        [dot(model.Z[idx], model.a_prior[idx])[0, 0] for idx in model.index],
        index=model.index,
    )

    confidence = 0.5
    percentile = 0.5 + 0.5 * confidence
    quantile = norm.ppf(percentile)
    forecast_error = pd.Series(
        [quantile * sqrt(x[0, 0]) for x in model.F], index=model.index
    )

    ax.plot(
        forecast_data.index[(model.d_diffuse + 1) : 100],
        forecast_data[(model.d_diffuse + 1) : 100],
        "k",
        label="One step ahead forecast",
    )
    ax.plot(forecast_data.index[100:], forecast_data[100:], "--k")

    ax.plot(
        forecast_error.index[99:],
        forecast_error[forecast_error.index[99:]],
        "g",
        label="50% Confidence interval",
    )
    ax.plot(forecast_error.index[99:], -forecast_error[forecast_error.index[99:]], "g")

    ax.legend()
    ax.set_title("{} ARMA(1, 1) - Fig. 8.9".format(BOX_JENKINS_DATA_TITLE))
    ax.set_ylabel("Change")
    ax.set_xlabel("Minutes")

    fig.savefig("figures/fig8.9.pdf")


def plot_fig810(model):
    """"""
    fig, axs = plt.subplots(2, 1)
    fig.suptitle("{} - Fig. 8.10".format(MOTORCYCLE_DATA_TITLE), fontsize=14)

    ax = axs[0]
    data_index = []
    data_full = []
    epsilon_full = []
    for idx in model.index:
        for i in range(model.y[idx].shape[0]):
            data_index.append(idx)
            data_full.append(model.y[idx][i, 0])
            epsilon_full.append(model.epsilon_hat[idx][i, 0] / sqrt(model.H[idx][i, i]))
    a_hat = array([model.a_hat[idx][0, 0] for idx in model.index])
    confidence = 0.95
    percentile = 0.5 + 0.5 * confidence
    quantile = norm.ppf(percentile)
    V_sqrt95 = quantile * array([sqrt(model.V[idx][0, 0]) for idx in model.index])

    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.scatter(data_index, data_full, s=5, c="b", marker="x", label="Simulated data")
    ax.plot(model.index, a_hat, "g-", label="Spline model")
    ax.plot(model.index, a_hat + V_sqrt95, "r--", label="95% Confidence")
    ax.plot(model.index, a_hat - V_sqrt95, "r--")
    ax.legend()
    ax.set_ylabel("Accelaration")
    ax.set_xlabel("Time")

    ax = axs[1]
    ax.axhline(y=0, color="k", ls="--", alpha=0.75, lw=0.5)
    ax.scatter(data_index, epsilon_full, s=5, c="b")
    ax.set_ylabel("Standardise residual")
    ax.set_xlabel("Time")

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig8.10.pdf")


def figs12(axs, particle_model, gaussian_model):
    """"""
    ax = axs[0, 0]
    ax.plot(
        particle_model.index, particle_model.a_posterior, "b", label="Particle Filter"
    )
    ax.scatter(particle_model.index, particle_model.y, label="Data")
    confidence90 = norm.ppf(0.95) * array([sqrt(x) for x in particle_model.P_posterior])
    ax.plot(
        particle_model.index,
        particle_model.a_posterior + confidence90,
        "k--",
        label="90% conf.",
    )
    ax.plot(particle_model.index, particle_model.a_posterior - confidence90, "k--")
    ax.legend()
    ax.set_ylim((400, 1400))

    ax = axs[0, 1]
    ax.plot(
        particle_model.index, particle_model.a_posterior, "--", label="Particle Filter"
    )
    ax.plot(gaussian_model.index, gaussian_model.a_posterior, label="Kalman Filter")
    ax.legend()
    ax.set_ylim((700, 1200))

    ax = axs[1, 0]
    ax.plot(
        particle_model.index, particle_model.P_posterior, "--", label="Particle Filter"
    )
    ax.plot(gaussian_model.index, gaussian_model.P_posterior, label="Kalman Filter")
    ax.legend()
    ax.set_ylim((0, 15000))

    ax = axs[1, 1]
    ax.plot(particle_model.index, particle_model.ESS)
    ax.set_ylim((0, 10000))


def plot_fig121(particle_model, gaussian_model):
    """"""
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 12.1".format(NILE_DATA_TITLE), fontsize=14)

    figs12(axs, particle_model, gaussian_model)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig12.1.pdf")


def plot_fig122(particle_model, gaussian_model):
    """"""
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 12.2".format(NILE_DATA_TITLE), fontsize=14)

    figs12(axs, particle_model, gaussian_model)

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig12.2.pdf")


def plot_fig123(particle_model, auxiliary_model):
    """"""
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("{} - Fig. 12.3".format(NILE_DATA_TITLE), fontsize=14)

    ax = axs[0, 0]
    ax.plot(
        particle_model.index, particle_model.a_posterior, "b", label="Particle Filter"
    )
    ax.scatter(particle_model.index, particle_model.y, label="Data")
    confidence90 = norm.ppf(0.95) * array([sqrt(x) for x in particle_model.P_posterior])
    ax.plot(
        particle_model.index,
        particle_model.a_posterior + confidence90,
        "k--",
        label="90% conf.",
    )
    ax.plot(particle_model.index, particle_model.a_posterior - confidence90, "k--")
    ax.legend()
    ax.set_ylim((400, 1400))

    ax = axs[0, 1]
    ax.plot(particle_model.index, particle_model.ESS)
    ax.set_ylim((0, 10000))

    ax = axs[1, 0]
    ax.plot(
        auxiliary_model.index, auxiliary_model.a_posterior, "b", label="Particle Filter"
    )
    ax.scatter(auxiliary_model.index, auxiliary_model.y, label="Data")
    confidence90 = norm.ppf(0.95) * array(
        [sqrt(x) for x in auxiliary_model.P_posterior]
    )
    ax.plot(
        auxiliary_model.index,
        auxiliary_model.a_posterior + confidence90,
        "k--",
        label="90% conf.",
    )
    ax.plot(auxiliary_model.index, auxiliary_model.a_posterior - confidence90, "k--")
    ax.legend()
    ax.set_ylim((400, 1400))

    ax = axs[1, 1]
    ax.plot(auxiliary_model.index, auxiliary_model.ESS)
    ax.set_ylim((0, 10000))

    fig.tight_layout(rect=FIG_LAYOUT)
    fig.savefig("figures/fig12.3.pdf")
