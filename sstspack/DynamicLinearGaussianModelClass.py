from numpy import (
    dot,
    zeros,
    ravel,
    log,
    pi as PI,
    identity,
    array,
    isnan,
    inf,
    sum,
    abs,
    sqrt,
    full,
    ndarray,
)
from numpy.linalg import inv, det, LinAlgError
from numpy.random import multivariate_normal as mv_norm
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

EPSILON = 1e-14


class DynamicLinearGaussianModel(object):
    """
    The DynamicLinearGaussianModel object based on the user provided parameters.

    Parameters
    ----------
    model_parameters_df : pandas.DataFrame
    """

    expected_columns = ("Z", "d", "H", "T", "c", "R", "Q")
    estimation_columns = [
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

    def __init__(
        self,
        y_series,
        model_design_df,
        a_prior_initial,
        P_prior_initial,
        diffuse_states=None,
        validate_input=True,
    ):
        """
        Constructor
        """
        self._m = None
        self.model_data_df = model_design_df.copy()

        self._add_estimation_columns()
        self._initialise_model_data(a_prior_initial)

        if validate_input:
            y_series = self._validate_input_variables(y_series)

        self.model_data_df.insert(0, "y", y_series)

        self._set_up_initial_terms(a_prior_initial, P_prior_initial, diffuse_states)

        self.filter_run = False
        self.smoother_run = False
        self.disturbance_smoother_run = False

    def _initialise_model_data(self, a_prior_initial):
        """"""
        pass

    def _validate_input_variables(self, y_series):
        """"""
        y_series = y_series.astype(object)
        # Verify y_series.index is a subset of self.index
        assert all(
            idx in self.index for idx in y_series.index
        ), f"Elements of y_series.index are not in model_design.index, elements missing: (y_series.index[y_series.index not in self.index])"

        self._check_model_data_columns()

        # Verify all data in matrix form
        p = self.p
        for idx in y_series.index:
            try:
                y_series[idx] = full((1, 1), y_series[idx])
            except ValueError:
                pass

            assert y_series[idx].shape == (p[idx], 1)

        for idx in self.index:
            verification_columns = self._verification_columns(p, idx)
            for col in verification_columns:
                try:
                    _ = self.model_data_df.loc[idx, col].shape
                except AttributeError:
                    self.model_data_df.loc[idx, col] = full(
                        (1, 1), self.model_data_df.loc[idx, col]
                    )
                assert (
                    self.model_data_df.loc[idx, col].shape == verification_columns[col],
                    f"model_data_df.{col}[{idx}].shape has dimension {self.model_data_df.loc[idx, col].shape}, expected: {verification_columns[col]}",
                )

        return y_series

    def _check_model_data_columns(self):
        """"""
        assert all(col in self.model_data_df.columns for col in self.expected_columns)

    def _add_estimation_columns(self):
        """"""
        self.model_data_df = self.model_data_df.reindex(
            columns=self.model_data_df.columns.tolist() + self.estimation_columns
        )
        self.model_data_df[self.estimation_columns] = pd.NA

    def _verification_columns(self, p, idx):
        """"""
        return {
            "Z": (p[idx], self.m),
            "d": (p[idx], 1),
            "H": (p[idx], p[idx]),
            "T": (self.m, self.m),
            "c": (self.m, 1),
            "R": (self.m, self.r_eta),
            "Q": (self.r_eta, self.r_eta),
        }

    @property
    def index(self):
        """"""
        return self.model_data_df.index

    @property
    def non_diffuse_index(self):
        """"""
        start_idx = self.d_diffuse + 1
        return self.model_data_df.index[start_idx:]

    def aggregate_field(self, field, mask=None):
        """"""
        data = []

        for idx in self.index:
            Z = mask
            if mask is None:
                Z = self.Z[idx]
            value = dot(Z, self.model_data_df.loc[idx, field])
            if value.shape == (1, 1):
                value = value[0, 0]
            data.append(value)

        result = pd.Series(data, index=self.index)
        return result

    def quadratic_aggregate_field(self, field, mask=None):
        """"""
        data = []

        for idx in self.index:
            Z = mask
            if mask is None:
                Z = self.Z[idx]
            value = dot(dot(Z, self.model_data_df.loc[idx, field]), Z.T)
            if value.shape == (1, 1):
                value = value[0, 0]
            data.append(value)

        result = pd.Series(data, index=self.index)
        return result

    @property
    def a_prior_aggregate(self):
        """"""
        return self.aggregate_field("a_prior")[self.non_diffuse_index]

    @property
    def a_posterior_aggregate(self):
        """"""
        return self.aggregate_field("a_posterior")[self.non_diffuse_index]

    @property
    def a_hat_aggregate(self):
        """"""
        return self.aggregate_field("a_hat")

    @property
    def P_prior_aggregate(self):
        """"""
        return self.quadratic_aggregate_field("P_prior")[self.non_diffuse_index]

    @property
    def P_posterior_aggregate(self):
        """"""
        return self.quadratic_aggregate_field("P_posterior")[self.non_diffuse_index]

    @property
    def V_aggregate(self):
        """"""
        return self.quadratic_aggregate_field("V")

    @property
    def P_prior_aggregate_sqrt(self):
        """"""
        result = self.quadratic_aggregate_field("P_prior")[self.non_diffuse_index]
        for idx in result.index:
            result[idx] = sqrt(result[idx])
        return result

    @property
    def P_posterior_aggregate_sqrt(self):
        """"""
        result = self.quadratic_aggregate_field("P_posterior")[self.non_diffuse_index]
        for idx in result.index:
            result[idx] = sqrt(result[idx])
        return result

    @property
    def V_aggregate_sqrt(self):
        """"""
        result = self.quadratic_aggregate_field("V")
        for idx in result.index:
            result[idx] = sqrt(result[idx])
        return result

    @property
    def initial_index(self):
        """"""
        return self.index[0]

    @property
    def n(self):
        """"""
        return len(self.model_data_df)

    @property
    def m(self):
        """"""
        if self._m is None:
            return self.Z[self.initial_index].shape[1]

        return self._m

    @property
    def p(self):
        """ """
        return pd.Series(
            data=[self.Z[key].shape[0] for key in self.index],
            index=self.model_data_df.index,
        )

    @property
    def r_eta(self):
        """"""
        return self.R[self.initial_index].shape[1]

    def _set_up_initial_terms(self, a_prior_initial, P_prior_initial, diffuse_states):
        """"""
        assert (a_prior_initial is not None and P_prior_initial is not None) or (
            diffuse_states is not None and all(diffuse_states)
        )

        if diffuse_states is None or not any(diffuse_states):
            self.d_diffuse = -1

            self.a_prior[self.initial_index] = a_prior_initial
            self.P_prior[self.initial_index] = P_prior_initial
        else:
            self.d_diffuse = self.n

            a_prior_initial[diffuse_states, 0] = 0
            self.a_prior[self.initial_index] = a_prior_initial

            I = identity(self.m)
            A = I[:, diffuse_states]
            self.P_infinity_prior[self.initial_index] = dot(A, A.T)

            proper_states = [not x for x in diffuse_states]
            R0 = I[:, proper_states]
            Q0 = P_prior_initial[proper_states, :]
            Q0 = Q0[:, proper_states]
            self.P_star_prior[self.initial_index] = dot(dot(R0, Q0), R0.T)

            self.P_prior[self.initial_index] = self.diffuse_P(
                self.P_star_prior[self.initial_index],
                self.P_infinity_prior[self.initial_index],
            )

    def adapt_row_to_any_missing_data(self, row):
        """"""
        v_shape = (self.p[row], 1)
        if self.is_partial_missing(self.y[row]):
            self.y_original[row] = self.y[row].copy()
            self.Z_original[row] = self.Z[row].copy()
            self.d_original[row] = self.d[row].copy()
            self.H_original[row] = self.H[row].copy()

        if self.is_all_missing(self.y[row]):
            self.Z[row] = zeros(self.Z[row].shape)
            self.v[row] = zeros(v_shape)
            self.p[row] = 0
        else:
            if self.is_partial_missing(self.y[row]):
                W = self.remove_missing_rows(identity(self.p[row]), self.y[row])
                self.y[row] = self.remove_missing_rows(self.y[row], self.y[row])
                self.Z[row] = dot(W, self.Z[row])
                self.d[row] = dot(W, self.d[row])
                self.H[row] = dot(dot(W, self.H[row]), W.T)

    def filter(self):
        """
        Perform the Kalman filter with the y data series and the model design
        """
        for index, key in enumerate(self.index):
            self._initialise_parameters(key)
            self._non_missing_F(key)
            self.adapt_row_to_any_missing_data(key)

            # TODO: Deal with multivariate data
            if index <= self.d_diffuse:
                (
                    a_prior_next,
                    P_prior_next,
                    P_infinity_prior,
                    P_star_prior,
                ) = self._diffuse_filter_recursion_step(key, index)
            else:
                a_prior_next, P_prior_next = self._filter_recursion_step(key)

            nxt_idx = index + 1
            try:
                nxt_key = self.index[nxt_idx]
            except IndexError:
                self.a_prior_final = a_prior_next
                self.P_prior_final = P_prior_next
                if index < self.d_diffuse:
                    self.P_infinity_prior_final = P_infinity_prior
                    self.P_star_prior_final = P_star_prior
                    # TODO: Warn user distribution is still diffuse
            else:
                self.a_prior[nxt_key] = a_prior_next
                self.P_prior[nxt_key] = P_prior_next
                if index < self.d_diffuse:
                    self.P_infinity_prior[nxt_key] = P_infinity_prior
                    self.P_star_prior[nxt_key] = P_star_prior

        self.filter_run = True

    def _prediction_error(self, key):
        """"""
        return self.y[key] - dot(self.Z[key], self.a_prior[key]) - self.d[key]

    def _initialise_parameters(self, key):
        """"""
        pass

    def _non_missing_F(self, key):
        """"""
        PZ = dot(self.P_prior[key], self.Z[key].T)
        self.F[key] = dot(self.Z[key], PZ) + self.H[key]

    def _diffuse_filter_recursion_step(self, key, index):
        """"""
        self.v[key] = self._prediction_error(key)
        self.F_infinity[key] = dot(
            dot(self.Z[key], self.P_infinity_prior[key]), self.Z[key].T
        )
        self.F_star[key] = (
            dot(dot(self.Z[key], self.P_star_prior[key]), self.Z[key].T) + self.H[key]
        )

        self.M_infinity[key] = dot(self.P_infinity_prior[key], self.Z[key].T)
        self.M_star[key] = dot(self.P_star_prior[key], self.Z[key].T)

        try:
            self.F1[key] = inv(self.F_infinity[key])
        except LinAlgError:
            F = self.F_star[key].copy()
            self.F_inverse[key] = inv(F)

            K0_hat = dot(self.M_star[key], self.F_inverse[key])
            K1_hat = zeros((self.m, self.Z[key].shape[0]))
        else:
            self.F2[key] = -1 * dot(dot(self.F1[key], self.F_star[key]), self.F1[key])

            K0_hat = dot(self.M_infinity[key], self.F1[key])
            K1_hat = dot(self.M_star[key], self.F1[key]) + dot(
                self.M_infinity[key], self.F2[key]
            )
        self.K0[key] = dot(self.T[key], K0_hat)
        self.K1[key] = dot(self.T[key], K1_hat)

        L0_hat = identity(self.m) - dot(K0_hat, self.Z[key])
        L1_hat = -1 * dot(K1_hat, self.Z[key])
        self.L0[key] = dot(self.T[key], L0_hat)
        self.L1[key] = dot(self.T[key], L1_hat)

        self.a_posterior[key] = self.a_prior[key] + dot(K0_hat, self.v[key])
        self.P_infinity_posterior[key] = dot(self.P_infinity_prior[key], L0_hat.T)
        self.P_star_posterior[key] = dot(self.P_infinity_prior[key], L1_hat.T) + dot(
            self.P_star_prior[key], L0_hat.T
        )
        self.P_posterior[key] = self.diffuse_P(
            self.P_star_posterior[key], self.P_infinity_posterior[key]
        )

        if all(abs(ravel(self.P_infinity_posterior[key])) <= EPSILON):
            self.d_diffuse = index

        RQR = dot(dot(self.R[key], self.Q[key]), self.R[key].T)
        a_prior_next = dot(self.T[key], self.a_posterior[key]) + self.c[key]
        P_prior_next = dot(dot(self.T[key], self.P_posterior[key]), self.T[key].T) + RQR

        P_infinity_prior = dot(
            dot(self.T[key], self.P_infinity_posterior[key]), self.T[key].T
        )
        P_star_prior = (
            dot(dot(self.T[key], self.P_star_posterior[key]), self.T[key].T) + RQR
        )
        P_prior_next = self.diffuse_P(P_star_prior, P_infinity_prior)

        return a_prior_next, P_prior_next, P_infinity_prior, P_star_prior

    def _filter_recursion_step(self, key):
        """"""
        self.v[key] = self._prediction_error(key)
        PZ = dot(self.P_prior[key], self.Z[key].T)
        F = dot(self.Z[key], PZ) + self.H[key]
        try:
            self.F_inverse[key] = inv(F)
        except LinAlgError:
            self.F_inverse[key] = 0
        PZF_inv = dot(PZ, self.F_inverse[key])

        self.a_posterior[key] = self.a_prior[key] + dot(PZF_inv, self.v[key])
        self.P_posterior[key] = self.P_prior[key] - dot(PZF_inv, PZ.T)

        RQR = dot(dot(self.R[key], self.Q[key]), self.R[key].T)
        a_prior_next = dot(self.T[key], self.a_posterior[key]) + self.c[key]
        P_prior_next = dot(dot(self.T[key], self.P_posterior[key]), self.T[key].T) + RQR

        return a_prior_next, P_prior_next

    def smoother(self):
        """"""
        if not self.filter_run:
            self.filter()

        self.r_final = zeros((self.m, 1))
        self.N_final = zeros((self.m, self.m))
        self.r0_final = None
        self.r1_final = None
        self.N0_final = None
        self.N1_final = None
        self.N2_final = None
        if self.d_diffuse + 1 >= self.n:
            self.r0_final = self.r_final.copy()
            self.r1_final = self.r_final.copy()
            self.N0_final = self.N_final.copy()
            self.N1_final = self.N_final.copy()
            self.N2_final = self.N_final.copy()

        for index, key in reversed(list(enumerate(self.index))):
            ZF_inv = dot(self.Z[key].T, self.F_inverse[key])
            self.K[key] = dot(dot(self.T[key], self.P_prior[key]), ZF_inv)
            self.L[key] = self.T[key] - dot(self.K[key], self.Z[key])

            next_index = index + 1
            try:
                next_key = self.index[next_index]
            except IndexError:
                next_r = self.r_final
                next_N = self.N_final
                if index <= self.d_diffuse:
                    next_r0 = self.r0_final
                    next_r1 = self.r1_final
                    next_N0 = self.N0_final
                    next_N1 = self.N1_final
                    next_N2 = self.N2_final
            else:
                next_r = self.r[next_key]
                next_N = self.N[next_key]
                if index <= self.d_diffuse:
                    next_r0 = self.r0[next_key]
                    next_r1 = self.r1[next_key]
                    next_N0 = self.N0[next_key]
                    next_N1 = self.N1[next_key]
                    next_N2 = self.N2[next_key]

            if index <= self.d_diffuse:
                if all(abs(ravel(self.F_infinity[key])) <= EPSILON):
                    self.r0[key] = dot(
                        dot(self.Z[key].T, self.F_inverse[key]), self.v[key]
                    ) - dot(self.L0[key].T, next_r0)
                    self.r1[key] = dot(self.T[key].T, next_r1)
                    self.N0[key] = dot(
                        dot(self.Z[key].T, self.F_inverse[key]), self.Z[key]
                    ) + dot(dot(self.L0[key].T, next_N0), self.L0[key])
                    self.N1[key] = dot(dot(self.T[key].T, next_N1), self.L0[key])
                    self.N2[key] = dot(dot(self.T[key], next_N2), self.T[key])
                else:
                    self.r0[key] = dot(self.L0[key].T, next_r0)
                    self.r1[key] = (
                        dot(dot(self.Z[key].T, self.F1[key]), self.v[key])
                        + dot(self.L0[key].T, next_r1)
                        + dot(self.L1[key].T, next_r0)
                    )
                    self.N0[key] = dot(dot(self.L0[key].T, next_N0), self.L0[key])
                    self.N1[key] = (
                        dot(dot(self.Z[key].T, self.F1[key]), self.Z[key])
                        + dot(dot(self.L0[key].T, next_N1), self.L0[key])
                        + dot(dot(self.L1[key].T, next_N0), self.L0[key])
                    )
                    self.N2[key] = (
                        dot(dot(self.Z[key].T, self.F2[key]), self.Z[key])
                        + dot(dot(self.L0[key].T, next_N2), self.L0[key])
                        + dot(dot(self.L0[key].T, next_N1), self.L1[key])
                        + dot(dot(self.L1[key].T, next_N1), self.L0[key])
                        + dot(dot(self.L1[key].T, next_N0), self.L1[key])
                    )

                self.a_hat[key] = (
                    self.a_prior[key]
                    + dot(self.P_star_prior[key], self.r0[key])
                    + dot(self.P_infinity_prior[key], self.r1[key])
                )
                self.V[key] = (
                    self.P_star_prior[key]
                    - dot(
                        dot(self.P_star_prior[key], self.N0[key]),
                        self.P_star_prior[key],
                    )
                    - (
                        dot(
                            dot(self.P_infinity_prior[key], self.N1[key]),
                            self.P_star_prior[key],
                        )
                    ).T
                    - dot(
                        dot(self.P_infinity_prior[key], self.N1[key]),
                        self.P_star_prior[key],
                    )
                    - dot(
                        dot(self.P_infinity_prior[key], self.N2[key]),
                        self.P_infinity_prior[key],
                    )
                )
            else:
                self.r[key] = dot(ZF_inv, self.v[key]) + dot(self.L[key].T, next_r)
                self.N[key] = dot(ZF_inv, self.Z[key]) + dot(
                    dot(self.L[key].T, next_N), self.L[key]
                )
                if index == self.d_diffuse + 1:
                    self.r0[key] = self.r[key]
                    self.r1[key] = zeros((self.m, 1))
                    self.N0[key] = self.N[key].copy()
                    self.N1[key] = zeros((self.m, self.m))
                    self.N2[key] = zeros((self.m, self.m))

                self.a_hat[key] = self.a_prior[key] + dot(
                    self.P_prior[key], self.r[key]
                )
                self.V[key] = self.P_prior[key] - dot(
                    dot(self.P_prior[key], self.N[key]), self.P_prior[key]
                )

        self.smoother_run = True

    def disturbance_smoother(self):
        """"""
        if not self.smoother_run:
            self.smoother()

        for index, key in reversed(list(enumerate(self.index))):
            if index <= self.d_diffuse:
                next_index = index + 1
                try:
                    next_key = self.index[next_index]
                except IndexError:
                    next_r0 = self.r_final
                    next_N0 = self.N_final
                else:
                    next_r0 = self.r0[next_key]
                    next_N0 = self.N0[next_key]

                if det(self.F_infinity[key]) == 0:
                    self.epsilon_hat[key] = dot(
                        self.H[key],
                        dot(self.F_inverse[key], self.v[key])
                        - dot(self.K0[key], next_r0),
                    )
                    self.epsilon_hat_sigma2[key] = self.H[key] - dot(
                        dot(
                            self.H[key],
                            self.F_inverse[key]
                            + dot(dot(self.K0[key].T, next_N0), self.K0[key]),
                        ),
                        self.H[key],
                    )
                else:
                    self.epsilon_hat[key] = -1 * dot(
                        dot(self.H[key], self.K0[key].T), next_r0
                    )
                    self.epsilon_hat_sigma2[key] = self.H[key] - dot(
                        dot(
                            dot(dot(self.H[key], self.K0[key].T), next_N0), self.K0[key]
                        ),
                        self.H[key],
                    )

                self.eta_hat[key] = dot(dot(self.Q[key], self.R[key].T), next_r0)
                self.eta_hat_sigma2[key] = self.Q[key] - dot(
                    dot(dot(dot(self.Q[key], self.R[key].T), next_N0), self.R[key]),
                    self.Q[key],
                )
            else:
                self.u[key] = dot(self.F_inverse[key], self.v[key])
                self.D[key] = self.F_inverse[key].copy()
                self.eta_hat_sigma2[key] = self.Q[key].copy()
                QR = dot(self.Q[key], self.R[key].T)

                next_index = index + 1
                try:
                    next_key = self.index[next_index]
                except IndexError:
                    self.eta_hat[key] = dot(QR, self.r_final)
                else:
                    self.u[key] = self.u[key] - dot(self.K[key].T, self.r[next_key])
                    self.D[key] = self.D[key] + dot(
                        dot(self.K[key].T, self.N[next_key]), self.K[key]
                    )

                    self.eta_hat[key] = dot(QR, self.r[next_key])
                    self.eta_hat_sigma2[key] = self.eta_hat_sigma2[key] - dot(
                        dot(QR, self.N[next_key]), QR.T
                    )

                self.epsilon_hat[key] = dot(self.H[key], self.u[key])
                HDH = dot(dot(self.H[key], self.D[key]), self.H[key])
                self.epsilon_hat_sigma2[key] = self.H[key] - HDH

        self.disturbance_smoother_run = True

    def simulate_smoother(self):
        """"""
        if not self.disturbance_smoother_run:
            self.disturbance_smoother()

        model_fields = ["Z", "d", "H", "T", "c", "R", "Q"]
        model_data_df = self.model_data_df[model_fields]

        if self.d_diffuse >= 0:
            a0 = self.a_hat[self.initial_index]
            V0 = self.V[self.initial_index]
            sim_df = self.simulate_model(model_data_df, a0, V0)
            P0 = zeros((self.m, self.m))
        else:
            a0 = self.a_prior[self.initial_index]
            P0 = self.P_prior[self.initial_index]
            sim_df = self.simulate_model(model_data_df, a0, P0)

        for key in self.index:
            sim_df.loc[key, "y"] = self.copy_missing(sim_df.loc[key, "y"], self.y[key])

        sim_ssm = DynamicLinearGaussianModel(sim_df["y"], model_data_df, a0, P0)
        sim_ssm.disturbance_smoother()

        sim_series_names = ["alpha", "epsilon", "eta"]
        ssm_series_names = ["a_hat", "epsilon_hat", "eta_hat"]
        estimation_error = pd.DataFrame()
        result_series = pd.DataFrame()

        for index, sim_col in enumerate(sim_series_names):
            ssm_col = ssm_series_names[index]
            estimation_error[sim_col] = sim_df[sim_col] - sim_ssm.model_data_df[ssm_col]
            result_series[sim_col] = (
                estimation_error[sim_col] + self.model_data_df[ssm_col]
            )

        return result_series

    def log_likelihood(self):
        """"""
        if not self.filter_run:
            self.filter()

        result = 0
        for index, key in enumerate(self.index):
            if self.p[key] > 0:
                result += self.p[key] * log(2 * PI)
                F_inv = self.F_inverse[key]
                if index <= self.d_diffuse and F_inv is pd.NA:
                    result += self.w(index, key)
                else:
                    result -= log(det(F_inv))
                    result += dot(dot(self.v[key].T, F_inv), self.v[key])[0, 0]
        result *= -0.5

        return result

    def w(self, index, key):
        """"""
        if index > self.d_diffuse:
            return 0

        if self.F_inverse[key] is pd.NA:
            return log(det(self.F_infinity[key]))
        return dot(dot(self.v[key], self.F_inverse[key]), self.v[key])[0, 0] - log(
            det(self.F_inverse[key])
        )

    def __getattr__(self, name):
        """"""
        try:
            return self.model_data_df[name]
        except KeyError:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(type(self).__name__, name)
            )

    @staticmethod
    def is_all_missing(value):
        """"""
        try:
            if value is pd.NA or value is None or isnan(value):
                return True
        except (TypeError, ValueError):
            pass

        try:
            if all([DynamicLinearGaussianModel.is_all_missing(val) for val in value]):
                return True
        except TypeError:
            pass

        return False

    @staticmethod
    def is_partial_missing(value):
        """"""
        try:
            if any([DynamicLinearGaussianModel.is_all_missing(val) for val in value]):
                return True
        except TypeError:
            pass

        return False

    @staticmethod
    def remove_missing_rows(value, ref):
        """"""
        not_missing_mask = [
            not DynamicLinearGaussianModel.is_all_missing(val) for val in ref
        ]
        result = array(value)
        if len(result.shape) == 2:
            return result[not_missing_mask, :]
        return result[not_missing_mask]

    @staticmethod
    def copy_missing(value, ref):
        """"""
        try:
            for index, val in enumerate(ref):
                if DynamicLinearGaussianModel.is_all_missing(val):
                    try:
                        value[index][0] = pd.NA
                    except TypeError:
                        value[index] = pd.NA
        except TypeError:
            if DynamicLinearGaussianModel.is_all_missing(ref):
                value = pd.NA

        return value

    @staticmethod
    def simulate_model(model_data_df, a0, P0):
        """"""
        series_length = len(model_data_df)
        result_dict = {
            "y": [pd.NA] * series_length,
            "alpha": [pd.NA] * series_length,
            "epsilon": [pd.NA] * series_length,
            "eta": [pd.NA] * series_length,
        }
        result_df = pd.DataFrame(result_dict, index=model_data_df.index)

        for index, key in enumerate(model_data_df.index):
            prev_index = index - 1

            if prev_index == -1:
                curr_alpha = mv_norm(a0.ravel(), P0)
                curr_alpha.shape = (curr_alpha.shape[0], 1)
            else:
                prev_key = model_data_df.index[prev_index]
                prev_alpha = result_df.loc[prev_key, "alpha"]
                curr_alpha_mu = (
                    dot(model_data_df.loc[key, "T"], prev_alpha)
                    + model_data_df.loc[key, "c"]
                )
                curr_R = model_data_df.loc[key, "R"]
                curr_Q = model_data_df.loc[key, "Q"]
                curr_eta = mv_norm(zeros(curr_Q.shape[0]), curr_Q)
                curr_eta.shape = (curr_eta.shape[0], 1)
                curr_alpha = curr_alpha_mu + dot(curr_R, curr_eta)

                result_df.loc[prev_key, "eta"] = curr_eta

            curr_H = model_data_df.loc[key, "H"]
            curr_Y_mu = (
                dot(model_data_df.loc[key, "Z"], curr_alpha)
                + model_data_df.loc[key, "d"]
            )
            curr_epsilon = mv_norm(zeros(curr_H.shape[0]), curr_H)
            curr_epsilon.shape = (curr_epsilon.shape[0], 1)
            curr_Y = curr_Y_mu + curr_epsilon

            result_df.loc[key, "epsilon"] = curr_epsilon
            result_df.loc[key, "alpha"] = curr_alpha
            result_df.loc[key, "y"] = curr_Y

        return result_df

    @staticmethod
    def diffuse_P(P_star, P_infinity):
        """"""
        result = P_star.copy()
        result[abs(P_infinity) > EPSILON] = inf
        return result
