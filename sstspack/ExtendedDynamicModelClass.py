from numpy import dot, reshape, zeros, identity, ravel
from numpy.linalg import inv, LinAlgError
import pandas as pd

from sstspack import DynamicLinearGaussianModel
from sstspack.Utilities import jacobian
from sstspack.DynamicLinearGaussianModelClass import EPSILON


class ExtendedDynamicModel(DynamicLinearGaussianModel):
    """"""

    expected_columns = ("Z_fn", "H_fn", "T_fn", "R_fn", "Q_fn")
    estimation_columns = [
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
    ] + DynamicLinearGaussianModel.estimation_columns

    def __init__(
        self,
        y_series,
        model_design_df,
        a_prior_initial,
        P_prior_initial,
        diffuse_states=None,
        validate_input=True,
    ):
        """"""
        DynamicLinearGaussianModel.__init__(
            self,
            y_series,
            model_design_df,
            a_prior_initial,
            P_prior_initial,
            diffuse_states,
            validate_input,
        )
        self.initial_smoother_run = False

    def _initialise_model_data(self, a_prior_initial):
        """"""
        self._m = a_prior_initial.shape[0]

        for key in self.index:
            self.Z[key] = self.Z_fn[key](a_prior_initial)
            self.H[key] = self.H_fn[key](a_prior_initial)
            self.T[key] = self.T_fn[key](a_prior_initial)
            self.R[key] = self.R_fn[key](a_prior_initial)
            self.Q[key] = self.Q_fn[key](a_prior_initial)

    def _verification_columns(self, p, idx):
        """"""
        return {
            "Z": (p[idx], 1),
            "H": (p[idx], p[idx]),
            "T": (self.m, 1),
            "R": (self.m, self.r_eta),
            "Q": (self.r_eta, self.r_eta),
        }

    def _prediction_error(self, key):
        """"""
        return self.y[key] - self.Z[key]

    def _non_missing_F(self, key):
        """"""
        self._initialise_data_fn(key)
        PZ = dot(self.P_prior[key], self.Z_prime[key].T)
        self.F[key] = dot(self.Z_prime[key], PZ) + self.H[key]

    def _diffuse_filter_recursion_step(self, key, index):
        """"""
        self.v[key] = self._prediction_error(key)
        self.F_infinity[key] = dot(
            dot(self.Z_prime[key], self.P_infinity_prior[key]), self.Z_prime[key].T
        )
        self.F_star[key] = (
            dot(dot(self.Z_prime[key], self.P_star_prior[key]), self.Z_prime[key].T)
            + self.H[key]
        )

        self.M_infinity[key] = dot(self.P_infinity_prior[key], self.Z_prime[key].T)
        self.M_star[key] = dot(self.P_star_prior[key], self.Z_prime[key].T)

        try:
            self.F1[key] = inv(self.F_infinity[key])
        except LinAlgError:
            F = self.F_star[key].copy()
            self.F_inverse[key] = inv(F)

            K0_hat = dot(self.M_star[key], self.F_inverse[key])
            K1_hat = zeros((self.m, self.Z_prime[key].shape[0]))
        else:
            self.F2[key] = -1 * dot(dot(self.F1[key], self.F_star[key]), self.F1[key])

            K0_hat = dot(self.M_infinity[key], self.F1[key])
            K1_hat = dot(self.M_star[key], self.F1[key]) + dot(
                self.M_infinity[key], self.F2[key]
            )

        L0_hat = identity(self.m) - dot(K0_hat, self.Z_prime[key])
        L1_hat = -1 * dot(K1_hat, self.Z_prime[key])

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

        self._initialise_state_fn(key)

        self.K0[key] = dot(self.T_prime[key], K0_hat)
        self.K1[key] = dot(self.T_prime[key], K1_hat)
        self.L0[key] = dot(self.T_prime[key], L0_hat)
        self.L1[key] = dot(self.T_prime[key], L1_hat)

        RQR = dot(dot(self.R[key], self.Q[key]), self.R[key].T)
        a_prior_next = dot(self.T_prime[key], self.a_posterior[key])
        P_prior_next = (
            dot(dot(self.T_prime[key], self.P_posterior[key]), self.T_prime[key].T)
            + RQR
        )

        P_infinity_prior_next = dot(
            dot(self.T_prime[key], self.P_infinity_posterior[key]), self.T_prime[key].T
        )
        P_star_prior_next = (
            dot(dot(self.T_prime[key], self.P_star_posterior[key]), self.T_prime[key].T)
            + RQR
        )
        P_prior_next = self.diffuse_P(P_star_prior_next, P_infinity_prior_next)

        return {
            "a_prior": a_prior_next,
            "P_prior": P_prior_next,
            "P_infinity_prior": P_infinity_prior_next,
            "P_star_prior": P_star_prior_next,
        }

    def _filter_recursion_step(self, key):
        """"""
        if not self.initial_smoother_run:
            a_prior = self.a_prior
            a_posterior = self.a_posterior
            P_prior = self.P_prior
            P_posterior = self.P_posterior

            v = self.v
            Z_prime = self.Z_prime
            H = self.H
            F_inverse = self.F_inverse
            T = self.T
            T_prime = self.T_prime
            R = self.R
            Q = self.Q
        else:
            a_prior = self.a_hat_prior
            a_posterior = self.a_hat_posterior
            P_prior = self.P_hat_prior
            P_posterior = self.P_hat_posterior

            v = self.v_hat
            Z_prime = self.Z_hat_prime
            H = self.H_hat
            F_inverse = self.F_hat_inverse
            T = self.T_hat
            T_prime = self.T_hat_prime
            R = self.R_hat
            Q = self.Q_hat

        v[key] = self._prediction_error(key)
        PZ = dot(P_prior[key], Z_prime[key].T)

        F = dot(Z_prime[key], PZ) + H[key]

        try:
            F_inv = inv(F)
        except LinAlgError:
            F_inv = 0
        PZF_inv = dot(PZ, F_inv)

        F_inverse[key] = F_inv
        a_posterior[key] = a_prior[key] + dot(PZF_inv, v[key])
        P_posterior[key] = P_prior[key] - dot(PZF_inv, PZ.T)

        self._initialise_state_fn(key)

        RQR = dot(dot(R[key], Q[key]), R[key].T)
        a_prior_next = T[key]
        P_prior_next = dot(dot(T_prime[key], P_posterior[key]), T_prime[key].T) + RQR
        if not self.initial_smoother_run:
            return {"a_prior": a_prior_next, "P_prior": P_prior_next}
        else:
            return {"a_hat_prior": a_prior_next, "P_hat_prior": P_prior_next}

    def _initialise_data_fn(self, key):
        """"""
        if not self.initial_smoother_run:
            self.Z[key] = self.Z_fn[key](self.a_prior[key])
            self.Z_prime[key] = reshape(
                jacobian(self.Z_fn[key], self.a_prior[key], relative=False),
                (self.p[key], self.m),
            )
            self.H[key] = self.H_fn[key](self.a_prior[key])
        else:
            self.Z_hat_prime[key] = reshape(
                jacobian(self.Z_fn[key], self.a_hat_initial[key], relative=False),
                (self.p[key], self.m),
            )
            self.H_hat[key] = self.H_fn[key](self.a_hat_initial[key])

    def _initialise_state_fn(self, key):
        """"""
        if not self.initial_smoother_run:
            self.T[key] = self.T_fn[key](self.a_posterior[key])
            self.T_prime[key] = reshape(
                jacobian(self.T_fn[key], self.a_posterior[key], relative=False),
                (self.m, self.m),
            )
            self.R[key] = self.R_fn[key](self.a_posterior[key])
            self.Q[key] = self.Q_fn[key](self.a_posterior[key])
        else:
            self.T_hat[key] = self.T_fn[key](self.a_hat_initial[key])
            self.T_hat_prime[key] = reshape(
                jacobian(self.T_fn[key], self.a_hat_initial[key], relative=False),
                (self.m, self.m),
            )
            self.R_hat[key] = self.R_fn[key](self.a_hat_initial[key])
            self.Q_hat[key] = self.Q_fn[key](self.a_hat_initial[key])

    def aggregate_field(self, field, mask=None):
        """"""
        data = []

        for idx in self.index:
            Z = mask
            if Z is None:
                Z = self.Z_prime[idx]
            value = dot(Z, self.model_data_df.loc[idx, field])
            if value.shape == (1, 1):
                value = value[0, 0]
            data.append(value)

        return pd.Series(data, index=self.index)

    def quadratic_aggregate_field(self, field, mask=None):
        """"""
        data = []

        for idx in self.index:
            Z = mask
            if Z is None:
                Z = self.Z_prime[idx]
            value = dot(dot(Z, self.model_data_df.loc[idx, field]), Z.T)
            if value.shape == (1, 1):
                value = value[0, 0]
            data.append(value)

        return pd.Series(data, index=self.index)

    def _diffuse_smoother_recursion_step(self, key, index):
        """"""
        next_index = index + 1
        try:
            next_key = self.index[next_index]
        except IndexError:
            next_r0 = self.r0_final
            next_r1 = self.r1_final
            next_N0 = self.N0_final
            next_N1 = self.N1_final
            next_N2 = self.N2_final
        else:
            next_r0 = self.r0[next_key]
            next_r1 = self.r1[next_key]
            next_N0 = self.N0[next_key]
            next_N1 = self.N1[next_key]
            next_N2 = self.N2[next_key]

        if all(abs(ravel(self.F_infinity[key])) <= EPSILON):
            self.r0[key] = dot(
                dot(self.Z_prime[key].T, self.F_inverse[key]), self.v[key]
            ) - dot(self.L0[key].T, next_r0)
            self.r1[key] = dot(self.T_prime[key].T, next_r1)
            self.N0[key] = dot(
                dot(self.Z_prime[key].T, self.F_inverse[key]), self.Z_prime[key]
            ) + dot(dot(self.L0[key].T, next_N0), self.L0[key])
            self.N1[key] = dot(dot(self.T_prime[key].T, next_N1), self.L0[key])
            self.N2[key] = dot(dot(self.T_prime[key], next_N2), self.T_prime[key])
        else:
            self.r0[key] = dot(self.L0[key].T, next_r0)
            self.r1[key] = (
                dot(dot(self.Z_prime[key].T, self.F1[key]), self.v[key])
                + dot(self.L0[key].T, next_r1)
                + dot(self.L1[key].T, next_r0)
            )
            self.N0[key] = dot(dot(self.L0[key].T, next_N0), self.L0[key])
            self.N1[key] = (
                dot(dot(self.Z_prime[key].T, self.F1[key]), self.Z_prime[key])
                + dot(dot(self.L0[key].T, next_N1), self.L0[key])
                + dot(dot(self.L1[key].T, next_N0), self.L0[key])
            )
            self.N2[key] = (
                dot(dot(self.Z_prime[key].T, self.F2[key]), self.Z_prime[key])
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

    def _smoother_recursion_step(self, key, index):
        """"""
        if not self.initial_smoother_run:
            a_prior = self.a_prior
            a_hat = self.a_hat_initial
            P_prior = self.P_prior
            V = self.V_initial

            v = self.v
            Z_prime = self.Z_prime
            T_prime = self.T_prime
            F_inverse = self.F_inverse
            K = self.K
            L = self.L

            r = self.r
            N = self.N
            r0 = self.r0
            r1 = self.r1
            N0 = self.N0
            N1 = self.N1
            N2 = self.N2
        else:
            a_prior = self.a_hat_prior
            a_hat = self.a_hat
            P_prior = self.P_hat_prior
            V = self.V

            v = self.v_hat
            Z_prime = self.Z_hat_prime
            T_prime = self.T_hat_prime
            F_inverse = self.F_hat_inverse
            K = self.K_hat
            L = self.L_hat

            r = self.r_hat
            N = self.N_hat
            r0 = self.r0_hat
            r1 = self.r1_hat
            N0 = self.N0_hat
            N1 = self.N1_hat
            N2 = self.N2_hat

        ZF_inv = dot(Z_prime[key].T, F_inverse[key])
        K[key] = dot(dot(T_prime[key], P_prior[key]), ZF_inv)
        L[key] = T_prime[key] - dot(K[key], Z_prime[key])

        next_index = index + 1

        try:
            next_key = self.index[next_index]
        except IndexError:
            next_r = self.r_final
            next_N = self.N_final
        else:
            next_r = r[next_key]
            next_N = N[next_key]

        r[key] = dot(ZF_inv, v[key]) + dot(L[key].T, next_r)
        N[key] = dot(ZF_inv, Z_prime[key]) + dot(dot(L[key].T, next_N), L[key])
        if index == self.d_diffuse + 1:
            r0[key] = r[key].copy()
            r1[key] = zeros((self.m, 1))
            N0[key] = N[key].copy()
            N1[key] = zeros((self.m, self.m))
            N2[key] = zeros((self.m, self.m))

        a_hat[key] = a_prior[key] + dot(P_prior[key], r[key])
        V[key] = P_prior[key] - dot(dot(P_prior[key], N[key]), P_prior[key])

    def _finish_smoother(self):
        """"""
        if not self.initial_smoother_run:
            self.initial_smoother_run = True
            self.a_hat_prior[self.initial_index] = self.a_prior[
                self.initial_index
            ].copy()
            self.P_hat_prior[self.initial_index] = self.P_prior[
                self.initial_index
            ].copy()

            self.filter()
            self.smoother()
