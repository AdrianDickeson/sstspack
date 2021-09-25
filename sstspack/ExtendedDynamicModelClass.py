from numpy import dot, reshape, zeros, identity, ravel, full
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
        "F1_hat",
        "F2_hat",
        "L0_hat",
        "L1_hat",
        "K0_hat",
        "K1_hat",
        "P_hat_infinity_prior",
        "P_hat_infinity_posterior",
        "P_hat_star_prior",
        "P_hat_star_posterior",
        "F_hat_infinity",
        "F_hat_star",
        "M_hat_infinity",
        "M_hat_star",
    ] + DynamicLinearGaussianModel.estimation_columns

    def __init__(
        self,
        y_series,
        model_design_df,
        a_prior_initial=None,
        P_prior_initial=None,
        diffuse_states=None,
        validate_input=True,
    ):
        """"""
        self.column_redirects = {}
        self.initial_smoother_run = False

        DynamicLinearGaussianModel.__init__(
            self,
            y_series,
            model_design_df,
            a_prior_initial,
            P_prior_initial,
            diffuse_states,
            validate_input,
        )

    def __getattr__(self, name):
        """"""
        if name in ["c", "d"]:
            return pd.Series([full((1, 1), 0)] * len(self.index), index=self.index)

        if name in self.column_redirects:
            name = self.column_redirects[name]

        return DynamicLinearGaussianModel.__getattr__(self, name)

    def _add_column_redirects(self):
        """"""
        if not self.initial_smoother_run:
            self.column_redirects = {
                # state terms
                "a_hat": "a_hat_initial",
                "V": "V_initial",
                # model terms
                "Z": "Z_prime",
                "T": "T_prime",
                # estimation terms
                "K": "K_hat",
                "L": "L_hat",
                "r": "r_hat",
                "N": "N_hat",
                "r_final": "r_hat_final",
                "N_final": "N_hat_final",
                "r0": "r0_hat",
                "r1": "r1_hat",
                "N0": "N0_hat",
                "N1": "N1_hat",
                "N2": "N2_hat",
                "r0_final": "r0_hat_final",
                "r1_final": "r1_hat_final",
                "N0_final": "N0_hat_final",
                "N1_final": "N1_hat_final",
                "N2_final": "N2_hat_final",
            }

        else:
            self.column_redirects = {
                # state terms
                "a_prior": "a_hat_prior",
                "a_posterior": "a_hat_posterior",
                "P_prior": "P_hat_prior",
                "P_posterior": "P_hat_posterior",
                "P_infinity_prior": "P_hat_infinity_prior",
                "P_infinity_posterior": "P_hat_infinity_posterior",
                "P_star_prior": "P_hat_star_prior",
                "P_star_posterior": "P_hat_star_posterior",
                # model terms
                "Z": "Z_hat_prime",
                "H": "H_hat",
                "T": "T_hat_prime",
                "R": "R_hat",
                "Q": "Q_hat",
                # estimation terms
                "v": "v_hat",
                "F_inverse": "F_hat_inverse",
                "F_infinity": "F_hat_infinity",
                "F_star": "F_hat_star",
                "M_infinity": "M_hat_infinity",
                "M_star": "M_hat_star",
                "F1": "F1_hat",
                "F2": "F2_hat",
                "K0": "K0_hat",
                "K1": "K1_hat",
                "L0": "L0_hat",
                "L1": "L1_hat",
            }

    def _initialise_model_data(self, a_prior_initial):
        """"""
        self._m = a_prior_initial.shape[0]

        for idx in self.index:
            self.Z[idx] = self.Z_fn[idx](a_prior_initial)
            self.H[idx] = self.H_fn[idx](a_prior_initial)
            self.T[idx] = self.T_fn[idx](a_prior_initial)
            self.R[idx] = self.R_fn[idx](a_prior_initial)
            self.Q[idx] = self.Q_fn[idx](a_prior_initial)

        self._add_column_redirects()

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
        if self.initial_smoother_run:
            return self.y[key] - self.model_data_df.Z_hat[key]
        return self.y[key] - self.model_data_df.Z[key]

    def _non_missing_F(self, key):
        """"""
        self._initialise_data_fn(key)
        return DynamicLinearGaussianModel._non_missing_F(self, key)

    def _diffuse_filter_posterior_recursion_step(self, key):
        """"""
        self._initialise_state_fn(key)
        return DynamicLinearGaussianModel._diffuse_filter_posterior_recursion_step(
            self, key
        )

    def _filter_posterior_recursion_step(self, key):
        """"""
        self._initialise_state_fn(key)
        return DynamicLinearGaussianModel._filter_posterior_recursion_step(self, key)

    def _initialise_data_fn(self, key):
        """"""
        if not self.initial_smoother_run:
            self.model_data_df.Z[key] = self.Z_fn[key](self.a_prior[key])
            if "Z_prime_fn" in self.model_data_df.columns:
                self.Z_prime[key] = self.Z_prime_fn[key](self.a_prior[key])
            else:
                self.Z_prime[key] = reshape(
                    jacobian(self.Z_fn[key], self.a_prior[key], h=1e-10),
                    (self.p[key], self.m),
                )
            self.H[key] = self.H_fn[key](self.a_prior[key])
        else:
            self.model_data_df.Z_hat[key] = self.Z_fn[key](self.a_hat_prior[key])
            if "Z_prime_fn" in self.model_data_df.columns:
                self.Z_hat_prime[key] = self.Z_prime_fn[key](self.a_hat_prior[key])
            else:
                self.Z_hat_prime[key] = reshape(
                    jacobian(self.Z_fn[key], self.a_hat_initial[key], h=1e-10),
                    (self.p[key], self.m),
                )
            self.H_hat[key] = self.H_fn[key](self.a_hat_initial[key])

    def _initialise_state_fn(self, key):
        """"""
        if not self.initial_smoother_run:
            self.model_data_df["T"][key] = self.T_fn[key](self.a_posterior[key])
            if "T_prime_fn" in self.model_data_df.columns:
                self.T_prime[key] = self.T_prime_fn[key](self.a_posterior[key])
            else:
                self.T_prime[key] = reshape(
                    jacobian(self.T_fn[key], self.a_posterior[key], h=1e-10),
                    (self.m, self.m),
                )
            self.R[key] = self.R_fn[key](self.a_posterior[key])
            self.Q[key] = self.Q_fn[key](self.a_posterior[key])
        else:
            self.T_hat[key] = self.T_fn[key](self.a_hat_initial[key])
            if "T_prime_fn" in self.model_data_df.columns:
                self.T_hat_prime[key] = self.T_prime_fn[key](self.a_hat_initial[key])
            else:
                self.T_hat_prime[key] = reshape(
                    jacobian(self.T_fn[key], self.a_hat_initial[key], h=1e-10),
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

    def smoother(self):
        """"""
        self.r_hat_final = zeros((self.m, 1))
        self.N_hat_final = zeros((self.m, self.m))

        DynamicLinearGaussianModel.smoother(self)

        if not self.initial_smoother_run:
            self.initial_smoother_run = True

            self.a_hat_prior[self.initial_index] = self.a_prior[
                self.initial_index
            ].copy()
            self.P_hat_prior[self.initial_index] = self.P_prior[
                self.initial_index
            ].copy()
            if self.diffuse_model:
                self.P_hat_infinity_prior[self.initial_index] = self.P_infinity_prior[
                    self.initial_index
                ].copy()
                self.P_hat_star_prior[self.initial_index] = self.P_star_prior[
                    self.initial_index
                ].copy()

            self._add_column_redirects()

            self.filter()
            self.smoother()
