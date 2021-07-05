from numpy import dot, reshape
from numpy.linalg import inv, LinAlgError

from sstspack import DynamicLinearGaussianModel
from sstspack.Utilities import jacobian


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
    ] + DynamicLinearGaussianModel.estimation_columns

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

    def _initialise_parameters(self, key):
        """"""
        self.Z[key] = self.Z_fn[key](self.a_prior[key])
        self.Z_prime[key] = reshape(
            jacobian(self.Z_fn[key], self.a_prior[key], relative=False),
            (self.p[key], self.m),
        )

    def _non_missing_F(self, key):
        """"""
        PZ = dot(self.P_prior[key], self.Z_prime[key].T)
        self.F[key] = dot(self.Z_prime[key], PZ) + self.H[key]

    def _filter_recursion_step(self, key):
        """"""
        self.H[key] = self.H_fn[key](self.a_prior[key])

        self.v[key] = self._prediction_error(key)
        PZ = dot(self.P_prior[key], self.Z_prime[key].T)
        self.F[key] = dot(self.Z_prime[key], PZ) + self.H[key]
        try:
            self.F_inverse[key] = inv(self.F[key])
        except LinAlgError:
            self.F_inverse[key] = 0
        PZF_inv = dot(PZ, self.F_inverse[key])

        self.a_posterior[key] = self.a_prior[key] + dot(PZF_inv, self.v[key])
        self.P_posterior[key] = self.P_prior[key] - dot(PZF_inv, PZ.T)

        self.T[key] = self.T_fn[key](self.a_posterior[key])
        self.T_prime[key] = reshape(
            jacobian(self.T_fn[key], self.a_posterior[key], relative=False),
            (self.m, self.m),
        )
        self.R[key] = self.R_fn[key](self.a_posterior[key])
        self.Q[key] = self.Q_fn[key](self.a_posterior[key])
        RQR = dot(dot(self.R[key], self.Q[key]), self.R[key].T)
        a_prior_next = self.T[key]
        P_prior_next = (
            dot(dot(self.T_prime[key], self.P_posterior[key]), self.T_prime[key].T)
            + RQR
        )

        return a_prior_next, P_prior_next
