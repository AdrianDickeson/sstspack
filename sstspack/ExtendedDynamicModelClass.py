from sstspack import DynamicLinearGaussianModel


class ExtendedDynamicModel(DynamicLinearGaussianModel):
    """"""

    expected_columns = ("Z_fn", "H_fn", "T_fn", "R_fn", "Q_fn")
    estimation_columns = [
        "Z",
        "H",
        "T",
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
