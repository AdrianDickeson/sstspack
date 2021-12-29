from numpy import sqrt, array, nan, sum, dot
from numpy.linalg import cholesky

from sstspack import DynamicLinearGaussianModel as DLGM


class UnscentedDynamicModel(DLGM):
    """"""

    unscented_columns = ["sigma_points", "sigma_weights"]

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
        DLGM.__init__(
            self,
            y_series,
            model_design_df,
            a_prior_initial,
            P_prior_initial,
            diffuse_states,
            validate_input,
        )
        self._add_columns_to_data_df(self.unscented_columns)
        self.k = 3 - self.m
        self.lambda_unscented = sqrt(self.m + self.k)

    def _diffuse_filter_prediction_recursion_step(self, key, index):
        """"""
        DLGM._diffuse_filter_prediction_recursion_step(self, key, index)

    def _filter_prediction_recursion_step(self, key):
        """"""
        P_star = cholesky(self.P_prior[key])
        self.sigma_points[key] = array([nan] * (1 + 2 * self.m))
        self.sigma_weights[key] = array([nan] * (1 + 2 * self.m))

        self.sigma_points[key][0] = self.a_prior[key]
        self.sigma_weights[key][0] = self.k / (self.m + self.k)
        for i in range(self.m):
            self.sigma_points[key][i + 1] = (
                self.a_prior[key] + self.lambda_unscented * P_star[:, i]
            )
            self.sigma_points[key][self.m + i + 1] = (
                self.a_prior[key] - self.lambda_unscented * P_star[:, i]
            )

            self.sigma_weights[key][i + 1] = 0.5 / (self.m + self.k)
            self.sigma_weights[key][self.m + i + 1] = 0.5 / (self.m + self.k)

        y_bar = sum(self.sigma_points[key] * self.sigma_weights[key])
        P_av = sum(
            self.sigma_weights
            * array(
                [
                    dot(
                        self.sigma_points[key][i] - y_bar,
                        (self.sigma_points[key][i] - y_bar).T,
                    )
                ]
            )
        )
        print(P_av)

        DLGM._filter_prediction_recursion_step(self, key)
