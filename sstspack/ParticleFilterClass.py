from numpy.random import normal, choice
from numpy import zeros, sqrt, array, exp, log, pi as PI, full
import pandas as pd


class ParticleFilter(object):
    """"""

    def __init__(self, y_series, model_design_df, a1, P1, N, resample_level, aux=False):
        """"""
        self.N = N
        self.resample_level = resample_level
        self.aux = aux
        estimation_columns = ["a", "P"]
        self.model_design = model_design_df.copy()
        self.model_design.insert(0, "y", y_series)
        self.model_design = self.model_design.reindex(
            columns=self.model_design.columns.tolist() + estimation_columns
        )

        estimation_columns = [
            "ESS",
            "particles_prior",
            "particles_posterior",
            "weights_prior",
            "weights_posterior",
            "a_prior",
            "a_posterior",
            "P_prior",
            "P_posterior",
        ]
        self.model_design = self.model_design.reindex(
            columns=self.model_design.columns.tolist() + estimation_columns
        )
        self.model_design[estimation_columns] = pd.NA

        self.a_prior[self.index_initial] = a1
        self.P_prior[self.index_initial] = P1

    def filter(self):
        """"""
        self.weights_prior[self.index_initial] = zeros(self.N)

        initial_weights = array([1 / self.N] * self.N)

        for idx, key in enumerate(self.index):
            if idx == 0:
                prev_weights = initial_weights
            else:
                prev_weights = self.weights_posterior[self.index[idx - 1]]

            # Auxiliary filter step
            if self.aux == True and idx > 0:
                weights_intermediate = self.calculate_weights(
                    key, self.particles_prior[key], prev_weights
                )

                idx_resample = choice(
                    range(self.N), self.N, replace=True, p=weights_intermediate
                )
                self.particles_prior[key] = self.particles_prior[key][idx_resample]

            # Step 1: Sample particles from importance distribution
            if idx == 0:
                self.particles_prior[key] = normal(
                    loc=self.a_prior[key],
                    scale=sqrt(self.P_prior[key]),
                    size=self.N,
                )
            else:
                self.particles_prior[key] = normal(
                    loc=self.particles_prior[key],
                    scale=sqrt(self.Q[key]),
                    size=self.N,
                )

            # Step 2: Calculate weight for particles
            self.weights_prior[key] = self.calculate_weights(
                key, self.particles_prior[key], prev_weights
            )

            # Step 3: Estimate quantities of interest
            for field in self.X[key]:
                full_field = "{}_posterior".format(field)
                self.model_design.loc[key, full_field] = self.X[key][field](
                    self.particles_prior[key], self.weights_prior[key]
                )

                if idx < len(self.index) - 1:
                    nxt_key = self.index[idx + 1]
                    nxt_full_field = "{}_prior".format(field)
                    self.model_design.loc[
                        nxt_key, nxt_full_field
                    ] = self.model_design.loc[key, full_field]

            self.ESS[key] = self.effective_sample_size(key)

            # Step 4: Calculate posterior particles
            if self.ESS[key] < self.resample_level and not self.aux:
                idx_resample = choice(
                    range(self.N), self.N, replace=True, p=self.weights_prior[key]
                )
                self.particles_posterior[key] = self.particles_prior[key][idx_resample]
                self.weights_posterior[key] = self.weights_prior[key][idx_resample]
            else:
                self.particles_posterior[key] = self.particles_prior[key].copy()
                self.weights_posterior[key] = self.weights_prior[key].copy()

            try:
                next_key = self.index[idx + 1]
            except IndexError:
                self.particles_prior_final = self.T_fn[key](
                    self.particles_posterior[key]
                )
            else:
                self.particles_prior[next_key] = self.T_fn[key](
                    self.particles_posterior[key]
                )

    def __getattr__(self, name):
        """"""
        try:
            return self.model_design[name]
        except KeyError:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(type(self).__name__, name)
            )

    @property
    def index(self):
        """"""
        return self.model_design.index

    @property
    def index_initial(self):
        """"""
        return self.index[0]

    def effective_sample_size(self, key):
        """"""
        result = sum(self.weights_prior[key] ** 2) ** -1
        return result

    def calculate_weights(self, key, particles, prev_weights):
        """"""
        sigma2 = self.H[key] + self.P_prior[key]

        weights = prev_weights * array(
            [
                exp(
                    -0.5 * log(2 * PI)
                    - 0.5 * log(sigma2)
                    - 0.5 * (self.y[key] - self.Z_fn[key](x)) ** 2 / sigma2
                )
                for x in particles
            ]
        )
        weights = weights / sum(weights)

        return weights
