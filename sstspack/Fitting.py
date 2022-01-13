import logging

from numpy import (
    inf,
    exp,
    sqrt,
    isinf,
    log,
    array,
    zeros,
    ravel,
    set_printoptions,
    dot,
    reshape,
    prod,
    diag,
    sin,
    cos,
)
from numpy.linalg import inv
from scipy.optimize import minimize

from sstspack import DynamicLinearGaussianModel as DLGM
from sstspack.Utilities import jacobian, hessian

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def parameter_transform_function(parameter_bounds):
    """"""
    lower_bound = parameter_bounds[0]
    upper_bound = parameter_bounds[1]

    def unconstrained(x):
        return x

    def constrained_upper_half_interval(x):
        return exp(x) + lower_bound

    def constrained_lower_half_interval(x):
        return -exp(-x) + upper_bound

    def constrained_closed_interval(x):
        half_range = 0.5 * (upper_bound - lower_bound)
        result = half_range * x / sqrt(1 + x ** 2)
        return result + (lower_bound + half_range)

    if isinf(lower_bound) and isinf(upper_bound):
        return unconstrained
    if isinf(upper_bound):
        return constrained_upper_half_interval
    if isinf(lower_bound):
        return constrained_lower_half_interval
    return constrained_closed_interval


def inverse_parameter_transform_function(parameter_bounds):
    """"""
    lower_bound = parameter_bounds[0]
    upper_bound = parameter_bounds[1]

    def inverse_unconstrained(x):
        return x

    def inverse_constrained_upper_half_interval(x):
        return log(x - lower_bound)

    def inverse_constrained_lower_half_interval(x):
        return -log(-x + upper_bound)

    def inverse_constrained_closed_interval(x):
        # translate interval to (-1, 1)
        half_range = 0.5 * (upper_bound - lower_bound)
        mid_point = lower_bound + half_range
        x1 = (x - mid_point) / half_range
        return x1 / sqrt(1 - x1 ** 2)

    if isinf(lower_bound) and isinf(upper_bound):
        return inverse_unconstrained
    if isinf(upper_bound):
        return inverse_constrained_upper_half_interval
    if isinf(lower_bound):
        return inverse_constrained_lower_half_interval
    return inverse_constrained_closed_interval


class FittedModel:
    def __repr__(self):
        return "\n".join(f"{key}:\t{self.__dict__[key]}" for key in self.__dict__)

    def __str__(self):
        set_printoptions(precision=5)
        warning = "\nWarning: {}".format(self.message) if not self.success else ""
        parameters = self.parameter_field_to_str(self.parameters)
        jacobian = self.parameter_field_to_str(self.jacobian)

        return f"""Maximum Likelihood Results
--------------------------
Maximum Log Likelihood Found: {self.log_likelihood:.5}{warning}
Parameters:
{parameters}
Jacobian:
{jacobian}
Variance Matrix:
{self.fisher_information_matrix}"""

    def parameter_field_to_str(self, field_data):
        """"""
        parameter_names = self.parameter_names
        if parameter_names is None:
            parameter_names = [
                f"Parameter {idx}" for idx in range(len(self.parameters))
            ]

        return "\n".join(
            f"{parameter_names[idx]}: {field_data[idx]:.6}"
            for idx in range(len(self.parameters))
        )


def fit_model_max_likelihood(
    initial_parameter_values,
    params_bounds,
    model_func,
    y_series,
    **kwargs,
):
    """"""
    # Obtain optional variables from kwargs
    parameter_names = kwargs["parameter_names"] if "parameter_names" in kwargs else None

    kwargs["y_series"] = y_series

    n = len(y_series)
    initial_objective_parameters = [
        inverse_parameter_transform_function(params_bounds[idx])(value)
        for idx, value in enumerate(initial_parameter_values)
    ]

    def objective_func(transformed_params, kwargs):
        params = [
            parameter_transform_function(params_bounds[idx])(value)
            for idx, value in enumerate(transformed_params)
        ]
        return inner_objective_func(params, kwargs)

    def inner_objective_func(params, kwargs):
        model = model_func(params, **kwargs)
        return -model.log_likelihood()

    res = minimize(
        objective_func,
        initial_objective_parameters,
        options={"disp": False},
        args=(kwargs,),
        method="BFGS",
        tol=1.0e-6,
    )

    result = FittedModel()
    result.count_iteration = res.nit
    result.count_function_evaluations = res.nfev
    result.count_gradient_evaluations = res.njev

    domain_params = array(
        [
            parameter_transform_function(params_bounds[idx])(value)
            for idx, value in enumerate(res.x)
        ]
    )

    hess = hessian(
        inner_objective_func,
        domain_params,
        1e-10,
        False,
        kwargs,
    )

    result.parameters = domain_params
    dimension = len(domain_params)
    result.parameter_names = parameter_names
    result.log_likelihood = -res.fun
    logger.debug(f"Best log likelihood found: {result.log_likelihood:.2f}")

    result.message = res.message
    result.status = res.status
    result.success = res.success
    result.jacobian = -res.jac
    result.fisher_information_matrix = inv(hess)
    result.akaike_information_criterion = akaike_information_criterion(
        result.log_likelihood, dimension
    )
    result.bayesian_information_criterion = bayesian_information_criterion(
        result.log_likelihood, dimension, n
    )

    model = model_func(domain_params, **kwargs)
    result.model = model

    return result


def akaike_information_criterion(log_likelihood, dimension):
    """"""
    return 2 * dimension - 2 * log_likelihood


def bayesian_information_criterion(log_likelihood, dimension, n):
    """"""
    return dimension * log(n) - 2 * log_likelihood


def correlation_matrix(anglar_coordenants):
    """"""
    len_coords = len(anglar_coordenants)
    n = int((1 + sqrt(1 + 4 * len_coords)) / 2)

    theta = reshape(anglar_coordenants, (n, n - 1))
    B = zeros((n, n))

    for i in range(n):
        for j in range(n):
            if j < n - 1:
                B[i, j] = cos(theta[i, j]) * prod([sin(x) for x in theta[i, :j]])
            else:
                B[i, j] = prod([sin(x) for x in theta[i, :j]])

    return dot(B, B.T)


def correlation_to_variance_matrix(correlation_matrix, variances):
    """"""
    std_deviations = sqrt(variances)
    V = diag(std_deviations)

    return dot(dot(V, correlation_matrix), V)
