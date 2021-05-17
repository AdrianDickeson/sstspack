from math import sin, cos

from numpy import (
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
)
from numpy.linalg import inv
from scipy.optimize import minimize

from sstspack import DynamicLinearGaussianModel as DLGM


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
        result = half_range * x / sqrt(1 + x * x)
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
        half_range = 0.5 * (upper_bound - lower_bound)
        mid_point = lower_bound + half_range
        term1 = half_range ** 2 / (x - lower_bound - half_range) ** 2 - 1
        term2 = sqrt(term1 ** -1)
        result = term2 if x >= mid_point else -term2
        return result

    if isinf(lower_bound) and isinf(upper_bound):
        return inverse_unconstrained
    if isinf(upper_bound):
        return inverse_constrained_upper_half_interval
    if isinf(lower_bound):
        return inverse_constrained_lower_half_interval
    return inverse_constrained_closed_interval


class FittedModel:
    def __repr__(self):
        return "\n".join(
            ["{}:\t{}".format(key, self.__dict__[key]) for key in self.__dict__]
        )

    def __str__(self):
        set_printoptions(precision=5)
        warning = ""
        if not self.success:
            warning = "\nWarning: {}".format(self.message)
        parameters = self.parameter_field_to_str(self.parameters)
        jacobian = self.parameter_field_to_str(self.jacobian)

        result = """Maximum Likelihood Results
--------------------------
Maximum Log Likelihood Found: {:.5}{}
Parameters:
{}
Jacobian:
{}
Variance Matrix
{}""".format(
            self.log_likelihood,
            warning,
            parameters,
            jacobian,
            self.fisher_information_matrix,
        )

        return result

    def parameter_field_to_str(self, field_data):
        """"""
        parameter_names = self.parameter_names
        if parameter_names is None:
            parameter_names = [
                "Parameter {}".format(idx) for idx in range(len(self.parameters))
            ]

        result = "\n".join(
            [
                "{}: {:.6}".format(parameter_names[idx], field_data[idx])
                for idx in range(len(self.parameters))
            ]
        )
        return result


def fit_model_max_likelihood(
    params0,
    params_bounds,
    model_func,
    y_series,
    a0,
    P0,
    diffuse_state,
    model_template=None,
    parameter_names=None,
    dt=None,
):
    """"""
    n = len(y_series)
    initial_params = [
        inverse_parameter_transform_function(params_bounds[idx])(value)
        for idx, value in enumerate(params0)
    ]
    param_funcs = [parameter_transform_function(bounds) for bounds in params_bounds]

    def objective_func(transformed_params, y_series, model_template, dt):
        params = [
            parameter_transform_function(params_bounds[idx])(value)
            for idx, value in enumerate(transformed_params)
        ]
        return inner_objective_func(params, y_series, model_template, dt)

    def inner_objective_func(params, y_series, model_template, dt):
        model_data = model_func(params, model_template, y_series, dt)
        model = DLGM(y_series, model_data, a0, P0, diffuse_state)
        return -model.log_likelihood()

    res = minimize(
        objective_func,
        initial_params,
        options={"disp": False},
        args=(y_series, model_template, dt),
        method="BFGS",
        tol=1.0e-16,
    )
    tparams = [
        parameter_transform_function(params_bounds[idx])(value)
        for idx, value in enumerate(res.x)
    ]
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

    jac = jacobian(
        inner_objective_func,
        domain_params,
        1e-10,
        False,
        y_series,
        model_template,
        dt,
    )

    hess = hessian(
        inner_objective_func, domain_params, 1e-10, False, y_series, model_template, dt
    )

    result.parameters = domain_params
    dimension = len(domain_params)
    result.parameter_names = parameter_names
    result.log_likelihood = -res.fun
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

    model_data = model_func(result.parameters, model_template, y_series, dt)
    result.model_data = model_data

    model = DLGM(y_series, model_data, a0, P0, diffuse_state)
    result.model = model

    return result


def jacobian(func, x, h=0.01, relative=True, *args):
    """"""
    result = zeros(x.shape)

    for idx in range(len(x)):
        dx = zeros(x.shape)
        if relative:
            hx = x[idx] * h
        else:
            hx = h
        dx[idx] = hx

        f1 = func(x + dx, *args)
        f2 = func(x - dx, *args)
        result[idx] = 0.5 * (f1 - f2) / hx

    return result


def hessian(func, x, h=1e-5, relative=False, *args):
    """"""
    len_x = len(x)
    result = zeros((len_x, len_x))

    for row in range(1, len_x):
        for col in range(row):
            dx = zeros(len_x)
            dy = zeros(len_x)
            if relative:
                hx = h * x[col]
                hy = h * x[row]
            else:
                hx = hy = h
            dx[col] = hx
            dy[row] = hy

            f1 = func(x + dx + dy, *args)
            f2 = func(x + dx - dy, *args)
            f3 = func(x - dx + dy, *args)
            f4 = func(x - dx - dy, *args)
            result[row, col] = 0.25 * (f1 - f2 - f3 + f4) / hx / hy

            result += result.T

            for idx in range(len_x):
                dx = zeros(len_x)
                if relative:
                    hx = h * x[idx]
                else:
                    hx = h
                dx[idx] = hx

                f1 = func(x + dx, *args)
                f2 = func(x, *args)
                f3 = func(x - dx, *args)

                result[idx, idx] = (f1 - 2 * f2 + f3) / hx / hx

    return result


def akaike_information_criterion(log_likelihood, dimension):
    """"""
    result = 2 * dimension - 2 * log_likelihood
    return result


def bayesian_information_criterion(log_likelihood, dimension, n):
    """"""
    result = dimension * log(n) - 2 * log_likelihood
    return result


def correlation_matrix(anglar_coordenants):
    """"""
    len_coords = len(anglar_coordenants)
    n = int((1 + sqrt(1 + 4 * len_coords)) / 2)

    theta = reshape(anglar_coordenants, (n, n - 1))
    print(theta, end="\n\n")
    B = zeros((n, n))

    for i in range(n):
        for j in range(n):
            if j < n - 1:
                B[i, j] = cos(theta[i, j]) * prod([sin(x) for x in theta[i, :j]])
            else:
                B[i, j] = prod([sin(x) for x in theta[i, :j]])

    print(B, end="\n\n")
    C = dot(B, B.T)
    return C


def correlation_to_variance_matrix(correlation_matrix, variances):
    """"""
    std_deviations = sqrt(variances)
    V = diag(std_deviations)

    result = dot(dot(V, correlation_matrix), V)
    return result


if __name__ == "__main__":
    pass
