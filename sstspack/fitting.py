'''
Created on 4 Sep 2020

@author: adriandickeson
'''

from numpy import exp, sqrt, isinf, log, array, zeros
from scipy.optimize import minimize

from sstspack.StateSpaceModelClass import StateSpaceModel as SSM

def parameter_transform_function(parameter_bounds):
    '''
    '''
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
        result = half_range * x / sqrt(1+x*x)
        return result + (lower_bound + half_range)

    if isinf(lower_bound) and isinf(upper_bound):
        return unconstrained
    if isinf(upper_bound):
        return constrained_upper_half_interval
    if isinf(lower_bound):
        return constrained_lower_half_interval
    return constrained_closed_interval

def inverse_parameter_transform_function(parameter_bounds):
    '''
    '''
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
        return '\n'.join(['{}:\t{}'.format(key, self.__dict__[key]) for key in self.__dict__])

def fit_model_max_likelihood(params0, params_bounds, model_func, y_series,
                             a0, P0, diffuse_state):
    '''
    '''
    initial_params = [inverse_parameter_transform_function(params_bounds[idx])(value)
                      for idx, value in enumerate(params0)]
    param_funcs = [parameter_transform_function(bounds) for bounds in params_bounds]

    def objective_func(transformed_params, y_series):
        params = [parameter_transform_function(params_bounds[idx])(value)
                  for idx, value in enumerate(transformed_params)]
        return inner_objective_func(params, y_series)

    def inner_objective_func(params, y_series):
        model_data = model_func(params)
        model = SSM(y_series, model_data, a0, P0, diffuse_state)
        return -model.log_likelihood()

    res = minimize(objective_func, initial_params, options={'disp': False},
                   args=(y_series,), method='BFGS', tol=1.e-16)
    result = FittedModel()
    result.count_iteration = res.nit 
    result.count_function_evaluations = res.nfev
    result.count_gradient_evaluations = res.njev

    new_initial_params = array([parameter_transform_function(params_bounds[idx])(value)
                               for idx, value in enumerate(res.x)])
    res = minimize(inner_objective_func, new_initial_params,
                   options={'disp': False}, args=(y_series,),
                   method='BFGS', tol=1.e-16)

    result.parameters = res.x
    result.log_likelihood = -res.fun
    result.count_iteration += res.nit 
    result.count_function_evaluations += res.nfev
    result.count_gradient_evaluations += res.njev
    result.message = res.message
    result.status = res.status
    result.success = res.success
    result.jacobian = -res.jac
    result.inverse_fisher_information_matrix = res.hess_inv

    return result

def jacobian(func, x, * args):
    '''
    '''
    h = 0.0001
    result = zeros(x.shape)

    for idx in range(len(x)):
        dx = zeros(x.shape)
        dx[idx] = h 
        f1 = func(x + dx, *args)
        f2 = func(x - dx, *args)
        result[idx] = 0.5 * (f1 - f2)/h

    return result


if __name__ == '__main__':
    pass
