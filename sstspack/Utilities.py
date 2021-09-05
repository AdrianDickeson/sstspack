from numpy import zeros

# scipy implementation has issues with different sized matrices
def block_diag(matrix_list):
    """"""
    row_length = sum(mat.shape[0] for mat in matrix_list)
    col_length = sum(mat.shape[1] for mat in matrix_list)

    result = zeros((row_length, col_length))

    row_idx = col_idx = 0
    for mat in matrix_list:
        row_len = mat.shape[0]
        col_len = mat.shape[1]
        result[row_idx : (row_idx + row_len), col_idx : (col_idx + col_len)] = mat[:, :]
        row_idx = row_idx + row_len
        col_idx = col_idx + col_len

    return result


def jacobian(func, x, h=1e-6, relative=False, *args):
    """"""
    y = func(x)
    result = zeros((len(y), len(x)))

    for row in range(len(y)):

        def row_func(x):
            return func(x)[row]

        for idx in range(len(x)):
            dx = zeros(x.shape)
            hx = x[idx] * h if relative else h
            dx[idx] = hx

            f1 = row_func(x + dx, *args)
            f2 = row_func(x - dx, *args)
            result[row, idx] = 0.5 * (f1 - f2) / hx

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
                hx = h * x[idx] if relative else h
                dx[idx] = hx

                f1 = func(x + dx, *args)
                f2 = func(x, *args)
                f3 = func(x - dx, *args)

                result[idx, idx] = (f1 - 2 * f2 + f3) / hx / hx

    return result


def identity_fn(x):
    """"""
    return x
