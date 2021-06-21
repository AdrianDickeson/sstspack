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
