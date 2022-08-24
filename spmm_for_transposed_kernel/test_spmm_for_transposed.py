import torch
import spmm_utils
import time
import numpy as np
from torch_sparse import SparseTensor
from torch_sparse import matmul, matmul_with_cached_transposed
from torch_sparse import spmm
from torch_sparse import transpose
import scipy.sparse as sci_sp

def check_error(res_ref, res, rows, cols, error):
    total_diff = 0.0
    for i in range(rows):
        for j in range(cols):
            diff = abs(res_ref[i][j] - res[i][j])
            if diff > error:
                print('row = {}, col = {}, ref = {}, res = {}, err = {}'.format(i, j, res_ref[i][j], res[i][j], diff))
                total_diff += diff

    return total_diff

def generate_sparse_tensor(row, col, density, format):
    sci_sparse_mat = sci_sp.rand(row, col, density=density, format=format, dtype=np.float32)
    return SparseTensor.from_scipy(sci_sparse_mat) 


sparse_rows = 50
sparse_cols = 50

dense_rows = 50
dense_cols = 16

# dense_mat = torch.arange(dense_rows * dense_cols).reshape(dense_rows, dense_cols).float()

row = torch.tensor([0, 1, 2, 3, 4, 4, 1, 6, 2, 3, 5])
col = torch.tensor([0, 1, 0, 1, 3, 5, 7, 1, 6, 5, 3])
value = torch.tensor([1, 2, 5, 3, 9, 6, 5, 10, 2, 3, 2], dtype=torch.float)
index = torch.tensor([[0, 1, 2, 3, 4, 4, 1, 6, 2, 3, 5],
                      [0, 1, 0, 1, 3, 5, 7, 1, 6, 5, 3]])

def test_spmm(rowptr, col, value, dense_mat, sparse_rows):
    my_res = spmm_utils.spmm_cpu_optimized(rowptr, col, value, dense_mat, sparse_rows, 1, "sum")[0]
    res_ref = matmul(sp, dense_mat)
    print(my_res)
    print("total_diff: ", check_error(res_ref, my_res, sparse_rows, dense_cols, 10e-5))

def print_csr(rowptr, col, value):
    print("rowptr")
    print(rowptr)
    print("col")
    print(col)
    print("value")
    print(value)
    
def test_spmm_performance(sparse_rows, sparse_cols, dense_rows, dense_cols, density, format):
    if sparse_rows * sparse_cols * density < 1:
        print("density is too small.")
        return 
    sparse_mat = generate_sparse_tensor(sparse_rows, sparse_cols, density, format)
    # dense_mat = torch.arange(dense_rows * dense_cols).reshape(dense_rows, dense_cols).float()
    dense_mat = torch.rand(dense_rows, dense_cols)
    print(dense_mat)
    # print(sparse_mat.to_dense())
    rowptr, col, value = sparse_mat.csr()
    print_csr(rowptr, col, value)
    
    # torch_sparse's spmm result
    res_ref = matmul(sparse_mat, dense_mat)
    print("torch_sparse.matmul() over.")

    # no tile version spmm result
    res_no_tile = matmul_with_cached_transposed(sparse_mat, dense_mat) 
    print("my_method.matmul_no_tile() over.")

    # tile version spmm result
    tile_num = 8
    res_tile = spmm_utils.spmm_cpu_optimized(rowptr, col, value, dense_mat, sparse_rows, tile_num, "sum")[0]
    print("my_method.matmul_with_tile_num={} over.".format(str(tile_num)))

    # check the error 
    no_tile_error = check_error(res_ref, res_no_tile, sparse_rows, dense_cols, 10e-5)
    print("total_diff of no tile version: " + str(no_tile_error))
    tile_error = check_error(res_ref, res_tile, sparse_rows, dense_cols, 10e-5)
    print("total_diff of tile version: " + str(tile_error))


# sp = SparseTensor(row=row, col=col, value=value)
# rowptr, col, value = sp.csr()

'''
csr2csc = sp.storage.csr2csc()
# opt_value = value.view(-1, 1).index_select(0, csr2csc).view(-1)
opt_value = sp.storage.value().view(-1, 1).index_select(0, csr2csc).view(-1)
colptr = sp.storage.colptr()
new_sp = SparseTensor(rowptr = colptr, col = row.index_select(0, csr2csc), value=opt_value)
print("new_sp:")
print(new_sp.to_dense())
res_ref = matmul(new_sp, dense_mat)

index[0] = sp.storage.row()
index[1] = sp.storage.col()
value = sp.storage.value()
index_trans, value_trans = transpose(index, value, sparse_rows, sparse_cols)
transposed_sp = SparseTensor(row = index_trans[0], col = index_trans[1], value = value_trans)
print("transposed_sp:")
print(transposed_sp.to_dense())
res_transposed = spmm(index_trans, value_trans, sparse_rows, sparse_cols, dense_mat)
colptr, row, value = transposed_sp.csc()

print("transposed colptr:")
print(colptr)
print("transposed row:")
print(row)
print("transposed value:")
print(value)
'''

# print(dense_mat)
# res = spmm_utils.spmm_for_transposed(rowptr, col, value, dense_mat, sparse_rows, "sum")[0]

'''
print("res_ref:")
print(res_ref)
print("res_transposed:")
print(res_transposed)
print("res:")
print(res)
print("total_diff: ", check_error(res_ref, res, sparse_rows, dense_cols, 10e-5))
'''

if __name__ == '__main__':
    # rowptr, col, value = sp.csr()
    # test_spmm(rowptr, col, value, dense_mat, sparse_rows)
    density = 0.5
    sparse_format = 'csr'
    sparse_rows = 1000
    sparse_cols = 1000
    dense_rows = sparse_cols
    dense_cols = 16

    test_spmm_performance(sparse_rows, sparse_cols, dense_rows, dense_cols, density, sparse_format)

