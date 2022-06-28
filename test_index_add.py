import torch
import example_cpp
import time

row = 2708
col1 = 10
col0 = 100
src_row = 13264


def test_torch_index_add_time(output, source, index, dim, alpha, repeat):
    output.index_add_(dim, index, source, alpha=1.0)
    start = time.perf_counter()
    for _ in range(repeat):
        output.index_add_(0, index, source, alpha=1.0)
    end = time.perf_counter()
    print("torch_index_add_ thread numbers: {}, repeat: {}, total time(s): {}".format(torch.get_num_threads(), repeat, (end - start)))

def test_my_index_add_time(output, source, index, dim, alpha, repeat):
    example_cpp.index_add(output, dim, index, source, alpha)
    start = time.perf_counter()
    for _ in range(repeat):
        example_cpp.index_add(output, dim, index, source, alpha)
    end = time.perf_counter()
    print("my_index_add_ thread numbers: {}, repeat: {}, total time(s): {}".format(torch.get_num_threads(), repeat, (end - start)))

def test_diff(source, index, dim, alpha):
    # output_ref = torch.ones(row, col)
    output_ref = torch.ones(row, col0, col1)
    # output = torch.ones(row, col)
    output = torch.ones(row, col0, col1)

    output_ref.index_add_(dim, index, source, alpha = alpha)
    example_cpp.index_add(output, dim, index, source, alpha)

    error = (output_ref - output).abs().max()
    print("difference: {:.6f}".format(error))
    print(output_ref)
    print(output)

    '''
    diff = 0.0
    for i in range(row):
        for j in range(col):
            tmp = abs(output_ref[i][j] - output[i][j])
            if tmp > 0.0001:
                diff += tmp
    
    print("difference: {:.5f}".format(diff))
    '''

# output_ref = torch.ones(row, col)
output_ref = torch.ones(row, col0, col1)
# output = torch.ones(row, col)
output = torch.ones(row, col0, col1)
# source = torch.rand(src_row, col) * 5 
source = torch.rand(src_row, col0, col1) * 5 

import numpy as np
tmp = np.random.randint(row, size=src_row)
index = torch.from_numpy(tmp)
print(index)

# output = torch.zeros(20, 3)
# print(output)
dim = 0
# index = torch.tensor([1, 2, 2], dtype=torch.long)
# source = torch.arange(1., 10.).reshape(3, 3)
# print(source)
alpha = 1
repeat = 100
test_torch_index_add_time(output_ref, source, index, dim, alpha, repeat)
print("-----------------------------")
test_my_index_add_time(output, source, index, dim, alpha, repeat)

test_diff(source, index, dim, alpha)

# example_cpp.index_add(output, dim, index, source, alpha)
# print(output)

# example_cpp.index_add(output)
