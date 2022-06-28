import torch
import example_cpp

source = torch.arange(24, dtype=torch.float).reshape(4,3,2)
output = torch.ones(4,3,2)
index = torch.tensor([0,3,2,1])
dim = 0
alpha = 1.0
example_cpp.index_add(output, dim, index, source, alpha)
print(output)
