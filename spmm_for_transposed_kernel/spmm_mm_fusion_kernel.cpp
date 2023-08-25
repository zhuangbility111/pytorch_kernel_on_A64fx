#include <iostream>

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu_optimized(torch::Tensor rowptr, torch::Tensor col,
                   torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
				   int64_t sparse_rows, int64_t tile_num, std::string reduce) {
}
