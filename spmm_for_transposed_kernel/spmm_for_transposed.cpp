#include <torch/extension.h>
#include <iostream>
#include <utility>
#include <vector>
#include <chrono> 
#include <stdio.h>

// #include "../../pytorch_sparse-0.6.13/csrc/cpu/spmm_cpu.h"
// #include "../../pytorch_sparse-0.6.13/csrc/cpu/reducer.h"
#include "../../pytorch_sparse-0.6.13/csrc/cpu/utils.h"

using namespace std::chrono;

#define _OPENMP
#include <ATen/ParallelOpenMP.h>

#include <omp.h>

#ifdef __ARM_FEATURE_SVE
	#include <arm_sve.h>
	#define VEC_LEN 16
#endif /* __ARM_FEATURE_SVE */


inline int32_t divup(int32_t x, int32_t y) {
	return (x + y - 1) / y;
}

void divide_work(int* work_range, int total_work, int num_threads) {
	int chunk_size;
	int remain_work = total_work;
	work_range[0] = 0;
	for (int i = 0; i < num_threads; i++) {
		chunk_size = divup(remain_work, num_threads - i);
		work_range[i+1] = work_range[i] + chunk_size;
		remain_work -= chunk_size;
	}
	work_range[num_threads] = total_work;
}

int obtain_tile_row_array(int64_t* row_ptr, int64_t* col, float* values,
						  int row_ptr_start, int row_ptr_end,
						  int* tile_row_ptr, int tile_num, int tile_size) {
	// std::cout << "tile_num: " << tile_num << ", tile_size: " << tile_size << std::endl;
	int tile_row_ptr_id = 0;
	int cur_tile_id = 0;
	tile_row_ptr[0] = row_ptr[row_ptr_start];
	for (int i = row_ptr_start; i < row_ptr_end; i++) {
		int col_id_start = row_ptr[i], col_id_end = row_ptr[i+1];
		int cur_col_id = col_id_start;
		int cur_col = col[cur_col_id];
		for (int cur_tile_id = 0; cur_tile_id < tile_num; cur_tile_id++) {
			++tile_row_ptr_id;
			tile_row_ptr[tile_row_ptr_id] = tile_row_ptr[tile_row_ptr_id - 1];
			while (cur_col_id < col_id_end && cur_col / tile_size == cur_tile_id) {
				// std::cout << "row: " << i << ", " << "cur_col: " << cur_col << ", tile_id: " << cur_col / tile_size << ", cur_tile_id: " << cur_tile_id << std::endl;
				tile_row_ptr[tile_row_ptr_id]++;
				++cur_col_id;
				cur_col = col[cur_col_id];
			}
		}
	}
	return 1;
}

void count_nnz_per_group(int64_t* row_ptr, int64_t* col, float* value,
						 int row_ptr_begin, int row_ptr_end, 
						 int* nnz_per_group, int num_group, int size_group) {
	// scan whole sparse matrix to count nnz per group
	for (int i = row_ptr_begin; i < row_ptr_end; i++) {
		int col_idx_begin = row_ptr[i], col_idx_end = row_ptr[i+1];
		for (int j = col_idx_begin; j < col_idx_end; j++) {
			int cur_col = col[j];
			nnz_per_group[cur_col / size_group]++;
		}
	}
}

int obtain_group_csr(int64_t* row_ptr, int64_t* col, float* value, 
					 int row_ptr_begin, int row_ptr_end, int nnz,
					 int num_group, int size_group,
					 int* nnz_per_group, int* group_row_ptr, int* group_col, int* group_value) {
	// count nnz per group
	count_nnz_per_group(row_ptr, col, value, row_ptr_begin, row_ptr_end, 
						nnz_per_group, num_group, size_group);

	int num_row = row_ptr_end - row_ptr_begin;
	int idx_in_group[num_group];
	group_row_ptr[0] = 0;

	// initialize group_row_ptr and idx in each group
	for (int i = 0; i < num_group; i++) {
		idx_in_group[i] = nnz_per_group[i];
		group_row_ptr[(i+1) * (num_row+1)] = nnz_per_group[i];
	}

	// fill value into group_row_ptr, group_col, group_value
	for (int i = row_ptr_begin; i < row_ptr_end; i++) {
		int col_idx_begin = row_ptr[i], col_idx_end = row_ptr[i+1];
		int idx_in_col = col_idx_begin;
		int cur_col = col[idx_in_col];
		float cur_value = value[idx_in_col];
		for (int g = 0; g < num_group; g++) {
			// obtain the current group's idx on new group_col and new group_value
			int idx_in_group_col = idx_in_group[g];
			int idx_in_group_row_ptr = g * (num_row+1) + i + 1;
			group_row_ptr[idx_in_group_row_ptr] = group_row_ptr[idx_in_group_row_ptr - 1];
			while (idx_in_col < col_idx_end && cur_col / size_group == g) {
				// update new group_row_ptr, group_col and group_value
				group_row_ptr[idx_in_group_row_ptr]++;
				group_col[idx_in_group_col] = cur_col;
				group_value[idx_in_group_col] = cur_value;
				// update the current group's idx on new group_col and new group_value
				++idx_in_col;
				++idx_in_group_col;
				// obtain col's value and value in global array
				cur_col = col[idx_in_col];
				cur_value = value[idx_in_col];
			}
		}
	}
	return 1;
}

/*
void obtain_tile_rowptr(torch::Tensor rowptr, torch::Tensor col, 
						torch::optional<torch::Tensor> optional_value,
						int64_t sparse_rows, int64_t sparse_cols) {
	auto rowptr_data = rowptr.data_ptr<int64_t>();
	auto col_data = col.data_ptr<int64_t>();
	float* value_data = nullptr;

	const bool HAS_VALUE = optional_value.has_value();

	if (HAS_VALUE)
		value_data = optional_value.value().data_ptr<float>();

	int64_t tile_num = 2;
	int64_t tile_size = sparse_cols / tile_num;
	int64_t tile_rowptr_size = sparse_rows * tile_num + 1;
	int64_t tile_rowptr[tile_rowptr_size];
	obtain_tile_row_array(rowptr_data, col_data, value_data,
						  0, sparse_rows, 
						  tile_rowptr, tile_num, tile_size);

	for (int i = 0; i < tile_rowptr_size; i++) {
		std::cout << tile_rowptr[i]	<< ", ";
	}
	std::cout << std::endl;

	for (int i = 0; i < tile_rowptr_size - 1; i++) {
		int col_start = tile_rowptr[i], col_end = tile_rowptr[i+1];
		std::cout << "tile " << i << ": ";
		for (int j = col_start; j < col_end; j++) {
			std::cout << value_data[j] << ", ";
		}
		std::cout << std::endl;
	}
}
*/

void check_input(torch::Tensor rowptr, torch::Tensor col,
				 torch::optional<torch::Tensor> optional_value, torch::Tensor mat) {
	// check sparse matrix ptr
	CHECK_CPU(rowptr);
	CHECK_CPU(col);
	if (optional_value.has_value())
		CHECK_CPU(optional_value.value());
	CHECK_CPU(mat);
	
	// check shape of sparse matrix 
	CHECK_INPUT(rowptr.dim() == 1);
	CHECK_INPUT(col.dim() == 1);

	if (optional_value.has_value()) {
		CHECK_INPUT(optional_value.value().dim() == 1);
		CHECK_INPUT(optional_value.value().size(0) == col.size(0));
	}
	CHECK_INPUT(mat.dim() == 2);
}

void obtain_MNK(torch::Tensor mat, int64_t sparse_rows,
                int &M, int &N, int &K) {
	// sparse matrix shape: M * K
	// dense matrix shape: B * K * N
	int64_t dense_rows = mat.size(-2);
	int64_t dense_cols = mat.size(-1);
	int64_t dense_batch_size = mat.numel() / (dense_rows * dense_cols);
	int64_t sparse_cols = dense_rows;

	M = static_cast<int>(sparse_rows);
	K = static_cast<int>(sparse_cols);
	N = static_cast<int>(dense_cols);
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu_optimized(torch::Tensor rowptr, torch::Tensor col, 
				   torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
				   int64_t sparse_rows, int64_t tile_num, std::string reduce) {

	// check input
	check_input(rowptr, col, optional_value, mat);

	mat = mat.contiguous();
	// allocate output memory
	auto sizes = mat.sizes().vec();
	sizes[mat.dim() - 2] = sparse_rows;
	auto out = torch::zeros(sizes, mat.options());

	// obtain MNK
	int M, N, K;
	obtain_MNK(mat, sparse_rows, M, N, K);

	if (mat.scalar_type() == at::ScalarType::Float && 
		optional_value.value().scalar_type() == at::ScalarType::Float) {

		// obtain data pointer
		int64_t* rowptr_data = rowptr.data_ptr<int64_t>();
		int64_t* col_data = col.data_ptr<int64_t>();
		float* value_data = optional_value.value().data_ptr<float>();
		float* mat_data = mat.data_ptr<float>();
		float* out_data = out.data_ptr<float>();
		const bool HAS_VALUE = optional_value.has_value();

		auto tile_start_time = system_clock::now();
		// int64_t tile_num = 2;
		int64_t tile_size = divup(K, tile_num);
		int64_t tile_rowptr_size = M * tile_num + 1;
		int tile_rowptr[tile_rowptr_size];
		obtain_tile_row_array(rowptr_data, col_data, value_data,
						      0, M,
						      tile_rowptr, tile_num, tile_size);

		auto tile_end_time = system_clock::now();
		duration<double, std::milli> tile_elapsed_time = tile_end_time - tile_start_time;
		std::cout << "tile rowptr time: " << tile_elapsed_time.count() << "ms" << std::endl;

		// int64_t batch_times_rows = dense_batch_size * sparse_rows;
		int max_num_threads = omp_get_max_threads();
		// std::cout << "max_num_threads = " << max_num_threads << std::endl;
		int num_threads_on_vertexs = max_num_threads;
		int num_threads_on_features = 1;

		int work_range_on_vertexs[num_threads_on_vertexs + 1];
		int work_range_on_features[num_threads_on_features + 1];

		double elapsed_time_array[num_threads_on_vertexs];
		
		auto start_time_1 = system_clock::now();
		// divide work
		divide_work(work_range_on_vertexs, M, num_threads_on_vertexs);
		divide_work(work_range_on_features, N, num_threads_on_features);
		duration<double, std::milli> diff = (system_clock::now() - start_time_1);
		std::cout << "elapsed time of dividing work: " << diff.count() << std::endl;

		auto start_time = system_clock::now();
		#pragma omp parallel 
		{
			// auto start_time = system_clock::now();
			int tid = omp_get_thread_num();
			int tid_on_vertexs = tid / num_threads_on_features;
			int tid_on_features = tid % num_threads_on_features;

			int start_on_M = work_range_on_vertexs[tid_on_vertexs];
			int end_on_M = work_range_on_vertexs[tid_on_vertexs + 1];

			int start_on_N = work_range_on_features[tid_on_features];
			int end_on_N = work_range_on_features[tid_on_features + 1];
/*
			std::cout << "tid = " << tid << " tid_on_vertexs = " << tid_on_vertexs
										 << " tid_on_features = " << tid_on_features
										 << " start_on_v = " << work_range_on_vertexs[0]
										 << " end_on_v = " << work_range_on_vertexs[1]
										 << " start_on_f = " << work_range_on_features[0]
										 << " end_on_f = " << work_range_on_features[1]
										 << std::endl;
*/
			for (int tile_id = 0; tile_id < tile_num; tile_id++) {
				for (int m = start_on_M; m < end_on_M; m++) {
					int idx_on_tile_rowptr = tile_id + m * tile_num;
					int start_on_cols = tile_rowptr[idx_on_tile_rowptr];
					int end_on_cols = tile_rowptr[idx_on_tile_rowptr+1];
					// std::cout << "idx_on_tile_rowptr = " << idx_on_tile_rowptr << std::endl;
					for (int n = start_on_N; n < end_on_N; n += VEC_LEN) {
						svbool_t pg = svwhilelt_b32(n, end_on_N);
						svfloat32_t vout = svld1(pg, &(out_data[m*N + n]));
						for (int id_on_cols = start_on_cols; id_on_cols < end_on_cols; id_on_cols++) {
							int k = col_data[id_on_cols];
							svfloat32_t va = svdup_n_f32(value_data[id_on_cols]);
							svfloat32_t vb = svld1(pg, &(mat_data[k*N + n]));
							vout = svmla_f32_x(pg, vout, va, vb);
						}
						svst1(pg, &(out_data[m*N + n]), vout);
					}
				}
			}
		/*	
			duration<double, std::milli> diff1 = (system_clock::now() - start_time);
			elapsed_time_array[tid] = diff1.count();
		*/
		}
		duration<double, std::milli> diff1 = (system_clock::now() - start_time);
		std::cout << "elapsed time of tile method's kernel = " << diff1.count() << "ms" << std::endl;

/*
		for (int i = 0; i < num_threads_on_vertexs; i++) {
			std::cout << "elapsed time of thread " << i << ": " << elapsed_time_array[i] << std::endl;	
		}
*/
	}
	else {
		std::cout << "the data type of one input matrix is not float" 
				  << ", float type: " << at::ScalarType::Float
				  << ", sparse_data_type: " << optional_value.value().scalar_type()
				  << ", dense_data_type: " << mat.scalar_type()
				  << std::endl;
	}

	torch::optional<torch::Tensor> arg_out = torch::nullopt;
	return std::make_tuple(out, arg_out);
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu_optimized_with_group_csr(torch::Tensor rowptr, torch::Tensor col, 
				   				  torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
				   				  int64_t sparse_rows, int64_t tile_num, std::string reduce) {

	// check input
	check_input(rowptr, col, optional_value, mat);

	mat = mat.contiguous();
	// allocate output memory
	auto sizes = mat.sizes().vec();
	sizes[mat.dim() - 2] = sparse_rows;
	auto out = torch::zeros(sizes, mat.options());

	// obtain MNK
	int M, N, K;
	obtain_MNK(mat, sparse_rows, M, N, K);

	if (mat.scalar_type() == at::ScalarType::Float && 
		optional_value.value().scalar_type() == at::ScalarType::Float) {

		// obtain data pointer
		int64_t* rowptr_data = rowptr.data_ptr<int64_t>();
		int64_t* col_data = col.data_ptr<int64_t>();
		float* value_data = optional_value.value().data_ptr<float>();
		float* mat_data = mat.data_ptr<float>();
		float* out_data = out.data_ptr<float>();
		const bool HAS_VALUE = optional_value.has_value();

		// auto tile_start_time = system_clock::now();
		int64_t num_group = tile_num;
		int64_t size_group = divup(K, tile_num);
		int64_t nnz = col.numel();

		// allocate the rowptr, col, value array for every group with nnz_per_group
		int* nnz_per_group = (int*)malloc(num_group * sizeof(int));
		int* group_row_ptr = (int*)malloc((M+1) * num_group * sizeof(int));
		int* group_col     = (int*)malloc(nnz * sizeof(int));
		int* group_value   = (int*)malloc(nnz * sizeof(int));
		memset(nnz_per_group, 0, sizeof(int) * num_group);

		obtain_group_csr(rowptr_data, col_data, value_data,
						 0, M, nnz, num_group, size_group,
						 nnz_per_group, group_row_ptr, group_col, group_value);

/*
		auto tile_end_time = system_clock::now();
		duration<double, std::milli> tile_elapsed_time = tile_end_time - tile_start_time;
		std::cout << "tile rowptr time: " << tile_elapsed_time.count() << "ms" << std::endl;

		// int64_t batch_times_rows = dense_batch_size * sparse_rows;
		int max_num_threads = omp_get_max_threads();
		// std::cout << "max_num_threads = " << max_num_threads << std::endl;
		int num_threads_on_vertexs = max_num_threads;
		int num_threads_on_features = 1;

		int work_range_on_vertexs[num_threads_on_vertexs + 1];
		int work_range_on_features[num_threads_on_features + 1];

		double elapsed_time_array[num_threads_on_vertexs];
		
		auto start_time_1 = system_clock::now();
		// divide work
		divide_work(work_range_on_vertexs, M, num_threads_on_vertexs);
		divide_work(work_range_on_features, N, num_threads_on_features);
		duration<double, std::milli> diff = (system_clock::now() - start_time_1);
		std::cout << "elapsed time of dividing work: " << diff.count() << std::endl;

		auto start_time = system_clock::now();
		#pragma omp parallel 
		{
			// auto start_time = system_clock::now();
			int tid = omp_get_thread_num();
			int tid_on_vertexs = tid / num_threads_on_features;
			int tid_on_features = tid % num_threads_on_features;

			int start_on_M = work_range_on_vertexs[tid_on_vertexs];
			int end_on_M = work_range_on_vertexs[tid_on_vertexs + 1];

			int start_on_N = work_range_on_features[tid_on_features];
			int end_on_N = work_range_on_features[tid_on_features + 1];
			*/
/*
			std::cout << "tid = " << tid << " tid_on_vertexs = " << tid_on_vertexs
										 << " tid_on_features = " << tid_on_features
										 << " start_on_v = " << work_range_on_vertexs[0]
										 << " end_on_v = " << work_range_on_vertexs[1]
										 << " start_on_f = " << work_range_on_features[0]
										 << " end_on_f = " << work_range_on_features[1]
										 << std::endl;
*/
/*
			for (int tile_id = 0; tile_id < tile_num; tile_id++) {
				for (int m = start_on_M; m < end_on_M; m++) {
					int idx_on_tile_rowptr = tile_id + m * tile_num;
					int start_on_cols = tile_rowptr[idx_on_tile_rowptr];
					int end_on_cols = tile_rowptr[idx_on_tile_rowptr+1];
					// std::cout << "idx_on_tile_rowptr = " << idx_on_tile_rowptr << std::endl;
					for (int n = start_on_N; n < end_on_N; n += VEC_LEN) {
						svbool_t pg = svwhilelt_b32(n, end_on_N);
						svfloat32_t vout = svld1(pg, &(out_data[m*N + n]));
						for (int id_on_cols = start_on_cols; id_on_cols < end_on_cols; id_on_cols++) {
							int k = col_data[id_on_cols];
							svfloat32_t va = svdup_n_f32(value_data[id_on_cols]);
							svfloat32_t vb = svld1(pg, &(mat_data[k*N + n]));
							vout = svmla_f32_x(pg, vout, va, vb);
						}
						svst1(pg, &(out_data[m*N + n]), vout);
					}
				}
			}
			*/
		/*	
			duration<double, std::milli> diff1 = (system_clock::now() - start_time);
			elapsed_time_array[tid] = diff1.count();
		}
		*/
		// duration<double, std::milli> diff1 = (system_clock::now() - start_time);
		// std::cout << "elapsed time of tile method's kernel = " << diff1.count() << "ms" << std::endl;

/*
		for (int i = 0; i < num_threads_on_vertexs; i++) {
			std::cout << "elapsed time of thread " << i << ": " << elapsed_time_array[i] << std::endl;	
		}
*/
	}
	else {
		std::cout << "the data type of one input matrix is not float" 
				  << ", float type: " << at::ScalarType::Float
				  << ", sparse_data_type: " << optional_value.value().scalar_type()
				  << ", dense_data_type: " << mat.scalar_type()
				  << std::endl;
	}

	torch::optional<torch::Tensor> arg_out = torch::nullopt;
	return std::make_tuple(out, arg_out);
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu_for_transposed_sparse(torch::Tensor rowptr, torch::Tensor col,
								torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
								int64_t sparse_rows, std::string reduce) {
	// check sparse matrix ptr
	CHECK_CPU(rowptr);
	CHECK_CPU(col);
	if (optional_value.has_value())
		CHECK_CPU(optional_value.value());
	CHECK_CPU(mat);
	
	// check shape of sparse matrix 
	CHECK_INPUT(rowptr.dim() == 1);
	CHECK_INPUT(col.dim() == 1);

	if (optional_value.has_value()) {
		CHECK_INPUT(optional_value.value().dim() == 1);
		CHECK_INPUT(optional_value.value().size(0) == col.size(0));
	}
	CHECK_INPUT(mat.dim() == 2);

	mat = mat.contiguous();

	// allocate output memory
	auto sizes = mat.sizes().vec();
	sizes[mat.dim() - 2] = sparse_rows;
	auto out = torch::zeros(sizes, mat.options());

	auto rowptr_data = rowptr.data_ptr<int64_t>();
	auto col_data = col.data_ptr<int64_t>();

	// sparse matrix shape: M * K
	// dense matrix shape: B * K * N
	int64_t sparse_cols = rowptr.numel() - 1;
	int64_t dense_rows = mat.size(-2);
	int64_t dense_cols = mat.size(-1);
	int64_t dense_batch_size = mat.numel() / (dense_rows * dense_cols);

	int64_t K = sparse_cols;
	int64_t N = dense_cols;

	if (mat.scalar_type() == at::ScalarType::Float && 
		optional_value.value().scalar_type() == at::ScalarType::Float) {
		float* value_data = nullptr;
		float* mat_data = mat.data_ptr<float>();
		float* out_data = out.data_ptr<float>();

		const bool HAS_VALUE = optional_value.has_value();

		if (HAS_VALUE)
			value_data = optional_value.value().data_ptr<float>();
	

/*
		int64_t batch_times_rows = dense_batch_size * sparse_rows;
		for (int64_t k = 0; k < K; k++) {
			int64_t row_start = rowptr_data[k], row_end = rowptr_data[k + 1];

			for (int64_t idx_in_col_data = row_start; idx_in_col_data < row_end; idx_in_col_data++) {
				int64_t m = col_data[idx_in_col_data];
				float value = value_data[idx_in_col_data];

				for (int64_t n = 0; n < N; n++) {
					out_data[m*N + n] += value * mat_data[k*N + n];
				}
			}
		}
		*/
	}
	else {
		std::cout << "the data type of one input matrix is not float." << std::endl;
	}

	torch::optional<torch::Tensor> arg_out = torch::nullopt;
	return std::make_tuple(out, arg_out);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    // m.def("index_add", &index_add, "A function that adds two numbers");
    m.def("spmm_for_transposed", &spmm_cpu_for_transposed_sparse, "A function that adds two numbers");
	// m.def("obtain_tile_rowptr", &obtain_tile_rowptr, "");
    m.def("spmm_cpu_optimized", &spmm_cpu_optimized, "A function that adds two numbers");
}
