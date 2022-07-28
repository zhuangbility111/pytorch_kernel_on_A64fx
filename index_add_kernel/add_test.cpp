#include <torch/extension.h>
#include <iostream>
#include <utility>
#include <vector>
#include <chrono> 
#include <stdio.h>

using namespace std::chrono;

/*
#define _OPENMP
#include <ATen/ParallelOpenMP.h>
*/

#include <omp.h>

#ifdef __ARM_FEATURE_SVE
	#include <arm_sve.h>
	#define VEC_LEN 16
#endif /* __ARM_FEATURE_SVE */


inline int32_t divup(int32_t x, int32_t y) {
	return (x + y - 1) / y;
}

template<typename T>
void add_slice_with_stride1(T* self, T* source, const int64_t self_stride, const int64_t source_stride, const int64_t len) {
	
}

template<>
void add_slice_with_stride1<float>(float* self, float* source, const int64_t self_stride, const int64_t source_stride, const int64_t len) {
/*
	 at::parallel_for(0, len, 0, [&](int64_t start, int64_t end) {
#ifdef __ARM_FEATURE_SVE
        for (int64_t i = start; i < end; i += svcntw()) {
			svbool_t pg = svwhilelt_b32(i, end);
			svfloat32_t va = svld1(pg, &(self[i]));
			svfloat32_t vb = svld1(pg, &(source[i]));
			svst1(pg, &(self[i]), svadd_f32_x(pg, va, vb));
        }
#endif 
//        std::cout << "thread number = " << at::get_num_threads() << std::endl;
//        std::cout << "hi there from " << at::get_thread_num() << std::endl;

    });
*/
	#pragma omp parallel for
	for (int64_t i = 0; i < len; i += VEC_LEN) {
		svbool_t pg = svwhilelt_b32(i, len);
        svfloat32_t va = svld1(pg, &(self[i]));
        svfloat32_t vb = svld1(pg, &(source[i]));
        svst1(pg, &(self[i]), svadd_f32_x(pg, va, vb));
        // std::cout << "hi there from " << omp_get_thread_num() << std::endl;
	}

}

void add_slice_with_stride1_v1(float* self, float* source, std::vector<std::pair<int32_t, float*>> &index_to_row_list,
								const svfloat32_t valpha, const int64_t self_stride, const int64_t source_stride,
			 					const int row_begin, const int row_end, const int col_begin, const int col_end) {
#ifdef __ARM_FEATURE_SVE
	// #pragma omp parallel for
	// #pragma fj loop swp
	for (int i = col_begin; i < col_end; i += VEC_LEN) {
		svbool_t pg = svwhilelt_b32(i, col_end);
		svfloat32_t va = svld1(pg, &(self[i]));
		svfloat32_t vb;
		for (int j = row_begin; j < row_end; j++) {
			vb = svld1(pg, &((index_to_row_list[j].second)[i]));
			__builtin_prefetch(&((index_to_row_list[j].second)[i + 4 * VEC_LEN]), 0, 2);
			// __builtin_prefetch(&((index_to_row_list[j].second)[i + 5 * VEC_LEN]), 0, 2);
			// __builtin_prefetch(&((index_to_row_list[j].second)[i + 6 * VEC_LEN]), 0, 2);
			// __builtin_prefetch(&((index_to_row_list[j].second)[i + 7 * VEC_LEN]), 0, 2);
			va = svmla_f32_x(pg, va, vb, valpha);
		}
		svst1(pg, &(self[i]), va);
	}
/*
	for (int j = row_begin; j < row_end; j++) {
		for (int i = col_begin; i < col_end; i++) {
			float a = self[i];
			float b = (index_to_row_list[j].second)[i];
			a = a + b * 1.0;
			self[i] = a;
		}
	}
*/
#endif 
}

void add_slice_with_stride1_dynamic(float* self, float* source, std::vector<float*>* work_list,
									const svfloat32_t valpha, const int64_t self_stride, const int64_t source_stride,
									const int row_begin, const int row_end, const int col_begin, const int col_end, const int row_id) {
#ifdef __ARM_FEATURE_SVE
	// #pragma omp parallel for
	// #pragma fj loop swp
	// printf("tid = %d, dst_row_id = %d, col_begin = %d, col_end = %d \n", omp_get_thread_num(), row_id, col_begin, col_end);
	for (int i = col_begin; i < col_end; i += VEC_LEN) {
		svbool_t pg = svwhilelt_b32(i, col_end);
		svfloat32_t va = svld1(pg, &(self[i]));
		svfloat32_t vb;
		for (int j = row_begin; j < row_end; j++) {
			vb = svld1(pg, &((*work_list)[j][i]));
			__builtin_prefetch(&((*work_list)[j][i + 4 * VEC_LEN]), 0, 2);
			// __builtin_prefetch(&((index_to_row_list[j].second)[i + 5 * VEC_LEN]), 0, 2);
			// __builtin_prefetch(&((index_to_row_list[j].second)[i + 6 * VEC_LEN]), 0, 2);
			// __builtin_prefetch(&((index_to_row_list[j].second)[i + 7 * VEC_LEN]), 0, 2);
			va = svmla_f32_x(pg, va, vb, valpha);
			// va = svadd_f32_x(pg, va, vb);
		}
		svst1(pg, &(self[i]), va);
	}
#endif 
}

template<typename T>
void add_slice(T* self, T* source, const int64_t self_stride, const int64_t source_stride, const int64_t len) {
	
}



void sort_by_index(std::vector<std::pair<int32_t, float*>> &index_to_row_list,
					int64_t *index_data, const torch::Tensor &source,
					const int64_t &numel, const int64_t &source_stride) {
	auto start_time = system_clock::now();
	index_to_row_list.resize(numel);
	#pragma omp parallel for 
	for (auto i = 0; i < numel; i++) {
		index_to_row_list[i].first = static_cast<int>(index_data[i]);
		index_to_row_list[i].second = static_cast<float*>(source.data_ptr()) + i * source_stride;
		// index_to_row_list.push_back(std::make_pair(static_cast<int>(index_data[i]), 
		// 							static_cast<float*>(source.data_ptr()) + i * source_stride));
	}
	duration<double, std::milli> diff = (system_clock::now() - start_time);
	std::cout << "create index_to_row_list elaspsed time: " << diff.count() << " ms" << std::endl; 

	auto start_time1 = system_clock::now();
	std::sort(index_to_row_list.begin(), index_to_row_list.end(), [](auto &left, auto &right) {
		return left.first < right.first;
	});
	duration<double, std::milli> diff1 = (system_clock::now() - start_time);
	std::cout << "sort elaspsed time: " << diff1.count() << " ms" << std::endl; 
}

void index_add_kernel_with_stride1_v0(torch::Tensor &self, const torch::Tensor &source, int64_t *index_data, const int64_t &numel,
										const size_t &self_stride_bytes, const size_t &source_stride_bytes) {
	auto self_stride = self.stride(1);
	auto source_stride = source.stride(1);
	auto self_dim_size = self.size(0);
	for (auto i = 0; i < numel; i++) {
		auto self_i = index_data[i];
		TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
		auto self_data = static_cast<char*>(self.data_ptr()) + self_i * self_stride_bytes; 
		auto source_data = static_cast<char*>(source.data_ptr()) + i * source_stride_bytes;
		add_slice_with_stride1(reinterpret_cast<float*>(self_data), 
								reinterpret_cast<float*>(source_data),
								self_stride, source_stride, self.size(1));
	}
}

void index_add_kernel_with_stride1_static(torch::Tensor &self, const torch::Tensor &source, int64_t *index_data, 
											const float alpha, const int64_t &numel,
											const int64_t &self_stride, const int64_t &source_stride) {
	auto self_stride1 = self.stride(1);
	auto source_stride1 = source.stride(1);
	auto self_dim_size = self.size(0);

	int max_num_threads = omp_get_max_threads();
	int num_threads_on_row = 1;
	int num_threads_on_col = 1;

	
	std::vector<std::pair<int32_t, float*>> index_to_row_list;
	sort_by_index(index_to_row_list, index_data, source, numel, source_stride);

	switch(max_num_threads) {
		case 2:	{
			num_threads_on_row = 2;
			num_threads_on_col = 1;
		}break;
		case 4: {
			num_threads_on_row = 4;
			num_threads_on_col = 1;
		}break;
		case 8: {
			num_threads_on_row = 8;
			num_threads_on_col = 1;
		}break;
		case 12: {
			num_threads_on_row = 6;
			num_threads_on_col = 2;
		}break;
		case 24: {
			num_threads_on_row = 12;
			num_threads_on_col = 2;
		}break;
		case 36: {
			num_threads_on_row = 12;
			num_threads_on_col = 3;
		}break;
		case 48: {
			num_threads_on_row = 48;
			num_threads_on_col = 1;
		}break;
		default: {
			num_threads_on_row = max_num_threads;
			num_threads_on_col = 1;
		}break;
	}


	int num_rows = static_cast<int>(source.size(0));
	int num_cols = static_cast<int>(source.size(1));

	int row_chunk_size = divup(num_rows, num_threads_on_row);
	int col_chunk_size = divup(num_cols, num_threads_on_col);

	auto start_time1 = system_clock::now();
	
	#pragma omp parallel 
	{
		int tid = omp_get_thread_num();
		int num_threads = omp_get_num_threads();

		int outer_row_begin = tid / num_threads_on_col * row_chunk_size;
		int outer_row_end = std::min(num_rows, outer_row_begin + row_chunk_size);

		int outer_col_begin = tid % num_threads_on_col * col_chunk_size;
		int outer_col_end = std::min(num_cols, outer_col_begin + col_chunk_size);

		if (outer_row_begin > 0) {
			while ((outer_row_begin-1) >= 0 && index_to_row_list[outer_row_begin].first == index_to_row_list[outer_row_begin-1].first)
				--outer_row_begin;	
		}

		if (outer_row_end < num_rows) {
			while ((outer_row_end-1) >= 0 && index_to_row_list[outer_row_end].first == index_to_row_list[outer_row_end-1].first)
				--outer_row_end;
		}

		// printf("tid = %d, outer_row_begin = %d, outer_row_end = %d, outer_col_begin = %d, outer_col_end = %d\n",
		// 		tid, outer_row_begin, outer_row_end, outer_col_begin, outer_col_end);

		// std::cout << "tid = " << tid << ", outer_row_begin = " << outer_row_begin << ", outer_row_end = " <<outer_row_end 
		// 						<< ", outer_row_len = " << outer_row_end - outer_row_begin << std::endl;

		if (outer_row_begin < outer_row_end) {
/*
			svfloat32_t valpha = svdup_f32(alpha);
			int inner_row_begin = outer_row_begin, inner_row_end = outer_row_begin + 1;
			while (inner_row_begin < outer_row_end) {
				auto idx = index_to_row_list[inner_row_begin].first;
				TORCH_CHECK_INDEX((idx >= 0) && (idx < self_dim_size), "index out of range in self");
				while (inner_row_end < outer_row_end && idx == index_to_row_list[inner_row_end].first) 
					++inner_row_end;
				
				float *self_data = static_cast<float*>(self.data_ptr()) + idx * self_stride;
				float *source_data = nullptr;
				add_slice_with_stride1_v1(self_data, source_data, index_to_row_list, valpha, self_stride1, source_stride1,
											inner_row_begin, inner_row_end, outer_col_begin, outer_col_end);

				inner_row_begin = inner_row_end;
				++inner_row_end;

			}
*/

			svfloat32_t valpha = svdup_f32(alpha);
			for (int i = outer_row_begin; i < outer_row_end; i++) {
				auto self_i = index_to_row_list[i].first;
				TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
				auto self_data = static_cast<float*>(self.data_ptr()) + self_i * self_stride; 
				auto source_data = index_to_row_list[i].second;
				for (int j = outer_col_begin; j < outer_col_end; j += VEC_LEN) {
					svbool_t pg = svwhilelt_b32(j, outer_col_end);
					svfloat32_t va = svld1(pg, &(self_data[j]));
					svfloat32_t vb = svld1(pg, &(source_data[j]));
					svst1(pg, &(self_data[j]), svmla_f32_x(pg, va, vb, valpha));	
				}
			}
		}
	}
	duration<double, std::milli> diff1 = (system_clock::now() - start_time1);
	std::cout << "add_kernel elaspsed time: " << diff1.count() << " ms" << std::endl;
}

void init_num_threads(int& max_num_threads, int& num_threads_on_row, int& num_threads_on_col) {
  switch(max_num_threads) {
    case 1: {
      num_threads_on_row = 1;
			num_threads_on_col = 1;
    }break;
		case 2:	{
			num_threads_on_row = 2;
			num_threads_on_col = 1;
		}break;
		case 4: {
			num_threads_on_row = 4;
			num_threads_on_col = 1;
		}break;
		case 8: {
			num_threads_on_row = 8;
			num_threads_on_col = 1;
		}break;
		case 12: {
			num_threads_on_row = 12;
			num_threads_on_col = 1;
		}break;
		case 24: {
			num_threads_on_row = 12;
			num_threads_on_col = 2;
		}break;
		case 36: {
			num_threads_on_row = 12;
			num_threads_on_col = 3;
		}break;
		case 48: {
			num_threads_on_row = 48;
			num_threads_on_col = 1;
		}break;
		default: {
			num_threads_on_row = max_num_threads;
			num_threads_on_col = 1;
		}break;
	}
}

void index_add_kernel_with_stride1_dynamic_v0(torch::Tensor &self, const torch::Tensor &source, int64_t *index_data, 
											const float alpha, const int64_t &numel,
											const int64_t &self_stride, const int64_t &source_stride) {
	int self_dim_size = static_cast<int>(self.size(0));
	auto self_stride1 = self.stride(1);
	auto source_stride1 = source.stride(1);

	int num_rows = static_cast<int>(source.size(0));
	int num_cols = static_cast<int>(source.size(1));
	
	std::vector<std::vector<float*>*> work_list(self_dim_size, nullptr);
	// work_list.resize(self_dim_size);
	float* source_data_ptr = static_cast<float*>(source.data_ptr());

	int max_num_threads = omp_get_max_threads();
	int num_threads_on_row = max_num_threads;
	int num_threads_on_col = 1;
  int numel_int32 = static_cast<int>(numel);
  // init_num_threads(max_num_threads, num_threads_on_row, num_threads_on_col);

	int chunk_size = divup(self_dim_size, max_num_threads);
	int row_chunk_size = divup(num_rows, num_threads_on_row);
	int col_chunk_size = divup(num_cols, num_threads_on_col);

	int work_index_list[num_threads_on_row];
	svfloat32_t valpha = svdup_f32(alpha);

	#pragma omp parallel
	{
		auto start_time1 = system_clock::now();
		int tid = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		int idx_begin = tid * chunk_size;
		int idx_end = std::min(self_dim_size, idx_begin + chunk_size);

		for (int i = idx_begin; i < idx_end; i++) {
			work_list[i] = new std::vector<float*>();
			work_list[i]->reserve(numel_int32 / self_dim_size);
		}	

		for (int i = 0; i < numel_int32; i++) {
			int idx = static_cast<int>(index_data[i]);
			if (idx_begin <= idx && idx < idx_end) {
				// if (!work_list[idx]) {
				// 	work_list[idx] = new std::vector<float*>();
				// 	work_list[idx]->reserve(numel_int32 / self_dim_size);	
				// }
				work_list[idx]->push_back(source_data_ptr + i * source_stride);	
			}
		}
	
		#pragma omp barrier

		duration<double, std::milli> diff1 = (system_clock::now() - start_time1);
		if (tid == 0) {
			std::cout << "create list elaspsed time: " << diff1.count() << " ms" << std::endl;
		}

		auto start_time2 = system_clock::now();

		if (tid == 0) {
			int num_tasks = 0;
			int num_remain_tasks = numel_int32;
			int cur_row_id = 0;
			for (int i = 0; i < num_threads_on_row - 1; i++) {
        int num_tasks_per_thread = divup(num_remain_tasks, num_threads_on_row - i);
				while (cur_row_id < self_dim_size) {
					// num_tasks += (work_list[cur_row_id] == nullptr ? 0 : work_list[cur_row_id]->size());
					num_tasks += work_list[cur_row_id]->size();
					++cur_row_id;
					if (num_tasks >= num_tasks_per_thread) {
						work_index_list[i+1] = cur_row_id;
						num_remain_tasks -= num_tasks;
						num_tasks = 0;
						break;
					}
				}
			}
			work_index_list[0] = 0;
			work_index_list[num_threads_on_row] = self_dim_size;
		}

		#pragma omp barrier

		duration<double, std::milli> diff2 = (system_clock::now() - start_time2);
		if (tid == 0) {
			std::cout << "distribute work elaspsed time: " << diff2.count() << " ms" << std::endl;
		}

		auto start_time3 = system_clock::now();
	
		int outer_row_begin = work_index_list[tid / num_threads_on_col];
		int outer_row_end = work_index_list[tid / num_threads_on_col + 1];
		int outer_row_idx = outer_row_begin;

		int outer_col_begin = tid % num_threads_on_col * col_chunk_size;
		int outer_col_end = std::min(num_cols, outer_col_begin + col_chunk_size);

		while (outer_row_idx < outer_row_end) {
			if (!(work_list[outer_row_idx]->empty())) {
				int inner_row_begin = 0, inner_row_end = work_list[outer_row_idx]->size();
				auto idx = outer_row_idx;
				TORCH_CHECK_INDEX((idx >= 0) && (idx < self_dim_size), "index out of range in self");
					
				float *self_data = static_cast<float*>(self.data_ptr()) + idx * self_stride;
				float *source_data = nullptr;
				add_slice_with_stride1_dynamic(self_data, source_data, work_list[outer_row_idx], 
                                        valpha, self_stride1, source_stride1,
                                        inner_row_begin, inner_row_end, outer_col_begin, outer_col_end, outer_row_idx);
			}
			++outer_row_idx;
		}

		#pragma omp barrier
		for (int i = idx_begin; i < idx_end; i++)
			delete(work_list[i]);
		duration<double, std::milli> diff3 = (system_clock::now() - start_time3);
		if (tid == 0) {
			std::cout << "add kernel elaspsed time: " << diff3.count() << " ms" << std::endl;
		}
	}
}

void index_add_kernel_with_stride1_dynamic_v1(torch::Tensor &self, const torch::Tensor &source, int64_t *index_data, 
											const float alpha, const int64_t &numel,
											const int64_t &self_stride, const int64_t &source_stride) {
	auto self_dim = static_cast<int>(self.dim());
	int self_dim_size = static_cast<int>(self.size(0));
	auto self_stride1 = self.stride(self_dim - 1);
	auto source_stride1 = source.stride(self_dim - 1);

	int num_rows = static_cast<int>(source.size(0));
	// int num_cols = static_cast<int>(source.size(1));
	int num_cols = 1;
	for (int i = 1; i < self_dim; i++)
		num_cols *= static_cast<int>(source.size(i));
	
	std::vector<std::vector<float*>*> work_list(self_dim_size, nullptr);
	work_list.resize(self_dim_size);
	float* source_data_ptr = static_cast<float*>(source.data_ptr());
	int max_num_threads = omp_get_max_threads();
	// printf("max_num_threads = %d\n", max_num_threads);
	int numel_int = static_cast<int>(numel);
	
	int num_threads_on_row = 1;
	int num_threads_on_col = 1;
  	init_num_threads(max_num_threads, num_threads_on_row, num_threads_on_col);

	int chunk_size = divup(self_dim_size, max_num_threads);

	int row_chunk_size = divup(num_rows, num_threads_on_row);
	int col_chunk_size = divup(num_cols, num_threads_on_col);

	std::vector<int> work_index_list(num_threads_on_row + 1, self_dim_size);
	work_index_list[0] = 0;
	svfloat32_t valpha = svdup_f32(alpha);

	#pragma omp parallel
	{
		auto start_time1 = system_clock::now();
		int tid = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		int idx_begin = tid * chunk_size;
		int idx_end = std::min(self_dim_size, idx_begin + chunk_size);

		// for (int i = idx_begin; i < idx_end; i++) {
		// 	work_list[i] = new std::vector<float*>();
		// 	work_list[i]->reserve(divup(numel_int, self_dim_size));
		// }	

		for (int i = 0; i < numel_int; i++) {
			int idx = static_cast<int>(index_data[i]);
			if (idx_begin <= idx && idx < idx_end) {
				if (work_list[idx] == nullptr) {
					work_list[idx] = new std::vector<float*>();
					work_list[idx]->reserve(numel_int / self_dim_size);	
				}
				work_list[idx]->push_back(source_data_ptr + i * source_stride);	
			}
		}
	
		#pragma omp barrier
		duration<double, std::milli> diff1 = (system_clock::now() - start_time1);
		if (tid == 0) {
			std::cout << "create list elaspsed time: " << diff1.count() << " ms" << std::endl;
		}

		auto start_time3 = system_clock::now();

		if (tid == 0) {
			int num_tasks = 0;
			int num_remain_tasks = numel_int;
			int cur_row_id = 0;
			for (int i = 0; i < num_threads_on_row; i++) {
				int num_tasks_per_thread = divup(num_remain_tasks, num_threads_on_row - i);
				if (num_remain_tasks > 0) {
					while (cur_row_id < self_dim_size) {
						num_tasks += (work_list[cur_row_id] == nullptr ? 0 : work_list[cur_row_id]->size());
						// num_tasks += work_list[cur_row_id]->size();
						++cur_row_id;
						if (num_tasks >= num_tasks_per_thread) {
							work_index_list[i+1] = cur_row_id;
							num_remain_tasks -= num_tasks;
							num_tasks = 0;
							break;
						}
					}
				}
			}
			// work_index_list[0] = 0;
			// work_index_list[num_threads_on_row] = self_dim_size;
		}

		#pragma omp barrier
		duration<double, std::milli> diff3 = (system_clock::now() - start_time3);
		if (tid == 0) {
			std::cout << "distribute work elaspsed time: " << diff3.count() << " ms" << std::endl;
		}

		auto start_time2 = system_clock::now();
	
		int outer_row_begin = work_index_list[tid / num_threads_on_col];
		int outer_row_end = work_index_list[tid / num_threads_on_col + 1];
		int outer_row_idx = outer_row_begin;

		int outer_col_begin = tid % num_threads_on_col * col_chunk_size;
		int outer_col_end = std::min(num_cols, outer_col_begin + col_chunk_size);

		while (outer_row_idx < outer_row_end) {
			if (work_list[outer_row_idx] != nullptr && !(work_list[outer_row_idx]->empty())) {
				int inner_row_begin = 0, inner_row_end = work_list[outer_row_idx]->size();
				auto idx = outer_row_idx;
				TORCH_CHECK_INDEX((idx >= 0) && (idx < self_dim_size), "index out of range in self");
					
				float *self_data = static_cast<float*>(self.data_ptr()) + idx * self_stride;
				float *source_data = nullptr;
				add_slice_with_stride1_dynamic(self_data, source_data, work_list[outer_row_idx], 
												valpha, self_stride1, source_stride1,
												inner_row_begin, inner_row_end, outer_col_begin, outer_col_end, outer_row_idx);
			}
			++outer_row_idx;
		}

		#pragma omp barrier
		for (int i = idx_begin; i < idx_end; i++)
			if (work_list[i] != nullptr)
				delete(work_list[i]);

		duration<double, std::milli> diff2 = (system_clock::now() - start_time2);
		if (tid == 0) {
			// double access_data_bytes = sizeof(float) * (2.0 * self.size(0) * self.size(1) 
			// 											+ source.size(0) * source.size(1)
			// 											+ source.size(0));
			// double bandwidth = access_data_bytes / (diff2.count() / 1000.0) / 1000.0 / 1000.0 / 1000.0;
			// std::cout << "add kernel elaspsed time: " << diff2.count() << " ms" 
			// 			<< ", bandwidth(GB/s) = " << bandwidth << std::endl;
		}
	}
}

torch::Tensor& index_add(torch::Tensor& self, int64_t dim, const torch::Tensor& index, const torch::Tensor& source, const float alpha) {
	// convert dim, if dim is negative, it will be converted to positive
	dim = torch::maybe_wrap_dim(dim, self.dim());
	// obtain the elem num of index array
	auto numel = index.numel();
	
	// check if the parameters are valid
	TORCH_CHECK_INDEX(index.dim() <= 1, "index_add_(): Index is supposed to be a vector");
  	TORCH_CHECK(index.scalar_type() == at::ScalarType::Long || index.scalar_type() == at::ScalarType::Int,
          	  "index_add_(): Expected dtype int32/int64 for index");
  	TORCH_CHECK(self.scalar_type() == source.scalar_type(),
              "index_add_(): self and source must have the same scalar type");
  	TORCH_CHECK(dim == 0 || dim < source.dim(),
              "index_add_(): Indexing dim ", dim, " is out of bounds of tensor");
  	TORCH_CHECK(numel == (source.dim() == 0 ? 1 : source.size(dim)),
              "index_add_(): Number of indices should be equal to self.size(dim)");
  	TORCH_CHECK(source.dim() <= self.dim(),
              "index_add_(): Source dim ", source.dim(), " should not be greater than self dim ", self.dim());

/*
	// check if data are overlapping 
	at::assert_no_internal_overlap(self);
    at::assert_no_overlap(self, index);
    at::assert_no_overlap(self, source);
*/
		
	if (self.dim() > 1) {
		auto self_dim = self.dim();
		auto self_stride = self.stride(dim);
		auto self_element_size = torch::elementSize(self.scalar_type());

		auto source_dim = source.dim();
		auto source_stride = source.stride(dim);
		auto source_element_size = torch::elementSize(source.scalar_type());

		if (numel == 0)
			return self;

		auto self_stride_bytes = self_stride * self_element_size;
		auto source_stride_bytes = source_stride * source_element_size;
	
		auto index_data = index.data_ptr<int64_t>();
		// std::vector<std::pair<int32_t, float*>> index_to_row_list;
		// sort_by_index(index_to_row_list, index_data, source, numel, source_stride);
/*
		for (auto i: index_to_row_list) {
			std::cout << i.first << ", ";
		}
		std::cout << std::endl;
*/
		// special case, self_dim == source_dim == 2
		if (self_dim == source_dim && (self_dim == 2 || self_dim == 3)) {
			if (dim == 0) {
/*
				auto start_time = system_clock::now();
				index_add_kernel_with_stride1_v0(self, source, index_data, numel, self_stride_bytes, source_stride_bytes);
				duration<double, std::milli> diff = (system_clock::now() - start_time);
				std::cout << "index_add_v0 elaspsed time: " << diff.count() << " ms" << std::endl; 
*/
/*
				auto start_time1 = system_clock::now();
				index_add_kernel_with_stride1_static(self, source, index_data, alpha, numel, self_stride, source_stride);
				duration<double, std::milli> diff1 = (system_clock::now() - start_time1);
				std::cout << "index_add_static elaspsed time: " << diff1.count() << " ms" << std::endl;
*/
				auto start_time2 = system_clock::now();
				index_add_kernel_with_stride1_dynamic_v1(self, source, index_data, alpha, numel, self_stride, source_stride);
				duration<double, std::milli> diff2 = (system_clock::now() - start_time2);
				std::cout << "index_add_dynamic elaspsed time: " << diff2.count() << " ms" << std::endl;
			}
			else if (dim == 1) { /* dim == 1 */ 
				self_stride = self.stride(0);
				source_stride = source.stride(0);
			}
		} 
		else { /* self_dim != source_dim or self_dim != 2 */
			std::cout << "didn't support self_dim != 2 or 3" << std::endl;
		}
	} 
	else { /* self_dim == 1 */
		std::cout << "didn't support self_dim != 2 or 3" << std::endl;
	}
	return self;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("index_add", &index_add, "A function that adds two numbers");
}
