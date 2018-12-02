/* Copyright (c) 2018 Bartosz Białoskórski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#ifndef NEURALNET_INCLUDE_CUDA_UTILITIES_CUH_
#define NEURALNET_INCLUDE_CUDA_UTILITIES_CUH_

namespace neuralnet {

const unsigned int kBlockSize = 256;
const unsigned int kTileDim = 16;

template <typename TType, unsigned int TBlockSize>
__global__ void matrix_vector_mult_kernel(TType* __restrict__ d_out_vector,
    const TType* __restrict__ d_in_matrix,
    const TType* __restrict__ d_in_vector,
    int num_rows, int num_columns);

template <typename TType, unsigned int TBlockSize>
__global__ void vector_sum_kernel(TType* d_out, TType* d_in, int size);

template <typename TType, unsigned int TBlockSize>
__global__ void vector_max_kernel(TType* d_out, TType* d_in, int size);

template <typename TType, unsigned int TTileDim>
__global__ void transpose_matrix_kernel(TType* d_out, TType* d_in,
                                        int num_rows, int num_cols);

template <typename TType, unsigned int TTileDim>
__global__ void matrix_mult_kernel(TType* __restrict__ d_out,
                                   const TType* __restrict__ d_in_a,
                                   const TType* __restrict__ d_in_b,
                                   int num_rows_a, int num_cols_a,
                                   int num_cols_b);

/*****************************************************************************/

template <typename TType, unsigned int TBlockSize>
__global__ void matrix_vector_mult_kernel(TType* __restrict__ d_out_vector,
  const TType* __restrict__ d_in_matrix,
  const TType* __restrict__ d_in_vector, int num_rows, int num_columns) {

  __shared__ TType shared_mem[TBlockSize];

  int tid = threadIdx.x;
  int grid_tid = tid + blockIdx.x * TBlockSize;
  int threads_per_row = ceilf((float)num_columns / (float)TBlockSize)
                        * TBlockSize;
  int row = grid_tid / threads_per_row;
  int column = grid_tid - row * threads_per_row;
  // Discarding extra threads from a block that would otherwise operate on
  // elements from two different rows of the input matrix.
  if (column < num_columns) {
    // Preparing data in shared memory for reduction.
    shared_mem[tid] = d_in_matrix[row * num_columns + column]
                      * d_in_vector[column];
  }
  else {
    // Padding extra shared memory.
    shared_mem[tid] = 0;
  }

  __syncthreads();

  // Performing unrolled sum reduction on shared memory inside this block.
  if (TBlockSize >= 1024) {
    if (tid < 512)
      shared_mem[tid] += shared_mem[tid + 512];

    __syncthreads();
  }

  if (TBlockSize >= 512) {
    if (tid < 256)
      shared_mem[tid] += shared_mem[tid + 256];

    __syncthreads();
  }

  if (TBlockSize >= 256) {
    if (tid < 128)
      shared_mem[tid] += shared_mem[tid + 128];

    __syncthreads();
  }

  if (TBlockSize >= 128) {
    if (tid < 64)
      shared_mem[tid] += shared_mem[tid + 64];

    __syncthreads();
  }

  // Reducing last warp. Everything in this if statement runs in a 'single
  // instruction multiple data' fashion so no synchronization is required.
  if (tid < 32) {
    if (TBlockSize >= 64) shared_mem[tid] += shared_mem[tid + 32];
    if (TBlockSize >= 32) shared_mem[tid] += shared_mem[tid + 16];
    if (TBlockSize >= 16) shared_mem[tid] += shared_mem[tid + 8];
    if (TBlockSize >= 8) shared_mem[tid] += shared_mem[tid + 4];
    if (TBlockSize >= 4) shared_mem[tid] += shared_mem[tid + 2];
    if (TBlockSize >= 2) shared_mem[tid] += shared_mem[tid + 1];
  }

  // Atomically adding partial result to output vector.
  if (tid == 0)
    atomicAdd(d_out_vector + row, shared_mem[0]);
}

template <typename TType, unsigned int TBlockSize>
__global__ void vector_sum_kernel(TType* d_out, TType* d_in, int size) {

  extern __shared__ TType shared_mem[];

  int tid = threadIdx.x;
  int grid_tid = tid + blockIdx.x * TBlockSize * 2;
  int grid_size = TBlockSize * 2 * gridDim.x;

  // Loading input data into shared memory.
  shared_mem[tid] = (TType)0;
  while (grid_tid < size) {
    shared_mem[tid] += d_in[grid_tid] + d_in[grid_tid + TBlockSize];
    grid_tid += grid_size;
  }

  __syncthreads();

  // Performing unrolled sum reduction on shared memory inside this block.
  if (TBlockSize >= 1024) {
    if (tid < 512)
      shared_mem[tid] += shared_mem[tid + 512];

    __syncthreads();
  }

  if (TBlockSize >= 512) {
    if (tid < 256)
      shared_mem[tid] += shared_mem[tid + 256];

    __syncthreads();
  }

  if (TBlockSize >= 256) {
    if (tid < 128)
      shared_mem[tid] += shared_mem[tid + 128];

    __syncthreads();
  }

  if (TBlockSize >= 128) {
    if (tid < 64)
      shared_mem[tid] += shared_mem[tid + 64];

    __syncthreads();
  }

  // Reducing last warp. Everything in this if statement runs in a 'single
  // instruction multiple data' fashion so no synchronization is required.
  if (tid < 32) {
    if (TBlockSize >= 64) shared_mem[tid] += shared_mem[tid + 32];
    if (TBlockSize >= 32) shared_mem[tid] += shared_mem[tid + 16];
    if (TBlockSize >= 16) shared_mem[tid] += shared_mem[tid + 8];
    if (TBlockSize >= 8) shared_mem[tid] += shared_mem[tid + 4];
    if (TBlockSize >= 4) shared_mem[tid] += shared_mem[tid + 2];
    if (TBlockSize >= 2) shared_mem[tid] += shared_mem[tid + 1];
  }
  // Writing result to output array.
  if (tid == 0)
    d_out[blockIdx.x] = shared_mem[0];
}

template <typename TType, unsigned int TBlockSize>
__global__ void vector_max_kernel(TType* d_out, TType* d_in, int size) {

  extern __shared__ TType shared_mem[];

  int tid = threadIdx.x;
  int grid_tid = tid + blockIdx.x * TBlockSize * 2;
  int grid_size = TBlockSize * 2 * gridDim.x;

  // Loading input data into shared memory.
  shared_mem[tid] = d_in[grid_tid];
  while (grid_tid < size) {
    TType tmp = d_in[grid_tid + TBlockSize];

    if (shared_mem[tid] < tmp) shared_mem[tid] = tmp;
    grid_tid += grid_size;
  }

  __syncthreads();

  // Performing unrolled max reduction on shared memory inside this block.
  if (TBlockSize >= 1024) {
    if (tid < 512)
      if (shared_mem[tid] < shared_mem[tid + 512])
        shared_mem[tid] = shared_mem[tid + 512];

    __syncthreads();
  }

  if (TBlockSize >= 512) {
    if (tid < 256)
      if (shared_mem[tid] < shared_mem[tid + 256])
        shared_mem[tid] = shared_mem[tid + 256];

    __syncthreads();
  }

  if (TBlockSize >= 256) {
    if (tid < 128)
      if (shared_mem[tid] < shared_mem[tid + 128])
        shared_mem[tid] = shared_mem[tid + 128];

    __syncthreads();
  }

  if (TBlockSize >= 128) {
    if (tid < 64)
      if (shared_mem[tid] < shared_mem[tid + 64])
        shared_mem[tid] = shared_mem[tid + 64];

    __syncthreads();
  }

  // Reducing last warp. Everything in this if statement runs in a 'single
  // instruction multiple data' fashion so no synchronization is required.
  if (tid < 32) {
    if (TBlockSize >= 64)
      if (shared_mem[tid] < shared_mem[tid + 32])
        shared_mem[tid] = shared_mem[tid + 32];
    if (TBlockSize >= 32)
      if (shared_mem[tid] < shared_mem[tid + 16])
        shared_mem[tid] = shared_mem[tid + 16];
    if (TBlockSize >= 16)
      if (shared_mem[tid] < shared_mem[tid + 8])
        shared_mem[tid] = shared_mem[tid + 8];
    if (TBlockSize >= 8)
      if (shared_mem[tid] < shared_mem[tid + 4])
        shared_mem[tid] = shared_mem[tid + 4];
    if (TBlockSize >= 4)
      if (shared_mem[tid] < shared_mem[tid + 2])
        shared_mem[tid] = shared_mem[tid + 2];
    if (TBlockSize >= 2)
      if (shared_mem[tid] < shared_mem[tid + 1])
        shared_mem[tid] = shared_mem[tid + 1];
  }
  // Writing result to output array.
  if (tid == 0)
    d_out[blockIdx.x] = shared_mem[0];
}

template <typename TType, unsigned int TTileDim>
__global__ void transpose_matrix_kernel(TType* d_out, TType* d_in,
                                        int num_rows, int num_cols) {

  __shared__ TType shared_mem[TTileDim][TTileDim];

  int x = threadIdx.x;
  int y = threadIdx.y;
  int top_left_x = blockIdx.x * TTileDim;
  int top_left_y = blockIdx.y * TTileDim;
  int transpose_top_left_x = top_left_y;
  int transpose_top_lefy_y = top_left_x;

  if (top_left_x + x < num_cols && top_left_y + y < num_rows) {
    // Writing data to shared memory in transpose fashion.
    shared_mem[y][x] = d_in[(top_left_y + y) * num_cols + top_left_x + x];
  }

  __syncthreads();

  if (transpose_top_lefy_y + y < num_cols && transpose_top_left_x + x < num_rows) {
    // Coalesced write to global memory.
    d_out[(transpose_top_lefy_y + y) * num_rows + transpose_top_left_x + x] =
        shared_mem[x][y];
  }
}

template <typename TType, unsigned int TTileDim>
__global__ void matrix_mult_kernel(TType* __restrict__ d_out,
                                   const TType* __restrict__ d_in_a,
                                   const TType* __restrict__ d_in_b,
                                   int num_rows_a, int num_cols_a,
                                   int num_cols_b) {

  __shared__ TType a_shared[TTileDim][TTileDim];
  __shared__ TType b_shared[TTileDim][TTileDim];

  int x = threadIdx.x;
  int y = threadIdx.y;
  int global_x = blockIdx.x * TTileDim + x;
  int global_y = blockIdx.y * TTileDim + y;

  // Padding shared memory.
  a_shared[x][y] = 0;
  b_shared[x][y] = 0;

  __syncthreads();

  TType dot_product = 0;
  int block_offset = 0;

  while (block_offset * TTileDim < num_cols_a) {
    // Loading corresponding elements of A to shared memory.
    if (global_y < num_rows_a && block_offset * TTileDim + x < num_cols_a) {
      a_shared[y][x] =
          d_in_a[global_y * num_cols_a + block_offset * TTileDim + x];
    }
    // Loading corresponding elements of B to shared memory in transposed
    // fashion.
    if (global_x < num_cols_b && block_offset * TTileDim + y < num_cols_a) {
      b_shared[x][y] =
          d_in_b[global_x + num_cols_b * (block_offset * TTileDim + y)];
    }

    ++block_offset;

    __syncthreads();

    // Performing multiplication of corresponding tiles of A and B.
    // Loops are unrolled for performance.
    dot_product += a_shared[y][0] * b_shared[x][0];
    if (TTileDim > 1) {
      dot_product += a_shared[y][1] * b_shared[x][1];
    }
    if (TTileDim > 2) {
      dot_product += a_shared[y][2] * b_shared[x][2]
                     + a_shared[y][3] * b_shared[x][3];
    }
    if (TTileDim > 4) {
      dot_product += a_shared[y][4] * b_shared[x][4]
                     + a_shared[y][5] * b_shared[x][5]
                     + a_shared[y][6] * b_shared[x][6]
                     + a_shared[y][7] * b_shared[x][7];
    }
    if (TTileDim > 8) {
      dot_product += a_shared[y][8] * b_shared[x][8]
                     + a_shared[y][9] * b_shared[x][9]
                     + a_shared[y][10] * b_shared[x][10]
                     + a_shared[y][11] * b_shared[x][11]
                     + a_shared[y][12] * b_shared[x][12]
                     + a_shared[y][13] * b_shared[x][13]
                     + a_shared[y][14] * b_shared[x][14]
                     + a_shared[y][15] * b_shared[x][15];
    }
    if (TTileDim > 16) {
      dot_product += a_shared[y][16] * b_shared[x][16]
                     + a_shared[y][17] * b_shared[x][17]
                     + a_shared[y][18] * b_shared[x][18]
                     + a_shared[y][19] * b_shared[x][19]
                     + a_shared[y][20] * b_shared[x][20]
                     + a_shared[y][21] * b_shared[x][21]
                     + a_shared[y][22] * b_shared[x][22]
                     + a_shared[y][23] * b_shared[x][23]
                     + a_shared[y][24] * b_shared[x][24]
                     + a_shared[y][25] * b_shared[x][25]
                     + a_shared[y][26] * b_shared[x][26]
                     + a_shared[y][27] * b_shared[x][27]
                     + a_shared[y][28] * b_shared[x][28]
                     + a_shared[y][29] * b_shared[x][29]
                     + a_shared[y][30] * b_shared[x][30]
                     + a_shared[y][31] * b_shared[x][31];
    }

    __syncthreads();

    // Padding shared memory in case only portion of this tile covers matrix
    // in the next iteration.
    a_shared[x][y] = 0;
    b_shared[x][y] = 0;

    __syncthreads();
  }
  if (global_y < num_rows_a && global_x < num_cols_b) {
    d_out[global_y * num_cols_b + global_x] = dot_product;
  }
}

} // namespace neuralnet

#endif // NEURALNET_INCLUDE_CUDA_UTILITIES_CUH_
