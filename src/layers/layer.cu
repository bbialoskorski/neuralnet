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

#include "layers/layer.hpp"

#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_double_functions.h"
#include "device_launch_parameters.h"
#if __CUDA_ARCH__ >= 600
#include "sm_60_atomic_functions.hpp"
#endif
#include "gpu_allocation_manager.hpp"
#include "gpu_utilities.cuh"

namespace neuralnet {

template <unsigned int TTileDim>
__global__ void compute_velocity_kernel(double* __restrict__ d_velocity,
    const double* __restrict__ d_error,
    const double* __restrict__ d_prev_layer_output,
    int num_neurons, int num_inputs, int mini_batch_size, double momentum) {
  // Weight derivative is calculated by applying the chain rule to
  // calculated error and the following partial derivative:
  //
  // d(a[row])/d(w[row][col]) = o[col],
  //
  // where:
  // a[row]: activation of neuron row in current layer
  // w[row][col]: weight of the connection between neuron row in
  //   current layer and neuron col in previous layer
  // o[col]: output of neuron col in previous layer.
  //
  // Applying chain rule to these derivatives gives us
  // dE/d(w[row][col]) which is the gradient we're looking for.
  __shared__ double shared_errors[TTileDim];
  __shared__ double shared_prev_layer_output[TTileDim];

  int x = threadIdx.x;
  int y = threadIdx.y;
  int global_x = blockIdx.x * TTileDim + x;
  int global_y = blockIdx.y * TTileDim + y;

  if (global_x < num_inputs && global_y < num_neurons) {
    // Calculating gradient on the current mini-batch.
    double gradient = 0.0;

    for (int batch_column = 0; batch_column < mini_batch_size; ++batch_column) {

      if (x == 0) {
        // Threads in the first column of a tile are loading layer's error to
        // shared memory.
        shared_errors[y] = d_error[global_y * mini_batch_size + batch_column];
      }

      if (y == 0) {
        // Threads in the first row of a tile are loading output from previous
        // layer in a forward pass to shared memory.
        shared_prev_layer_output[x] =
            d_prev_layer_output[global_x * mini_batch_size + batch_column];
      }

      __syncthreads();

      if (global_x != num_inputs - 1) {
        gradient += shared_errors[y] * shared_prev_layer_output[x];
      }
      else {
        // Bias input weight.
        gradient += shared_errors[y];
      }

      __syncthreads();

    }
    // Momentum backpropagation:
    // velocity(t) = momentum * velocity(t - 1) + (1.0 - momentum) * dE/dW,
    //
    // where:
    //  t is a time step,
    //  dE/dW is a weights gradient calculated on the current mini-batch.
    d_velocity[global_y * num_inputs + global_x] =
        d_velocity[global_y * num_inputs + global_x] * momentum
        + (1.0 - momentum) * gradient;
  }
}

__global__ void update_kernel(double* __restrict__ d_weights,
                              const double* __restrict__ d_velocity,
                              int size, double learning_rate) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size)
    d_weights[tid] -= learning_rate * d_velocity[tid];
}

void Layer::ComputeActivationGpu(double* d_activation,
                                  const std::vector<double>& input) {
  GpuAllocationManager manager;
  int mini_batch_size = input.size() / (num_inputs_ - 1);
  // We allocate additional #mini_batch_size elements for input matrix because
  // for each input in the batch we want to add bias neuron input value.
  double* d_input = (double*)manager.AllocateDevice(
      (input.size() + mini_batch_size) * sizeof(double));
  double* d_weights = (double*)manager.AllocateDevice(
      weights_.size() * sizeof(double));
  cudaError_t cuda_status;

  cuda_status = cudaMemset(d_activation, 0,
                           num_neurons_ * mini_batch_size * sizeof(double));
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "Error ("
                          + std::string(cudaGetErrorString(cuda_status))
                          + ") occured while setting activations array to 0s on\
 device.";
  }

  cuda_status = cudaMemcpy(d_input, input.data(),
                           input.size() * sizeof(double),
                           cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "Error ("
                          + std::string(cudaGetErrorString(cuda_status))
                          + ") occured while copying input data to device.";
    throw std::runtime_error(err_msg);
  }
  // Adding biases to input array by copying data from vector to device.
  // This is the least hacky way to do it because there is no cuda memset
  // function for 8 byte words.
  std::vector<double> biases(mini_batch_size, 1.0);

  cuda_status = cudaMemcpy(d_input + input.size(), biases.data(),
                           biases.size() * sizeof(double),
                           cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "Error ("
                          + std::string(cudaGetErrorString(cuda_status))
                          + ") occured while copying bias input data to device.";
    throw std::runtime_error(err_msg);
  }

  cuda_status = cudaMemcpy(d_weights, weights_.data(),
                           weights_.size() * sizeof(double),
                           cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "Error ("
                          + std::string(cudaGetErrorString(cuda_status))
                          + ") occured while copying weights data to device.";
    throw std::runtime_error(err_msg);
  }

  if (mini_batch_size == 1) {
    // Input is a vector, using specialized multiplication function for
    // performance.
    dim3 grid(std::ceil((float)num_inputs_ / (float)kBlockSize)*num_neurons_);
    dim3 block(kBlockSize);

    matrix_vector_mult_kernel<double, kBlockSize><<<grid, block>>>(
        d_activation, d_weights, d_input, num_neurons_, num_inputs_);
  }
  else {
    // Input is a matrix.
    dim3 grid(std::ceil((float)mini_batch_size / (float)kTileDim),
              std::ceil((float)num_neurons_ / (float)kTileDim));
    dim3 tile(kTileDim, kTileDim);

    matrix_mult_kernel<double, kTileDim><<<grid, tile>>>(
        d_activation, d_weights, d_input, num_neurons_, num_inputs_,
        mini_batch_size);
  }

  // Freeing memory allocated on device.
  manager.FreeDevice(d_input);
  manager.FreeDevice(d_weights);
}

void Layer::ComputeWeightedErrorGpu(double* d_weighted_error,
                                    double* d_weights, double* d_error) {
  int mini_batch_size = error_.size() / num_neurons_;
  cudaMemset(d_weighted_error, 0,
             (num_inputs_ - 1) * mini_batch_size * sizeof(double));
  dim3 transpose_grid(std::ceil((float)num_inputs_ / (float)kTileDim),
                      std::ceil((float)num_neurons_ / (float)kTileDim));
  dim3 tile(kTileDim, kTileDim);

  transpose_matrix_kernel<double, kTileDim><<<transpose_grid, tile>>>(
      d_weights, d_weights, num_neurons_, num_inputs_);
  // While multiplying matrices we will not be using d_weights corresponging to
  // bias neuron link so we are skipping last row of transposed d_weights
  // matrix (thus number of rows in transposed d_weights matrix is
  // 'num_inputs_ - 1').
  if (mini_batch_size == 1) {
    // Error is a vector, using specialized multiplication function for
    // performance.
    dim3 grid(
        std::ceil((float)num_neurons_ / (float)kBlockSize) * (num_inputs_ - 1));
    dim3 block(kBlockSize);

    matrix_vector_mult_kernel<double, kBlockSize><<<grid, block>>>(
        d_weighted_error, d_weights, d_error, num_inputs_ - 1,
        num_neurons_);
  }
  else {
    // Error is a matrix.
    dim3 grid(std::ceil((float)mini_batch_size / (float)kTileDim),
              std::ceil((float)(num_inputs_ - 1) / (float)kTileDim));

    matrix_mult_kernel<double, kTileDim><<<grid, tile>>>(d_weighted_error,
        d_weights, d_error, num_inputs_ - 1, num_neurons_, mini_batch_size);
  }
}

void Layer::ComputeVelocityGpu(double* d_velocity, const double* d_error,
                               const double* d_prev_layer_output,
                               double momentum) {
  if (momentum <= 0 || momentum >= 1)
    throw std::invalid_argument(
        "momentum coefficient should have a value between 0 and 1.");

  int mini_batch_size = error_.size() / num_neurons_;
  dim3 grid(std::ceil((float)num_inputs_ / (float)kTileDim),
            std::ceil((float)num_neurons_ / (float)kTileDim));
  dim3 tile(kTileDim, kTileDim);

  compute_velocity_kernel<kTileDim><<<grid, tile>>>(d_velocity, d_error,
                                                    d_prev_layer_output,
                                                    num_neurons_, num_inputs_,
                                                    mini_batch_size, momentum);
}

void Layer::UpdateGpu(double learning_rate) {
  if (learning_rate <= 0)
    throw std::invalid_argument("learning_rate has to be a positive number.");

  GpuAllocationManager manager;
  double* d_weights = (double*)manager.AllocateDevice(
      weights_.size() * sizeof(double));
  double* d_velocity = (double*)manager.AllocateDevice(
      velocity_.size() * sizeof(double));;

  cudaError_t cuda_status;
  cuda_status = cudaMemcpy(d_weights, weights_.data(),
                           weights_.size() * sizeof(double),
                           cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "Error ("
                          + std::string(cudaGetErrorString(cuda_status))
                          + ") occured while copying weights data to device.";
    throw std::runtime_error(err_msg);
  }

  cuda_status = cudaMemcpy(d_velocity, velocity_.data(),
                           velocity_.size() * sizeof(double),
                           cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "Error ("
                          + std::string(cudaGetErrorString(cuda_status))
                          + ") occured while copying velocity data to device.";
    throw std::runtime_error(err_msg);
  }

  dim3 grid(std::ceil((float)weights_.size() / (float)kBlockSize));
  dim3 block(kBlockSize);

  update_kernel<<<grid, block>>>(d_weights, d_velocity, weights_.size(),
                                 learning_rate);

  // Copying updated weights back to host.
  cuda_status = cudaMemcpy(weights_.data(), d_weights,
                           weights_.size() * sizeof(double),
                           cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "Error ("
                          + std::string(cudaGetErrorString(cuda_status))
                          + ") occured while copying weights data to host.";
    throw std::runtime_error(err_msg);
  }

  // Freeing memory allocated on device.
  manager.FreeDevice(d_weights);
  manager.FreeDevice(d_velocity);
}

} // namespace neuralnet
