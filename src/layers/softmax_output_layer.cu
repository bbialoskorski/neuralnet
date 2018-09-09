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

#include "layers/softmax_output_layer.hpp"

#include <algorithm>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "gpu_allocation_manager.hpp"
#include "gpu_utilities.cuh"

namespace neuralnet {

__global__ void cross_entropy_derivative_kernel(double* __restrict__ d_error,
    const double* __restrict__ d_output,
    const double* __restrict__ d_target_output, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size)
    d_error[tid] = d_output[tid] - d_target_output[tid];
}

void SoftmaxOutputLayer::ForwardPropGpu(const std::vector<double>& input) {
  GpuAllocationManager manager;
  int mini_batch_size = input.size() / (num_inputs_ - 1);
  double* d_activation = (double*)manager.AllocateDevice(
      num_neurons_ * mini_batch_size * sizeof(double));
  // Resizing activation to fit size of current mini batch.
  activation_.resize(num_neurons_ * mini_batch_size);

  ComputeActivationGpu(d_activation, input);

  cudaError_t cuda_status;
  cuda_status = cudaMemcpy(activation_.data(), d_activation,
                           activation_.size() * sizeof(double),
                           cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "Error ("
                          + std::string(cudaGetErrorString(cuda_status))
                          + ") occured while copying activation data to host.";
    throw std::runtime_error(err_msg);
  }

  // Applying numerically stable softmax to computed activation.
  StableSoftmax();

  // Freeing memory allocated on device.
  manager.FreeDevice(d_activation);
}

void SoftmaxOutputLayer::BackPropGpu(const std::vector<double>& target_output,
    const std::vector<double>& prev_layer_output, double momentum) {
  if (momentum <= 0 || momentum >= 1)
    throw std::invalid_argument("momentum coefficient should have a value\
 between 0 and 1.");

  GpuAllocationManager manager;
  int mini_batch_size = target_output.size() / num_neurons_;
  double* d_error = (double*)manager.AllocateDevice(
      num_neurons_ * mini_batch_size * sizeof(double));
  double* d_weights = (double*)manager.AllocateDevice(
      weights_.size() * sizeof(double));
  double* d_velocity = (double*)manager.AllocateDevice(
      velocity_.size() * sizeof(double));
  double* d_output = (double*)manager.AllocateDevice(
      output_.size() * sizeof(double));
  double* d_target_output = (double*)manager.AllocateDevice(
      target_output.size() * sizeof(double));
  double* d_prev_layer_output = (double*)manager.AllocateDevice(
      prev_layer_output.size() * sizeof(double));
  double* d_weighted_error = (double*)manager.AllocateDevice(
      (num_inputs_ - 1) * mini_batch_size * sizeof(double));

  // Resizing error and weigted_error to fit size of current mini-batch.
  error_.resize(num_neurons_ * mini_batch_size);
  weighted_error_.resize((num_inputs_ - 1) * mini_batch_size);

  cudaError_t cuda_status;
  // Moving data to device.
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

  cuda_status = cudaMemcpy(d_output, output_.data(),
                           output_.size() * sizeof(double),
                           cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "Error ("
                          + std::string(cudaGetErrorString(cuda_status))
                          + ") occured while copying output data to device.";
    throw std::runtime_error(err_msg);
  }

  cuda_status = cudaMemcpy(d_target_output, target_output.data(),
                           target_output.size() * sizeof(double),
                           cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "Error ("
                          + std::string(cudaGetErrorString(cuda_status))
                          + ") occured while copying target output data to\
 device.";
    throw std::runtime_error(err_msg);
  }

  cuda_status = cudaMemcpy(d_prev_layer_output, prev_layer_output.data(),
                           prev_layer_output.size() * sizeof(double),
                           cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "Error ("
                          + std::string(cudaGetErrorString(cuda_status))
                          + ") occured while copying previous layer output data\
 to device.";
    throw std::runtime_error(err_msg);
  }

  dim3 grid(std::ceil((float)error_.size() / (float)kBlockSize));
  dim3 block(kBlockSize);

  cross_entropy_derivative_kernel<<<grid, block>>>(d_error, d_output,
                                                   d_target_output,
                                                   error_.size());

  ComputeVelocityGpu(d_velocity, d_error, d_prev_layer_output, momentum);

  ComputeWeightedErrorGpu(d_weighted_error, d_weights, d_error);

  // Moving results to host.
  cuda_status = cudaMemcpy(velocity_.data(), d_velocity,
                           velocity_.size() * sizeof(double),
                           cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "Error ("
                          + std::string(cudaGetErrorString(cuda_status))
                          + ") occured while copying velocity data to host.";
    throw std::runtime_error(err_msg);
  }

  cuda_status = cudaMemcpy(weighted_error_.data(), d_weighted_error,
                           weighted_error_.size() * sizeof(double),
                           cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "Error ("
                          + std::string(cudaGetErrorString(cuda_status))
                          + ") occured while copying weighted error data to\
 host.";
    throw std::runtime_error(err_msg);
  }

  // Freeing memory allocated on device.
  manager.FreeDevice(d_error);
  manager.FreeDevice(d_weights);
  manager.FreeDevice(d_velocity);
  manager.FreeDevice(d_output);
  manager.FreeDevice(d_target_output);
  manager.FreeDevice(d_prev_layer_output);
  manager.FreeDevice(d_weighted_error);
}

}
