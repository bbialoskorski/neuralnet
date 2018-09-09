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

#include "layers/sigmoid_output_layer.hpp"

#include <algorithm>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "gpu_allocation_manager.hpp"
#include "gpu_utilities.cuh"

namespace neuralnet {

__global__ void sigmoid_kernel(double* d_out, double* d_in, int size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) {
      // Calculating sigmoid function value for each input.
      // S(x) = 1 / (1 + exp(-x))
      d_out[tid] = 1.0 / (1.0 + exp(-d_in[tid]));
  }
}

__global__ void compute_mean_squared_error_derivative_kernel(
    double* __restrict__ d_error, const double* __restrict__ d_layer_output,
    const double* __restrict__ d_target_output, int size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) {
    double output = d_layer_output[tid];
    d_error[tid] = (output - d_target_output[tid]) * output * (1.0 - output);
  }
}

__global__ void update_kernel(double* __restrict__ d_weights,
                              const double* __restrict__ d_velocity,
                              int size, double learning_rate,
                              int num_training_examples) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size)
    d_weights[tid] -= learning_rate * d_velocity[tid]
                      / (double)num_training_examples;
}

void SigmoidOutputLayer::ForwardPropGpu(const std::vector<double>& input) {
  GpuAllocationManager manager;
  int mini_batch_size = input.size() / (num_inputs_ - 1);
  double* d_activations = (double*)manager.AllocateDevice(
      num_neurons_ * mini_batch_size * sizeof(double));
  double* d_output = (double*)manager.AllocateDevice(
      num_neurons_ * mini_batch_size * sizeof(double));

  // Resizing output and activation to fit size of current mini-batch.
  output_.resize(num_neurons_ * mini_batch_size);
  activation_.resize(num_neurons_ * mini_batch_size);

  ComputeActivationGpu(d_activations, input);

  dim3 grid(std::ceil((float)output_.size() / (float)kBlockSize));
  dim3 block(kBlockSize);

  // Applying sigmoid to computed activation.
  sigmoid_kernel<<<grid, block>>>(d_output, d_activations,
                                  num_neurons_ * mini_batch_size);

  cudaError_t cuda_status;
  cuda_status = cudaMemcpy(output_.data(), d_output,
                           output_.size() * sizeof(double),
                           cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess) {
    std::string err_msg = "Error ("
                          + std::string(cudaGetErrorString(cuda_status))
                          + ") occured while copying output data to host.";
    throw std::runtime_error(err_msg);
  }
 
  // Freeing memory allocated on device.
  manager.FreeDevice(d_activations);
  manager.FreeDevice(d_output);
}

void SigmoidOutputLayer::BackPropGpu(
    const std::vector<double>& target_output,
    const std::vector<double>& prev_layer_output, double momentum) {
  if (momentum <= 0 || momentum >= 1)
    throw std::invalid_argument("momentum coefficient should have a value\
 between 0 and 1.");

  int mini_batch_size = target_output.size() / num_neurons_;
  // Incrementing num_training_examples after back pass instead of forward pass
  // to maintain correctness in case of forward passes not being part of
  // training.
  num_training_examples_ += mini_batch_size;

  GpuAllocationManager manager;
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

  // Resizing error and weighted_error_ to fit size of current mini_batch.
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
  compute_mean_squared_error_derivative_kernel<<<grid, block>>>(
      d_error, d_output, d_target_output, error_.size());
  
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

void SigmoidOutputLayer::UpdateGpu(double learning_rate) {
  if (learning_rate <= 0)
    throw std::invalid_argument("learning_rate has to be a positive number.");

  GpuAllocationManager manager;
  double* d_weights = (double*)manager.AllocateDevice(
      weights_.size() * sizeof(double));
  double* d_velocity = (double*)manager.AllocateDevice(
      velocity_.size() * sizeof(double));

  cudaError_t cuda_status;
  // Copying data to gpu memory.
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
                                 learning_rate, num_training_examples_);

  // Resetting count of training examples after update.
  num_training_examples_ = 0;

  // Moving results to host.
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
