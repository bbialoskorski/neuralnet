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

#include "layers/rectified_linear_unit_layer.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <thread>

#include <omp.h>

#include "gpu_stack_allocator.hpp"

namespace neuralnet {

ReLuLayer::ReLuLayer() {
  type_ = "ReLuLayer";
  gpu_alloc_manager_ = std::make_shared<GpuAllocationManager>();
}

ReLuLayer::ReLuLayer(std::shared_ptr<GpuAllocationManager> gpu_alloc_manager) {
  type_ = "ReLuLayer";
  gpu_alloc_manager_ = gpu_alloc_manager;
}

void ReLuLayer::InitializeWeights() {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(
      0.0, sqrt(2.0 / (double)num_inputs_));

  for (double& weight : weights_) weight = distribution(generator);
}

void ReLuLayer::ForwardPropCpu(const std::vector<double>& input) {
  ComputeActivationCpu(input);
  int mini_batch_size = input.size() / (num_inputs_ - 1);

  // Resizing output to fit size of current mini-batch.
  output_.resize(num_neurons_ * mini_batch_size);

  // Calculating outputs by using rectifier on previously computed activation.
#pragma omp parallel for
  for (int i = 0; i < activation_.size(); ++i)
    output_[i] = std::max(0.0, activation_[i]);
}

void ReLuLayer::ComputeErrorCpu(const std::vector<double>& weighted_error_top) {
  int mini_batch_size = weighted_error_top.size() / num_neurons_;

  // Resizing error to fit size of current mini-batch.
  error_.resize(num_neurons_ * mini_batch_size);

#pragma omp parallel for
  for (int i = 0; i < error_.size(); ++i)
    error_[i] = activation_[i] > 0.0 ? weighted_error_top[i] : 0.0;
}

void ReLuLayer::BackPropCpu(const std::vector<double>& weighted_error_top,
                            const std::vector<double>& prev_layer_output,
                            double momentum) {
  if (momentum <= 0 || momentum >= 1)
    throw std::invalid_argument(
        "momentum coefficient should have a value\
 between 0 and 1.");

  ComputeErrorCpu(weighted_error_top);
  ComputeVelocityCpu(prev_layer_output, momentum);
  ComputeWeightedErrorCpu();
}

}  // namespace neuralnet
