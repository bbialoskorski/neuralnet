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
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

#include <omp.h>

namespace neuralnet {

SoftmaxOutputLayer::SoftmaxOutputLayer() { type_ = "SoftmaxOutputLayer"; }

void SoftmaxOutputLayer::InitializeWeights() {
  weights_.assign(weights_.size(), 0.0);
}

void SoftmaxOutputLayer::StableSoftmax() {
  int mini_batch_size = activation_.size() / num_neurons_;
  // Resizing output to fit size of current mini-batch.
  output_.resize(num_neurons_ * mini_batch_size);

#pragma omp parallel for
  for (int column = 0; column < mini_batch_size; ++column) {
    double max_activation = std::numeric_limits<double>::lowest();

    for (int row = 0; row < num_neurons_; ++row) {
      double val = activation_[row * mini_batch_size + column];

      if (val > max_activation) {
        max_activation = val;
      }
    }

    double exponents_sum = 0.0;

    for (int row = 0; row < num_neurons_; ++row) {
      // Squishing activations so their exponents don't overflow by
      // substracting max activation from each argument.
      double exponent = std::exp(activation_[row * mini_batch_size + column] -
                                 max_activation);

      output_[row * mini_batch_size + column] = exponent;
      exponents_sum += exponent;
    }

    for (int row = 0; row < num_neurons_; ++row) {
      // Computing softmax function values according to following formula:
      // f(x[i]) = exp(x[i]) / sum(exp(x)) where x is an arguments column
      // vector.
      output_[row * mini_batch_size + column] /= exponents_sum;
    }
  }
}

void SoftmaxOutputLayer::ForwardPropCpu(const std::vector<double>& input) {
  ComputeActivationCpu(input);
  // Applying numerically stable softmax to computed activation.
  StableSoftmax();
}

void SoftmaxOutputLayer::CrossEntropyDerivative(
    const std::vector<double>& target_outputs) {
  int batch_size = target_outputs.size() / num_neurons_;

  error_.resize(num_neurons_ * batch_size);

#pragma omp parallel
  for (int index = 0; index < output_.size(); ++index)
    error_[index] = output_[index] - target_outputs[index];
}

void SoftmaxOutputLayer::BackPropCpu(
    const std::vector<double>& target_outputs,
    const std::vector<double>& prev_layer_output, double momentum) {
  if (momentum <= 0 || momentum >= 1)
    throw std::invalid_argument(
        "momentum coefficient should have a value\
 between 0 and 1.");

  CrossEntropyDerivative(target_outputs);
  ComputeVelocityCpu(prev_layer_output, momentum);
  ComputeWeightedErrorCpu();
}

double SoftmaxOutputLayer::GetLoss(const std::vector<double>& target_outputs) {
  double loss = 0.0;
  for (int i = 0; i < target_outputs.size(); ++i)
    loss -= target_outputs[i] * std::log(output_[i]);
  return loss;
}

}  // namespace neuralnet
