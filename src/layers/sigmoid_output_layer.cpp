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
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

#include <omp.h>

namespace neuralnet {

SigmoidOutputLayer::SigmoidOutputLayer() { type_ = "SigmoidOutputLayer"; }

void SigmoidOutputLayer::InitializeWeights() {
  std::default_random_engine random_number_generator;
  random_number_generator.seed(
      std::chrono::system_clock::now().time_since_epoch().count());
  std::normal_distribution<double> normal_distribution(0.0, 1.0);
  // Initial weights for layer with sigmoid activation function are
  // set according to the following formula:
  //
  // w[i][j] = d * sqrt(1 / r[k - 1]),
  //
  // where:
  // w[i][j]: weight of connection between neuron i in current layer
  //   and neuron j in previous layer
  // r[k - 1]: number of neurons in previous layer i.e. number of
  //   inputs to each neuron in this layer
  // d: random double sampled from a univariate Gaussian distribution
  //   with mean 0 and variance 1.
  //
  // 'weights_' is a matrix with #num_neurons_ rows and #num_inputs_ columns,
  // where bias node is taken into account in num_inputs_ but not in
  // num_neurons_.
  for (double& weight : weights_)
    weight = normal_distribution(random_number_generator) *
             std::sqrt(1.0 / (double)num_inputs_);
}

void SigmoidOutputLayer::ForwardPropCpu(const std::vector<double>& input) {
  ComputeActivationCpu(input);
  int mini_batch_size = input.size() / (num_inputs_ - 1);

  // Resizing output to fit size of current mini-batch.
  output_.resize(num_neurons_ * mini_batch_size);

  // Calculating sigmoid function value for each activation.
  // S(x) = 1 / (1 + exp(-x))
#pragma omp parallel for
  for (int i = 0; i < activation_.size(); ++i) {
    output_[i] = 1.0 / (1.0 + std::exp(-activation_[i]));
  }
}

void SigmoidOutputLayer::BackPropCpu(
    const std::vector<double>& target_output,
    const std::vector<double>& prev_layer_output, double momentum) {
  if (momentum <= 0 || momentum >= 1)
    throw std::invalid_argument(
        "momentum coefficient should have a value\
 between 0 and 1.");

  // Incrementing num_training_examples after back pass instead of forward pass
  // to maintain correctness in case of forward passes not being part of
  // training.
  int mini_batch_size = target_output.size() / num_neurons_;
  num_training_examples_ += mini_batch_size;

  ComputeErrorCpu(target_output);
  ComputeVelocityCpu(prev_layer_output, momentum);
  ComputeWeightedErrorCpu();
}

void SigmoidOutputLayer::UpdateCpu(double learning_rate) {
  if (learning_rate <= 0)
    throw std::invalid_argument("learning_rate has to be a positive number.");

#pragma omp parallel for
  for (int i = 0; i < weights_.size(); ++i) {
    weights_[i] -=
        learning_rate * velocity_[i] / (double)num_training_examples_;
  }

  // Resetting count of training examples after update.
  num_training_examples_ = 0;
}

void SigmoidOutputLayer::ComputeErrorCpu(
    const std::vector<double>& target_output) {
  // Sigmoid output layer uses mean squared error function.
  //
  // Calculating neuron's 'error' as a partial derivative of error
  // function of it's activation variable:
  //
  // dE/d(a[row]) = (y'[row] - y[row]) * g'(a[row]),
  //
  // where:
  // g': derivative of activation function
  // a[row]: activation of neuron row in current layer
  // y'[row]: output of neuron row in current layer
  // y[row]: expected output for neuron row in current layer.
  int batch_size = target_output.size() / num_neurons_;

  error_.resize(num_neurons_ * batch_size);

#pragma omp parallel for
  for (int i = 0; i < error_.size(); ++i) {
    double difference = output_[i] - target_output[i];
    error_[i] = difference * output_[i] * (1.0 - output_[i]);
  }
}

double SigmoidOutputLayer::GetLoss(const std::vector<double>& target_output) {
  double loss = 0.0;
  int batch_size = target_output.size() / num_neurons_;

  for (int i = 0; i < output_.size(); ++i)
    loss += std::pow(output_[i] - target_output[i], 2.0) / batch_size;

  return loss;
}

}  // namespace neuralnet
