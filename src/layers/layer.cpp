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

#include <assert.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <new>

#include <omp.h>

namespace neuralnet {

void Layer::Initialize(int num_inputs, int num_neurons) {
  if (num_inputs < 1)
    throw std::invalid_argument("num_inputs has to be positive.");
  if (num_neurons < 1)
    throw std::invalid_argument("num_neurons has to be positive.");

  Reshape(num_inputs, num_neurons);
  InitializeWeights();
}

void Layer::Initialize(int num_inputs, int num_neurons,
                       WeightsInitializationStrategy& generator) {
  if (num_inputs < 1)
    throw std::invalid_argument("num_inputs has to be positive.");
  if (num_neurons < 1)
    throw std::invalid_argument("num_neurons has to be positive.");

  Reshape(num_inputs, num_neurons);
  InitializeWeights(generator);
}

void Layer::Reshape(int num_inputs, int num_neurons) {
  if (num_inputs < 1)
    throw std::invalid_argument("num_inputs has to be positive.");
  if (num_neurons < 1)
    throw std::invalid_argument("num_neurons has to be positive.");

  num_inputs_ = num_inputs + 1;
  num_neurons_ = num_neurons;
  weights_.resize(num_neurons_ * num_inputs_);
  velocity_.resize(num_neurons_ * num_inputs_);
  velocity_.assign(velocity_.size(), 0.0);
  error_.resize(num_neurons);
  activation_.resize(num_neurons);
  output_.resize(num_neurons);
  weighted_error_.resize(num_inputs_ - 1);
}

void Layer::InitializeWeights(WeightsInitializationStrategy& generator) {
  for (double& weight : weights_)
    weight = generator.GetWeight(num_inputs_, num_neurons_);
}

void Layer::ComputeActivationCpu(const std::vector<double>& input) {
  int mini_batch_size = input.size() / (num_inputs_ - 1);
  int tile_dim =
      64 / sizeof(double);  // 64 bytes is most common cache line size.
  // Resizing activation to fit size of current mini-batch.
  activation_.resize(num_neurons_ * mini_batch_size);
  activation_.assign(activation_.size(), 0.0);

#pragma omp parallel
  {
    // Multiplying weights matrix without bias column by input matrix, result
    // matrix being activation.
#pragma omp for  // collapse(2)
    // Tiled Matrix multiplication A * B = C. We are sliding a tile_dim by
    // tile_dim tile on top of result matrix C. Each tile calculates
    // corresponding elements of C by sliding tile_dim by tile_dim tiles along
    // A's columns and B's rows and multiplying them accumulating results across
    // different tile positions.
    for (int c_lower_y = 0; c_lower_y < num_neurons_; c_lower_y += tile_dim) {
      // Position of tiles is bounded from bottom and top in x and y dimentions.
      int c_upper_y = std::min(c_lower_y + tile_dim, num_neurons_);

      for (int c_lower_x = 0; c_lower_x < mini_batch_size;
           c_lower_x += tile_dim) {
        int c_upper_x = std::min(c_lower_x + tile_dim, mini_batch_size);

        for (int a_lower_x = 0; a_lower_x < num_inputs_ - 1;
             a_lower_x += tile_dim) {
          int a_upper_x = std::min(a_lower_x + tile_dim, num_inputs_ - 1);

          for (int a_y = c_lower_y; a_y < c_upper_y; ++a_y) {
            // Multiplying corresponding tiles of A and B.
            for (int b_x = c_lower_x; b_x < c_upper_x; ++b_x) {
              double product = 0.0;

              for (int a_x = a_lower_x; a_x < a_upper_x; ++a_x) {
                product += weights_[a_y * num_inputs_ + a_x] *
                           input[a_x * mini_batch_size + b_x];
              }
              activation_[a_y * mini_batch_size + b_x] += product;
            }
          }
        }
      }
    }
    // Adding bias to activations.
#pragma omp for
    for (int a_y = 0; a_y < num_neurons_; ++a_y) {
      double bias = weights_[(a_y + 1) * num_inputs_ - 1];

      for (int c_x = 0; c_x < mini_batch_size; ++c_x) {
        activation_[a_y * mini_batch_size + c_x] += bias;
      }
    }
  }
}

void Layer::ComputeWeightedErrorCpu() {
  int mini_batch_size = error_.size() / num_neurons_;
  // Resizing weighted_error to fit size of current mini batch.
  weighted_error_.resize((num_inputs_ - 1) * mini_batch_size);
  weighted_error_.assign(weighted_error_.size(), 0.0);
  std::vector<double> transposed_weights(weights_.size());
  int tile_dim = 64 / sizeof(double);
#pragma omp parallel
  {
// Transposing weights matrix.
#pragma omp for
    for (int i = 0; i < weights_.size(); ++i) {
      int row = i / num_inputs_;
      int col = i - row * num_inputs_;
      transposed_weights[col * num_neurons_ + row] = weights_[i];
    }

    // Multiplying transposed weights matrix without last row corresponding to
    // bias by error matrix resulting in matrix of weighted errors.
#pragma omp for
    // Tiled Matrix multiplication A * B = C. We are sliding a tile_dim by
    // tile_dim tile on top of result matrix C. Each tile calculates
    // corresponding elements of C by sliding tile_dim by tile_dim tiles along
    // A's columns and B's rows and multiplying them accumulating results across
    // different tile positions.
    for (int c_lower_y = 0; c_lower_y < num_inputs_ - 1;
         c_lower_y += tile_dim) {
      // Position of tiles is bounded from bottom and top in x and y dimentions.
      int c_upper_y = std::min(c_lower_y + tile_dim, num_inputs_ - 1);

      for (int c_lower_x = 0; c_lower_x < mini_batch_size;
           c_lower_x += tile_dim) {
        int c_upper_x = std::min(c_lower_x + tile_dim, mini_batch_size);

        for (int a_lower_x = 0; a_lower_x < num_neurons_;
             a_lower_x += tile_dim) {
          int a_upper_x = std::min(a_lower_x + tile_dim, num_neurons_);

          for (int a_y = c_lower_y; a_y < c_upper_y; ++a_y) {
            // Multiplying corresponding tiles of A and B.
            for (int b_x = c_lower_x; b_x < c_upper_x; ++b_x) {
              double product = 0.0;

              for (int a_x = a_lower_x; a_x < a_upper_x; ++a_x) {
                product += transposed_weights[a_y * num_neurons_ + a_x] *
                           error_[a_x * mini_batch_size + b_x];
              }
              weighted_error_[a_y * mini_batch_size + b_x] += product;
            }
          }
        }
      }
    }
  }
}

void Layer::UpdateCpu(double learning_rate) {
  if (learning_rate <= 0)
    throw std::invalid_argument("learning_rate has to be a positive number.");

#pragma omp parallel for
  for (int i = 0; i < weights_.size(); ++i) {
    weights_[i] -= learning_rate * velocity_[i];
  }
}

void Layer::ComputeVelocityCpu(const std::vector<double>& prev_layer_output,
                               double momentum) {
  if (momentum <= 0 || momentum >= 1)
    throw std::invalid_argument(
        "momentum coefficient should have a value\
 between 0 and 1.");

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
  int mini_batch_size = prev_layer_output.size() / (num_inputs_ - 1);
#pragma omp parallel for
  for (int i = 0; i < velocity_.size(); ++i) {
    int row = i / num_inputs_;
    int col = i - row * num_inputs_;
    // Calculating gradient on the current mini-batch.
    double gradient = 0.0;
    for (int batch_index = 0; batch_index < mini_batch_size; ++batch_index) {
      if (col != num_inputs_ - 1) {
        gradient += error_[row * mini_batch_size + batch_index] *
                    prev_layer_output[col * mini_batch_size + batch_index];
      } else {
        // Bias input weight.
        gradient += error_[row * mini_batch_size + batch_index];
      }
    }
    // Momentum backpropagation.
    // velocity(t) = momentum * velocity(t - 1) + (1.0 - momentum) * dE/dW,
    //
    // where:
    //  t is a time step,
    //  dE/dW is a weights gradient calculated on the current mini-batch.
    velocity_[i] = velocity_[i] * momentum + (1.0 - momentum) * gradient;
  }
}

}  // namespace neuralnet
