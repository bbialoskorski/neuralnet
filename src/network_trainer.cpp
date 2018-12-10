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

#include "network_trainer.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace neuralnet {

void NetworkTrainer::Train(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& target_outputs,
    double learning_rate, double momentum, int num_epochs) {
  if (learning_rate <= 0)
    throw std::invalid_argument("learning_rate has to be a positive number.");
  if (momentum <= 0 || momentum >= 1)
    throw std::invalid_argument(
        "momentum coefficient should have a value\
 between 0 and 1.");

  std::chrono::high_resolution_clock::time_point start_time =
      std::chrono::high_resolution_clock::now();

  for (int epoch = 0; epoch < num_epochs; epoch++) {
    double loss = 0.0;

    for (int mini_batch = 0; mini_batch < inputs.size(); ++mini_batch) {
      neural_network_.ForwardProp(inputs[mini_batch]);
      neural_network_.BackProp(target_outputs[mini_batch], momentum);
      loss += neural_network_.GetLoss(target_outputs[mini_batch]);
      neural_network_.Update(learning_rate);
    }

    std::cout << "Epoch " << epoch << " out of " << num_epochs
              << " completed. Loss: " << loss << std::endl;

    neural_network_.ResetVelocity();
  }

  int num_samples = 0;
  int input_size = neural_network_.GetNumInputs();

  std::for_each(inputs.begin(), inputs.end(),
                [&num_samples, input_size](std::vector<double> in) {
                  num_samples += in.size() / input_size;
                });

  std::chrono::high_resolution_clock::time_point finish_time =
      std::chrono::high_resolution_clock::now();

  std::cout << "Network training complete! " << num_epochs
            << " epochs elapsed with learning rate " << learning_rate << " on "
            << num_samples << " training samples in "
            << std::chrono::duration_cast<std::chrono::seconds>(finish_time -
                                                                start_time)
                   .count()
            << " seconds." << std::endl;
}

void NetworkTrainer::Test(const std::vector<std::vector<double>>& inputs,
                          const std::vector<std::vector<double>>& labels) {
  int input_size = neural_network_.GetNumInputs();
  int batch_size = inputs[0].size() / input_size;
  int correct_count = 0;

  std::vector<double> output;

  for (int input_index = 0; input_index < inputs.size(); ++input_index) {
    output = neural_network_.ForwardProp(inputs[input_index]);
    int num_rows = output.size() / batch_size;

    for (int col = 0; col < batch_size; ++col) {
      double max_val = output[col];
      int max_row = 0;

      for (int row = 0; row < num_rows; ++row) {
        if (output[row * batch_size + col] > max_val) {
          max_val = output[row * batch_size + col];
          max_row = row;
        }
      }

      if (labels[input_index][max_row * batch_size + col] == 1.0)
        ++correct_count;
    }
  }

  std::cout << std::endl;
  std::cout << "Network achieved "
            << ((double)correct_count /
                ((double)inputs.size() * (double)batch_size)) *
                   100.0
            << " % accuracy on test data set." << std::endl;
}

}  // namespace neuralnet
