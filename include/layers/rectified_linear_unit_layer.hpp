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

#ifndef NEURALNET_INCLUDE_LAYERS_RECTIFIED_LINEAR_UNIT_LAYER_HPP_
#define NEURALNET_INCLUDE_LAYERS_RECTIFIED_LINEAR_UNIT_LAYER_HPP_

#include "layer.hpp"

#include <string>
#include <vector>

namespace neuralnet {

/**
 * @brief Hidden layer with rectifier activation function.
 *
 * Rectifier is defined as:
 * R(x) = max(0, x).
 */
class ReLuLayer : public Layer {
 public:
  ReLuLayer();
  ~ReLuLayer(){};

 protected:
  /**
   * @brief Initializes weights using normal distribution with 0 mean and
   *        sqrt(2 / #inputs) standard deviation.
   */
  void InitializeWeights();

  /**
   * @brief Given input computes neurons' activations and applies rectifier
   *        function using cpu.
   *
   * Rectifier is defined as: R(x) = max(0, x).
   *
   * Writes to activation_ and output_. This functions accepts mini-batch laid
   * in matrix columns stored in a vector using row major ordering and computes
   * a mini-batch of output.
   *
   * @param input Layer's input.
   */
  void ForwardPropCpu(const std::vector<double>& input);

  /**
   * @brief Given input computes neurons' activations and applies rectifier
   *        function using gpu.
   *
   * Rectifier is defined as: R(x) = max(0, x).
   *
   * Writes to activation_ and output_. This functions accepts mini-batch laid
   * in matrix columns stored in a vector using row major ordering and computes
   * a mini-batch of output.
   *
   * @param input Layer's input.
   */
  void ForwardPropGpu(const std::vector<double>& input);
  void BackPropCpu(const std::vector<double>& weighted_error_top,
                   const std::vector<double>& prev_layer_output,
                   double momentum);
  void BackPropGpu(const std::vector<double>& weighted_error_top,
                   const std::vector<double>& prev_layer_output,
                   double momentum);

 private:
  void ComputeErrorCpu(const std::vector<double>& weighted_error_top);
};

}  // namespace neuralnet

#endif  // NEURALNET_INCLUDE_LAYERS_RECTIFIED_LINEAR_UNIT_LAYER_HPP_
