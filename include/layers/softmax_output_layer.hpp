/* Copyright (c) 2019 Bartosz Białoskórski

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

#ifndef NEURALNET_INCLUDE_LAYERS_SOFTMAX_OUTPUT_LAYER_HPP_
#define NEURALNET_INCLUDE_LAYERS_SOFTMAX_OUTPUT_LAYER_HPP_

#include "layer.hpp"

#include <string>
#include <vector>

namespace neuralnet {

/**
 * @brief Output layer with Softmax activation function and cross entropy loss
 *        function.
 *
 * Softmax layer returns probability distribution.
 *
 * Softmax function is defined as follows:
 * \f$softmax(a_{j}) = \frac{e^{a_{j}}}{\sum_{i}e^{a_{i}}} \f$ ,
 * whereas cross entropy is defined as:
 * \f$L = -y * \log{p} \f$ ,
 * where:
 *  y: vector of desired outputs
 *  p: vector of actual outputs
 */
class SoftmaxOutputLayer : public Layer {
 public:
  SoftmaxOutputLayer();

  /**
   * @brief Constructs SoftmaxOutputLayer using specified gpu memory
   * allocation manager.
   *
   * @param gpu_alloc_manager GpuAllocationManager managing allocations of
   * gpu device memory.
   */
  SoftmaxOutputLayer(std::shared_ptr<GpuAllocationManager> gpu_alloc_manager);
  ~SoftmaxOutputLayer(){};

  /**
   * @brief Computes cross entropy loss.
   *
   * Cross entropy is defined as:
   * \f$L = -y * \log{p} \f$ ,
   * where:
   *  y: vector of desired outputs
   *  p: vector of actual outputs
   *
   * This functions accepts mini-batch laid in matrix columns stored in a
   * vector using row major ordering.
   *
   * @param target_output Correct output to most recent minibatch of input.
   */
  double GetLoss(const std::vector<double>& target_output);

 protected:
  /** @brief Initializes all weights to 0. */
  void InitializeWeights();

  /**
   * @brief Given input computes neurons' activations and applies softmax
   *        function using cpu.
   *
   * Softmax function is defined as follows:
   *   \f$softmax(a_{j}) = \frac{e^{a_{j}}}{\sum_{i}e^{a_{i}}} \f$.
   *
   * Writes to activation_ and output_. This functions accepts mini-batch laid
   * in matrix columns stored in a vector using row major ordering and computes
   * a mini-batch of output.
   *
   * @param input Layer's input.
   */
  void ForwardPropCpu(const std::vector<double>& input);

  /**
   * @brief Given input computes neurons' activations and applies softmax
   *        function using gpu.
   *
   * Softmax function is defined as follows:
   *   \f$softmax(a_{j}) = \frac{e^{a_{j}}}{\sum_{i}e^{a_{i}}} \f$.
   *
   * Writes to activation_ and output_. This functions accepts mini-batch laid
   * in matrix columns stored in a vector using row major ordering and computes
   * a mini-batch of output.
   *
   * @param input Layer's input.
   */
  void ForwardPropGpu(const std::vector<double>& input);

  /**
   * @brief Computes velocity of weights and weighted error of this layer using
   *        cpu.
   *
   * Backpropagates through this layer using cross entropy as a loss function.
   * Cross entropy is defined as:
   *   \f$L = -y * \log{p} \f$ ,
   * where:
   *  y: vector of desired outputs
   *  p: vector of actual outputs
   *
   * This function writes to error_, velocity_ and weighted_error_.
   * This functions accepts mini-batch laid in matrix columns stored in a
   * vector using row major ordering and computes a mini-batch of output.
   *
   * @param target_output     Correct output to most recent minibatch of input.
   * @param prev_layer_output Return value of forward step in previous layer.
   * @param momentum          Momentum coefficient for velocity calculation.
   * @throws std::invalid_argument If momentum value lies outside of (0, 1) set.
   */
  void BackPropCpu(const std::vector<double>& target_output,
                   const std::vector<double>& prev_layer_output,
                   double momentum);

  /**
   * @brief Computes velocity of weights and weighted error of this layer using
   *        gpu.
   *
   * Backpropagates through this layer using cross entropy as a loss function.
   * Cross entropy is defined as:
   *   \f$L = -y * \log{p} \f$ ,
   * where:
   *  y: vector of desired outputs
   *  p: vector of actual outputs
   *
   * This function writes to error_, velocity_ and weighted_error_.
   * This functions accepts mini-batch laid in matrix columns stored in a
   * vector using row major ordering and computes a mini-batch of output.
   *
   * @param target_output     Correct output to most recent minibatch of input.
   * @param prev_layer_output Return value of forward step in previous layer.
   * @param momentum          Momentum coefficient for velocity calculation.
   * @throws std::invalid_argument If momentum value lies outside of (0, 1) set.
   */
  void BackPropGpu(const std::vector<double>& target_output,
                   const std::vector<double>& prev_layer_output,
                   double momentum);

 private:
  void StableSoftmax();
  void CrossEntropyDerivative(const std::vector<double>& target_output);
};

}  // namespace neuralnet

#endif  // NEURALNET_INCLUDE_LAYERS_SOFTMAX_OUTPUT_LAYER_HPP_
