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

#ifndef NEURALNET_INCLUDE_LAYERS_SIGMOID_OUTPUT_LAYER_HPP_
#define NEURALNET_INCLUDE_LAYERS_SIGMOID_OUTPUT_LAYER_HPP_

#include "layer.hpp"

#include <string>
#include <vector>

namespace neuralnet {

/**
 * @brief Output layer with Sigmoid activation function and mean squared error
 * cost function.
 *
 * Sigmoid function is defined as follows:
 * \f$S(x) = \frac{1}{{1 + e^{-x}}}\f$ ,
 *
 * whereas mean squared error is defined as:
 * \f$MSE = \frac{1}{2n}\sum_{1=1}^{n} (Y_{i} - \hat{Y_{i}})^2\f$ .
 */
class SigmoidOutputLayer : public Layer {
 public:
  SigmoidOutputLayer();
  ~SigmoidOutputLayer(){};

  /* @brief Computes mean squared error.
   *
   * Mean squared error is defined as:
   * \f$MSE = \frac{1}{2n}\sum_{1=1}^{n} (Y_{i} - \hat{Y_{i}})^2\f$ .
   *
   * This functions accepts mini-batch laid in matrix columns stored in a
   * vector using row major ordering.
   *
   * @param target_output Correct output to most recent minibatch of input.
   */
  double GetLoss(const std::vector<double>& target_output);

 protected:
  /**
   * @brief Initializes weights to n * sqrt(1 / #inputs), where n is sampled
   *        from N(0, 1).
   */
  void InitializeWeights();

  /**
   * @brief Given input computes neurons' activations and applies sigmoid
   *        function using cpu.
   *
   * Sigmoid function is defined as:
   *   \f$S(x) = \frac{1}{{1 + e^{-x}}}\f$ ,
   *
   * Writes to activation_ and output_. This functions accepts mini-batch laid
   * in matrix columns stored in a vector using row major ordering and computes
   * a mini-batch of output.
   *
   * @param input Layer's input.
   */
  void ForwardPropCpu(const std::vector<double>& input);

  /**
   * @brief Given input computes neurons' activations and applies sigmoid
   *        function using cpu.
   *
   * Sigmoid function is defined as:
   *   \f$S(x) = \frac{1}{{1 + e^{-x}}}\f$ ,
   *
   * Writes to output_. This functions accepts mini-batch laid in matrix columns
   * stored in a vector using row major ordering and computes a mini-batch of
   * output.
   *
   * @param input Layer's input.
   */
  void ForwardPropGpu(const std::vector<double>& input);

  /**
   * @brief Computes velocity of weights and weighted error of this layer using
   *        cpu.
   *
   * Backpropagates through this layer using mean squared error as a cost
   * function. MSE is defined as:
   *   \f$MSE = \frac{1}{2n}\sum_{1=1}^{n} (Y_{i} - \hat{Y_{i}})^2\f$.
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
   * Backpropagates through this layer using mean squared error as a cost
   * function. MSE is defined as:
   *   \f$MSE = \frac{1}{2n}\sum_{1=1}^{n} (Y_{i} - \hat{Y_{i}})^2\f$.
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

  /**
   * @brief Updates weights with momentum accumulated across forward-backward
   *        passes using cpu.
   *
   * Weights are updated by substracting velocity multiplied by learning_rate
   * and divided by number of training samples from current weight.
   *
   * @param learning_rate         Speed with which this layer accepts new
   *                              information. Pick this value carefully
   *                              because if it's too big layer can diverge,
   *                              and if it's too small it will take a long
   *                              time to converge.
   * @throws std::invalid_argument If learning_rate is not positive.
   */
  void UpdateCpu(double learning_rate);

  /**
   * @brief Updates weights with momentum accumulated across forward-backward
   *        passes using gpu.
   *
   * Weights are updated by substracting velocity multiplied by learning_rate
   * and divided by number of training samples from current weight.
   *
   * @param learning_rate         Speed with which this layer accepts new
   *                              information. Pick this value carefully
   *                              because if it's too big layer can diverge,
   *                              and if it's too small it will take a long
   *                              time to converge.
   * @throws std::invalid_argument If learning_rate is not positive.
   */
  void UpdateGpu(double learning_rate);

 private:
  int num_training_examples_ = 0;

  void ComputeErrorCpu(const std::vector<double>& target_outputs);
};

}  // namespace neuralnet

#endif  // NEURALNET_INCLUDE_LAYERS_SIGMOID_OUTPUT_LAYER_HPP_
