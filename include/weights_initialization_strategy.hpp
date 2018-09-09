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

#ifndef NEURALNET_INCLUDE_WEIGHTSINITIALIZATIONSTRATEGY_HPP_
#define NEURALNET_INCLUDE_WEIGHTSINITIALIZATIONSTRATEGY_HPP_

namespace neuralnet {

/**
 * @brief Weights initialization interface implementing strategy design pattern.
 */
class WeightsInitializationStrategy {
 public:
  virtual ~WeightsInitializationStrategy(){};
  /**
   * @brief Generates weight of connection.
   *
   * @param num_inputs  Number of inputs to the layer discounting bias.
   * @param num_neurons Number of neurons in layer for which we are generating
   *                    weights.
   */
  virtual double GetWeight(int num_inputs, int num_neurons) = 0;
};

}  // namespace neuralnet

#endif  // NEURALNET_INCLUDE_WEIGHTSINITIALIZATIONSTRATEGY_HPP_
