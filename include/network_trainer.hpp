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

#ifndef NEURALNET_INCLUDE_NETWORKTRAINER_HPP_
#define NEURALNET_INCLUDE_NETWORKTRAINER_HPP_

#include <vector>

#include "net.hpp"

namespace neuralnet {

/**
 * @brief Network's training wrapper.
 *
 * This class menages network's training on provided dataset.
 */
class NetworkTrainer {
 public:
  NetworkTrainer(Net& neural_network) : neural_network_(neural_network){};
  virtual ~NetworkTrainer(){};
  /**
   * @brief Trains the network with provided training data and labels.
   *
   * Trains the network with provided training data and labels using mini-batch
   * gradient descent with momentum. This function accepts input in form of
   * mini-batches with size being multiples of size of the network's input
   * layer, each input of a mini-batch laid in columns of a matrix stored as a
   * vector using row major ordering. Each mini-batch is propagated forward and
   * then backward through a network and then the network is updated.
   *
   * @param inputs         Vector of mini-batches of training data.
   * @param target_outputs Vector of mini-batches of labels corresponding to the
   *                       training data.
   * @param learning_rate  Speed with which this network accepts new
   *                       information. Pick this value carefully because if
   *                       it's too big network can diverge, and if it's too
   *                       small it will take a long time time to converge.
   * @param momentum       Momentum coefficient for velocity calculation in
   *                       momentum gradient descent algorithm. This value is
   *                       usually 0.9 or 0.99.
   * @param num_epochs     Number of forward-backward passes through the entire
   *                       training data.
   * @throws std::invalid_argument("learning_rate has to be a positive number.")
   * @throws std::invalid_argument("momentum coefficient should have a value
   * between 0 and 1.")
   */
  virtual void Train(const std::vector<std::vector<double>>& inputs,
                     const std::vector<std::vector<double>>& target_outputs,
                     double learning_rate, double momentum, int num_epochs);

  /**
   * @brief Tests network's prediction accuracy on test data with 'one hot'
   *        labels.
   *
   * Labels to each input have to be a 'one hot' vectors. For each input trainer
   * propagates it forward and schecks which neuron of network's output layer
   * returned the largest value and compares it with provided label. If the
   * label corresponding to that neuron has value 1, output is counted as
   * correct. This function accepts input in form of mini-batches with size
   * being multiples of size of the network's input layer, each input of a
   * mini-batch laid in columns of a matrix stored as a vector using row major
   * ordering.
   *
   * @param inputs Vector of mini-batches of test data.
   * @param labels Vector of mini-batches of labels corresponding to test data.
   */
  virtual void Test(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& labels);

 protected:
  Net& neural_network_;
  /**< Network being a subject to training. */
};

}  // namespace neuralnet
#endif  // !NEURALNET_INCLUDE_NETWORKTRAINER_HPP_
