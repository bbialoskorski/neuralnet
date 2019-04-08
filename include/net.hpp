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

#ifndef NEURALNET_INCLUDE_NET_HPP_
#define NEURALNET_INCLUDE_NET_HPP_

#include <string>
#include <utility>
#include <vector>

#include "layers/layer.hpp"
#include "net_io_handler.hpp"

namespace neuralnet {

/**
 * @brief Abstraction of feedforward neural network.
 *
 * Net assembles layers together and abstracts out forward/backward propagation
 * and updates to the level of a single entity. Network handles mini-batches of
 * input laid in matrix columns and stored in a vector using row major ordering.
 * Network created with default constructor uses NetJsonIoHandler for Save and
 * Load.
 */
class Net {
 public:
  /**
   * @brief Constructs an empty network.
   *
   * Network created with default constructor uses NetJsonIoHandler for Save and
   * Load.
   *
   * @param input_layer_size Size of input layer.
   * @param gpu_flag         Flag controlling whether gpu implementation is used
   *                         in computation.
   * @throws std::invalid_argument If input_layer_size is a positive number.
   */
  Net(int input_layer_size, bool gpu_flag);

  /**
   * @brief Constructs an empty network.
   *
   * @param input_layer_size Size of input layer.
   * @param gpu_flag         Flag controlling whether gpu implementation is used
   *                         in computation.
   * @param io_handler       Shared pointer to object of class derived from
   *                         NetIoHandler which is used by Save() and Load()
   *                         functions.
   * @throws std::invalid_argument If input_layer_size is a positive number.
   */
  Net(int input_layer_size, bool gpu_flag,
      std::shared_ptr<NetIoHandler> io_handler);
  virtual ~Net();

  /**
   * @brief Given target output of most recent forward pass computes loss of
   *        output layer.
   *
   * @param target_output Correct output to most recent minibatch of input.
   */
  double GetLoss(const std::vector<double>& target_output);

  /** @brief Exports network to a file. */
  void Save(std::string file_path);

  /** @brief Imports network to a file. */
  void Load(std::string file_path);

  /** @brief Returns number of inputs to a network. */
  int GetNumInputs() const { return input_layer_size_; }

  /** @brief Returns number of layer's discounting input layer */
  int GetNumLayers() const { return layers_.size(); }

  /** @brief Returns network's layers. */
  std::vector<std::shared_ptr<Layer>> GetLayers() { return layers_; }

  /**
   * @brief Adds layer to the end of the network and initializes its weights
   *        with layer specific default algorithm.
   *
   * @param layer Shared pointer to the layer you want to add.
   * @num_neurons Desired size of the layer discounting bias neuron.
   * @throws std::invalid_argument If num_neurons is not positive.
   **/
  virtual void AddLayer(std::shared_ptr<Layer> layer, int num_neurons);

  /**
   * @brief Adds layer to the end of the network and initializes its weights
   *        with provided strategy.
   *
   * @param layer         Shared pointer to the layer you want to add.
   * @param num_neurons   Desired size of the layer discounting bias neuron.
   * @param init_strategy Concrete strategy for weights initialization.
   * @throws std::invalid_argument If num_neurons is not positive.
   */
  virtual void AddLayer(std::shared_ptr<Layer> layer, int num_neurons,
                        WeightsInitializationStrategy& init_strategy);

  /** @brief Sets flag controlling whether gpu implementation is used.*/
  void SetGpuFlag();

  /** @brief Clears flag controlling whether gpu implementation is used. */
  void ClearGpuFlag();

  /**
   * @brief Propagates input forward through all layers.
   *
   * Network's input is fed to the first layer and each layer's output is fed to
   * the next layer.
   * This functions accepts mini-batch laid in matrix columns stored in a vector
   * using row major ordering and returns mini-batch of output.
   *
   * @param input Network's input.
   * @returns Output of last layer in the network.
   */
  virtual std::vector<double> ForwardProp(const std::vector<double>& input);

  /**
   * @brief Propagates backward through layers using mini-batch momentum
   *        backpropagation algorithm.
   *
   * For this function to work properly it has to be called after a single
   * forward pass. This functions accepts mini-batch laid in matrix columns
   * stored in a vector using row major ordering.
   *
   * @pre There should be a ForwardProp preceding BackProp.
   * @param target_output Output expected from the last ForwardProp call.
   * @param momentum      Momentum coefficient for velocity calculation.
   * @throws std::invalid_argument If momentum value lies outside of (0,1) set.
   */
  virtual void BackProp(const std::vector<double>& target_output,
                        double momentum);

  /**
   * @brief Updates weights of every layer using velocity accumulated across
   *        forward-backward passes.
   *
   * @param learning_rate Speed with which this network accepts new information.
   *                      Pick this value carefully because if it's too big
   *                      network can diverge, and if it's too small it will
   *                      take a long time time to converge.
   * @throws std::invalid_argument if learning_rate is not positive.
   */
  virtual void Update(double learning_rate);

  /** @brief Sets velocity of every layer to zeros. */
  void ResetVelocity();

 protected:
  bool gpu_flag_ = false;
  /**< Flag controlling whether gpu implementation is used. */
  int input_layer_size_;
  /**<
   * Number of double values that network accepts as a single input in forward
   * pass. Mini-batches need to have a size being a multiple of this number.
   */
  std::vector<std::shared_ptr<Layer>> layers_;
  /**< Layers composing neural network. */
  std::vector<std::vector<double>> output_;
  /**<
   *  Output of the network. Output to each input of mini-batch is laid in
   *  columns of a matrix stored as a vector using row major ordering.
   */
  std::shared_ptr<NetIoHandler> io_handler_;
  /**< Object handling import/export of the network to/from file. */

  friend class NetIoHandler;
};

}  // namespace neuralnet

#endif  // NEURALNET_INCLUDE_NET_HPP_
