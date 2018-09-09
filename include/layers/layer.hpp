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

#ifndef NEURALNET_INCLUDE_LAYERS_LAYER_HPP_
#define NEURALNET_INCLUDE_LAYERS_LAYER_HPP_

#include <iostream>
#include <memory>
#include <vector>

#include "weights_initialization_strategy.hpp"

namespace neuralnet {

/**
 * @brief Abstract base for unit of computation of a network.
 *
 * Layer uses mini-batch backpropagation with momentum as a learning algorithm.
 * Classes deriving from Layer have to implement ForwardPropCpu,
 * ForwardPropGpu, BackPropCpu, BackPropGpu and InitializeWeights functions.
 * Forward and Backward functions have to handle mini-batches of arbitrary
 * size laid in matrix columns (stored in a vector using row major ordering).
 */
class Layer {
 public:
  virtual ~Layer() {}

  /** @brief Returns number of inputs including bias. */
  int GetNumInputs() { return num_inputs_; }

  /** @brief Returns number of neurons in this layer. */
  int GetSize() { return num_neurons_; }

  /** @brief Returns layer's class name. */
  std::string GetLayerType() { return type_; }

  /** @brief Returns vector of outputs from this layer's neurons. */
  std::vector<double> GetOutputs() { return output_; }

  /** @brief Returns layer's weights matrix. */
  std::vector<double> &GetWeights() { return weights_; }

  /** @brief Return's layer's loss value. */
  virtual double GetLoss(const std::vector<double> &target_output) {
    return 0.0;
  }

  /** @brief Sets flag controlling whether gpu implementation is used.*/
  void SetGpuFlag() { gpu_flag_ = true; };

  /** @brief Clears flag controlling whether gpu implementation is used. */
  void ClearGpuFlag() { gpu_flag_ = false; };

  /**
   * @brief Initializes layer by reshaping its components and generating
   *        initial weights using layer specific default algorithm.
   *
   * @param num_inputs  Number of neurons in previous layer discounting bias.
   * @param num_neurons Number of neurons in this layer discounting bias.
   * @throws std::invalid_argument If num_inputs or num_neurons is not positive.
   */
  virtual void Initialize(int num_inputs, int num_neurons);

  /**
   * @brief Initializes layer by reshaping its components and generating
   *        initial weights using provided strategy.
   *
   * @param num_inputs  Number of neurons in previous layer discounting bias.
   * @param num_neurons Number of neurons in this layer discounting bias.
   * @param generator   Concrete strategy for weights initialization.
   * @throws std::invalid_argument If num_inputs or num_neurons is not positive.
   */
  virtual void Initialize(int num_inputs, int num_neurons,
                          WeightsInitializationStrategy &generator);

  /**
   * @brief Activates this layer's neurons and returns output.
   *
   * This functions accepts mini-batch laid in matrix columns stored in a
   * vector using row major ordering and returns mini-batch of output.
   *
   * @param input Layer's input.
   * @returns     Output from this layer's neurons.
   */
  inline std::vector<double> ForwardProp(const std::vector<double> &input) {
    if (!gpu_flag_)
      ForwardPropCpu(input);
    else
      ForwardPropGpu(input);
    return output_;
  }

  /**
   * @brief Computes velocity of weights and returns weighted error of this
   *        layer.
   *
   * For this function to work properly it has to be called after a single
   * forward pass. This functions accepts mini-batch laid in matrix columns
   * stored in a vector using row major ordering and returns a mini-batch of
   * output.
   * Note that if this layer is an output layer then target output should be
   * passed as a weighted_error. In case this layer is first in network's
   * topology, network's input should be passed as a prev_layer_output.
   *
   * @pre                     There should be a ForwardProp preceding BackProp.
   * @param weighted_error    Weighted sum of succeeding layer's error for each
   *                          neuron in this layer with coefficients being
   *                          weights of connections with these neurons.
   * @param prev_layer_output Return value of forward step in previous layer.
   * @param momentum          Momentum coefficient for velocity calculation.
   * @returns                 Weighted sum of this layer's error for each input
   *                          neuron excluding bias.
   * @throws std::invalid_argument If momentum value lies outside of (0, 1) set.
   */
  inline std::vector<double> BackProp(
      const std::vector<double> &weighted_error,
      const std::vector<double> &prev_layer_output, double momentum) {
    if (momentum <= 0 || momentum >= 1)
      throw std::invalid_argument(
          "momentum coefficient should have a value\
 between 0 and 1.");

    if (!gpu_flag_) {
      BackPropCpu(weighted_error, prev_layer_output, momentum);
      ComputeWeightedErrorCpu();
    } else
      BackPropGpu(weighted_error, prev_layer_output, momentum);
    return weighted_error_;
  }

  /**
   * @brief Updates weights matrix using velocity accumulated across training
   *        samples.
   *
   * @param learning_rate         Speed with which this layer accepts new
   *                              information. Pick this value carefully
   *                              because if it's too big layer can diverge,
   *                              and if it's too small it will take a long
   *                              time to converge.
   * @throws std::invalid_argument If learning_rate is not positive.
   */
  inline void Update(double learning_rate) {
    if (learning_rate <= 0)
      throw std::invalid_argument("learning_rate has to be a positive number.");

    if (!gpu_flag_)
      UpdateCpu(learning_rate);
    else
      UpdateGpu(learning_rate);
  }

 protected:
  std::string type_;
  /**< String storing layer's class name. */
  bool gpu_flag_ = false;
  /**< Flag controlling whether gpu implementation is used. */
  int num_neurons_ = 1;
  /**< Number of neurons in this layer discounting bias. */
  int num_inputs_ = 2;
  /**< Number of inputs to each neuron in this layer taking into account bias.
   */
  std::vector<double> weights_;
  /**< Linear representation of weights matrix using row major ordering. */
  std::vector<double> velocity_;
  /**< Linear representation of velocity matrix using row major ordering. */
  std::vector<double> error_;
  /**< Error values for each neuron in this layer. */
  std::vector<double> activation_;
  /**<
   * Activation of neurons in this layer i.e. linear combinations of inputs of a
   * mini-batch (as one dimensional vectors) with coefficients being weigths of
   * incoming connections with correspoding neurons from previous layer.
   * Activations of each input in a mini-batch are laid in columns of a matrix
   * stored as a vector using row major ordering.
   */
  std::vector<double> output_;
  /**<
   * Outputs of this layer's neurons i.e. outputs of activation function with
   * activations as argument. Output to each input of mini-batch is laid in
   * columns of a matrix stored as a vector using row major ordering.
   */
  std::vector<double> weighted_error_;
  /**<
   * Mini-batch of weighted error i.e. for each neuron in previous layer it's a
   * weighted sum of this layer's error with coefficients being weights of
   * connections to that neuron. These values are laid in columns of a matrix
   * stored as a vector using row major ordering.
   */

  /**
   * @brief Resizes and resets members of this class to accomodate required
   *        number of inputs and neurons.
   *
   * @param num_inputs  Number of neurons in previous layer discounting bias.
   * @param num_neurons Number of neurons in this layer discounting bias.
   * @throws std::invalid_argument If num_inputs or num_neurons is not positive.
   */
  virtual void Reshape(int num_inputs, int num_neurons);

  /** @brief Initializes weigths using default algorithm. */
  virtual void InitializeWeights() = 0;

  /**
   * @brief Initializes weights using concrete WeightsInitializationStrategy.
   *
   * @param generator Instance of concrete WeightsInitializationStrategy.
   */
  virtual void InitializeWeights(WeightsInitializationStrategy &generator);

  /**
   * @brief Given input computes neurons' activations and applies activation
   *        function using cpu.
   *
   * This function should write to output_. This functions accepts mini-batch
   * laid in matrix columns stored in a vector using row major ordering and
   * computes a mini-batch of output.
   *
   * @param input Layer's input.
   */
  virtual void ForwardPropCpu(const std::vector<double> &input) = 0;

  /**
   * @brief Given input computes neurons' activations and applies activation
   *        function using gpu.
   *
   * This function should write to output_. This functions accepts mini-batch
   * laid in matrix columns stored in a vector using row major ordering and
   * computes a mini-batch of output.
   *
   * @param input Layer's input.
   */
  virtual void ForwardPropGpu(const std::vector<double> &input) = 0;

  /**
   * @brief Computes velocity of weights and weighted error of this layer using
   *        cpu.
   *
   * This function should write to error_, velocity_ and weighted_error_.
   * This functions accepts mini-batch laid in matrix columns stored in a
   * vector using row major ordering and computes a mini-batch of output.
   * Note that if this layer is an output layer then target output should be
   * passed as a weighted_error. In case this layer is first in network's
   * topology, network's input should be passed as a prev_layer_output.
   *
   * @param weighted_error    Weighted sum of succeeding layer's error for each
   *                          neuron in this layer with coefficients being
   *                          weights of connections with these neurons.
   * @param prev_layer_output Return value of forward step in previous layer.
   * @param momentum          Momentum coefficient for velocity calculation.
   * @throws std::invalid_argument If momentum value lies outside of (0, 1) set.
   */
  virtual void BackPropCpu(const std::vector<double> &weighted_error,
                           const std::vector<double> &prev_layer_output,
                           double momentum) = 0;

  /**
   * @brief Computes velocity of weights and weighted error of this layer using
   *        gpu.
   *
   * This function should write to error_, velocity_ and weighted_error_.
   * This functions accepts mini-batch laid in matrix columns stored in a
   * vector using row major ordering and computes a mini-batch of output.
   * Note that if this layer is an output layer then target output should be
   * passed as a weighted_error. In case this layer is first in network's
   * topology, network's input should be passed as a prev_layer_output.
   *
   * @param weighted_error    Weighted sum of succeeding layer's error for each
   *                          neuron in this layer with coefficients being
   *                          weights of connections with these neurons.
   * @param prev_layer_output Return value of forward step in previous layer.
   * @param momentum          Momentum coefficient for velocity calculation.
   * @throws std::invalid_argument If momentum value lies outside of (0, 1) set.
   */
  virtual void BackPropGpu(const std::vector<double> &weighted_error,
                           const std::vector<double> &prev_layer_output,
                           double momentum) = 0;

  /**
   * @brief Updates weights with momentum accumulated across forward-backward
   *        passes using cpu.
   *
   * Weights are updated by substracting velocity multiplied by learning_rate
   * from current weight.
   *
   * @param learning_rate         Speed with which this layer accepts new
   *                              information. Pick this value carefully
   *                              because if it's too big layer can diverge,
   *                              and if it's too small it will take a long
   *                              time to converge.
   * @throws std::invalid_argument If learning_rate is not positive.
   */
  virtual void UpdateCpu(double learning_rate);

  /**
   * @brief Updates weights with momentum accumulated across forward-backward
   *        passes using gpu.
   *
   * Weights are updated by substracting velocity multiplied by learning_rate
   * from current weight.
   *
   * @param learning_rate         Speed with which this layer accepts new
   *                              information. Pick this value carefully
   *                              because if it's too big layer can diverge,
   *                              and if it's too small it will take a long
   *                              time to converge.
   * @throws std::invalid_argument If learning_rate is not positive.
   */
  virtual void UpdateGpu(double learning_rate);

  /**
   * @brief Given inputs computes neurons' activations using cpu.
   *
   * Activations vector is calculated by multiplying weights matrix by input
   * matrix. Input matrix shouldn't contain bias inputs since this function adds
   * it artificially during computation.
   * This functions accepts mini-batch laid in matrix columns stored in a
   * vector using row major ordering and computes a mini-batch of output.
   *
   * @param input Layer's input.
   */
  virtual void ComputeActivationCpu(const std::vector<double> &input);

  /**
   * @brief Given inputs computes neurons' activations using gpu.
   *
   * Activations vector is calculated by multiplying weights matrix by input
   * matrix. Input matrix shouldn't contain bias inputs since this function adds
   * it artificially during computation.
   * This functions accepts mini-batch laid in matrix columns stored in a
   * vector using row major ordering and computes a mini-batch of output.
   *
   * @param d_activation Pointer to memory location on gpu device containing
   *                     array with activations.
   * @param input        Layer's input.
   */
  virtual void ComputeActivationGpu(double *d_activation,
                                    const std::vector<double> &input);

  /**
   * @brief Computes vector required for calculating the error term in
   *        backpropagation step for hidden layers using cpu.
   *
   * This function writes to weighted_error_, Weighted error is calculated by
   * multiplying transpose of weights matrix(without last row corresponding to
   * bias input) by this layer's error vector.
   * This functions accepts mini-batch laid in matrix columns stored in a
   * vector using row major ordering and computes a mini-batch of output.
   */
  virtual void ComputeWeightedErrorCpu();

  /**
   * @brief Computes vector required for calculating the error term in
   *        backpropagation step for hidden layers using gpu.
   *
   * Weighted error is calculated by multiplying transpose of weights matrix
   * (without last row corresponding to bias input) by this layer's error
   * vector.
   * This functions accepts mini-batch laid in matrix columns stored in a
   * vector using row major ordering and computes a mini-batch of output.
   *
   * @param d_weighted_error Pointer to memory location on gpu device containing
   *                         weighted error array.
   * @param d_weights        Pointer to memory location on gpu device containing
   *                         array storing weights matrix.
   * @param d_error          Pointer to memory location on gpu device containing
   *                         error array.
   */
  virtual void ComputeWeightedErrorGpu(double *d_weighted_error,
                                       double *d_weights, double *d_error);

  /**
   * @brief Computes velocity of weights using cpu.
   *
   * velocity(t) = momentum * velocity(t - 1) + (1.0 - momentum) * dE/dW,
   *
   * where:
   *   t is a time step,
   *   dE/dW is a weights gradient calculated on the current mini-batch.
   *
   * This function writes to velocity_ vector and assumes that error_ vector has
   * been already calculated.
   * This functions accepts mini-batch laid in matrix columns stored in a
   * vector using row major ordering and computes a mini-batch of output.
   *
   * @pre Error term for this layer is computed and stored in errors_.
   * @param prev_layer_output Return value of forward step in previous layer.
   * @param momentum          Momentum coefficient for velocity calculation.
   * @throws std::invalid_argument If momentum value lies outside of (0, 1) set.
   */
  virtual void ComputeVelocityCpu(const std::vector<double> &prev_layer_output,
                                  double momentum);

  /**
   * @brief Computes velocity of weights using gpu.
   *
   * velocity(t) = momentum * velocity(t - 1) + (1.0 - momentum) * dE/dW,
   *
   * where:
   *   t is a time step,
   *   dE/dW is a weights gradient calculated on the current mini-batch.
   *
   * This functions accepts mini-batch laid in matrix columns stored in a
   * vector using row major ordering and computes a mini-batch of output.
   *
   * @param d_velocity          Pointer to memory location on gpu device
   *                            containing velocity matrix.
   * @param d_error             Pointer to memory location on gpu device
   *                            containing error matrix.
   * @param d_prev_layer_output Pointer to memory location on gpu device
   *                            containing array storing preceding layer's
   *                            output.
   * @param momentum            Momentum coefficient for velocity calculation.
   * @throws std::invalid_argument If momentum value lies outside of (0, 1) set.
   */
  virtual void ComputeVelocityGpu(double *d_velocity, const double *d_error,
                                  const double *d_prev_layer_output,
                                  double momentum);
};

}  // namespace neuralnet

#endif  // NEURALNET_INCLUDE_LAYERS_LAYER_HPP_
