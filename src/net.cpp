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

#include "net.hpp"

#include <iostream>

#include "net_json_io_handler.hpp"

namespace neuralnet {

Net::Net(int input_layer_size, bool gpu_flag)
    : input_layer_size_(input_layer_size), gpu_flag_(gpu_flag) {
  if (input_layer_size < 1)
    throw std::invalid_argument("Network needs to take at least one input.");

  io_handler_ = std::make_shared<NetJsonIoHandler>();
}

Net::Net(int input_layer_size, bool gpu_flag,
         std::shared_ptr<NetIoHandler> io_handler)
    : input_layer_size_(input_layer_size), gpu_flag_(gpu_flag) {
  if (input_layer_size < 1)
    throw std::invalid_argument("Network needs to take at least one input.");

  io_handler_ = io_handler;
}

Net::~Net(){};

void Net::Save(std::string file_name) {
  io_handler_->DumpToFile(*this, file_name);
}

void Net::Load(std::string file_path) {
  io_handler_->LoadFromFile(*this, file_path);
}

void Net::AddLayer(std::shared_ptr<Layer> layer, int num_neurons) {
  if (num_neurons < 1)
    throw std::invalid_argument("num_neurons needs to be positive.");

  int num_inputs =
      layers_.empty() ? input_layer_size_ : layers_.back()->GetSize();

  layer->Initialize(num_inputs, num_neurons);

  if (gpu_flag_)
    layer->SetGpuFlag();
  else
    layer->ClearGpuFlag();

  layers_.push_back(layer);
}

void Net::AddLayer(std::shared_ptr<Layer> layer, int num_neurons,
                   WeightsInitializationStrategy& init_strategy) {
  if (num_neurons < 1)
    throw std::invalid_argument("num_neurons needs to be positive.");

  int num_inputs =
      layers_.empty() ? input_layer_size_ : layers_.back()->GetSize();

  layer->Initialize(num_inputs, num_neurons, init_strategy);

  if (gpu_flag_)
    layer->SetGpuFlag();
  else
    layer->ClearGpuFlag();

  layers_.push_back(layer);
}

double Net::GetLoss(const std::vector<double>& target_outputs) {
  return layers_[layers_.size() - 1]->GetLoss(target_outputs);
}

void Net::SetGpuFlag() {
  gpu_flag_ = true;
  for (std::shared_ptr<Layer> layer : layers_) layer->SetGpuFlag();
}

void Net::ClearGpuFlag() {
  gpu_flag_ = false;
  for (std::shared_ptr<Layer> layer : layers_) layer->ClearGpuFlag();
}

std::vector<double> Net::ForwardProp(const std::vector<double>& input) {
  if (layers_.size() == 0)
    throw std::logic_error("Error: attempt to ForwardProp on empty network.");

  std::vector<double> output = input;

  output_.push_back(input);

  for (auto layer_itr = layers_.begin(); layer_itr != std::prev(layers_.end());
       ++layer_itr) {
    output = (*layer_itr).get()->ForwardProp(output);
    output_.push_back(output);
  }

  output = layers_.back()->ForwardProp(output);

  return output;
}

void Net::BackProp(const std::vector<double>& target_output, double momentum) {
  if (layers_.size() == 0) {
    throw std::logic_error("Error: attempt to BackProp on empty network.");
  }

  if (momentum <= 0 || momentum >= 1)
    throw std::invalid_argument(
        "momentum coefficient should have a value\
 between 0 and 1.");

  std::vector<double> backprop_vector = target_output;

  // Propagates backward through layers.
  for (int layer_index = layers_.size() - 1; layer_index >= 0; --layer_index) {
    backprop_vector = layers_[layer_index]->BackProp(
        backprop_vector, output_[layer_index], momentum);
  }

  output_.erase(output_.begin(), output_.end());
}

void Net::Update(double learning_rate) {
  if (layers_.size() == 0)
    throw std::logic_error("Error: attempt to update empty network");
  if (learning_rate <= 0)
    throw std::invalid_argument("learning_rate has to be a positive number.");

  for (std::shared_ptr<Layer> layer : layers_) layer->Update(learning_rate);
}

void Net::ResetVelocity() {
  for (std::shared_ptr<Layer> layer : layers_) {
    std::vector<double> weights = layer->GetWeights();
    weights.assign(weights.size(), 0.0);
  }
}

}  // namespace neuralnet
