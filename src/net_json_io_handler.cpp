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

#include "net_json_io_handler.hpp"

#include <fstream>
#include <iomanip>

#include "layers/rectified_linear_unit_layer.hpp"
#include "layers/sigmoid_output_layer.hpp"
#include "layers/softmax_output_layer.hpp"
#include "net.hpp"
#include "third_party/json-3.2.0/single_include/nlohmann/json.hpp"

namespace neuralnet {

void NetJsonIoHandler::DumpToFile(Net& network, std::string file_path) {
  std::vector<std::shared_ptr<Layer>> layers = GetLayers(network);

  if (layers.empty()) throw std::logic_error("Dump called on empty network.");
  // Creating json representation of the network.
  nlohmann::json net_json;
  net_json["NetworkInfo"]["NumInputs"] = network.GetNumInputs();
  net_json["NetworkInfo"]["NumLayers"] = network.GetNumLayers();

  for (int i = 0; i < network.GetNumLayers(); ++i) {
    std::string key = "Layer_" + std::to_string(i);
    net_json[key]["LayerType"] = layers[i]->GetLayerType();
    net_json[key]["NumNeurons"] = layers[i]->GetSize();
    net_json[key]["NumInputs"] = layers[i]->GetNumInputs();
    net_json[key]["Weights"] = layers[i]->GetWeights();
  }

  std::ofstream file_stream(file_path + ".json");

  if (file_stream.is_open()) {
    file_stream << std::setw(4) << net_json << std::endl;
    file_stream.close();
  } else {
    throw std::runtime_error("Error opening json file for net export.");
  }
}

void NetJsonIoHandler::LoadFromFile(Net& network, std::string file_path) {
  nlohmann::json net_json;
  std::ifstream file_stream(file_path);
  if (file_stream.is_open()) {
    file_stream >> net_json;
    file_stream.close();
  } else {
    throw std::runtime_error("Error opening json file containing network.");
  }

  std::vector<std::shared_ptr<Layer>> layers = GetLayers(network);
  if (!layers.empty()) {
    throw std::logic_error("Load has been called on non empty network.");
  }

  int num_inputs = net_json["NetworkInfo"]["NumInputs"];
  int num_layers = net_json["NetworkInfo"]["NumLayers"];
  SetInputLayerSize(network, num_inputs);

  for (int i = 0; i < num_layers; ++i) {
    std::string key = "Layer_" + std::to_string(i);
    int num_neurons = net_json[key]["NumNeurons"];
    std::string layer_type = net_json[key]["LayerType"];

    if (layer_type == "ReLuLayer") {
      std::shared_ptr<Layer> new_layer = std::make_shared<ReLuLayer>();
      network.AddLayer(new_layer, num_neurons);
      auto weights_json = net_json[key]["Weights"];
      std::copy(weights_json.begin(), weights_json.end(),
                new_layer->GetWeights().begin());
    } else if (layer_type == "SoftmaxOutputLayer") {
      std::shared_ptr<Layer> new_layer = std::make_shared<SoftmaxOutputLayer>();
      network.AddLayer(new_layer, num_neurons);
      auto weights_json = net_json[key]["Weights"];
      std::copy(weights_json.begin(), weights_json.end(),
                new_layer->GetWeights().begin());
    } else if (layer_type == "SigmoidOutputLayer") {
      std::shared_ptr<Layer> new_layer = std::make_shared<SigmoidOutputLayer>();
      network.AddLayer(new_layer, num_neurons);
      auto weights_json = net_json[key]["Weights"];
      std::copy(weights_json.begin(), weights_json.end(),
                new_layer->GetWeights().begin());
    } else {
      // Unknown Layer type.
      throw std::runtime_error("Json file contains unknown layer type.");
    }
  }
}

}  // namespace neuralnet
