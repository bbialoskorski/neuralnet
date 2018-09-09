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

#ifndef NEURALNET_INCLUDE_NET_JSON_IO_HANDLER_HPP_
#define NEURALNET_INCLUDE_NET_JSON_IO_HANDLER_HPP_

#include "net_io_handler.hpp"

namespace neuralnet {

/** @brief Handler for importing/exporting Net to/from json format. */
class NetJsonIoHandler : public NetIoHandler {
 public:
  ~NetJsonIoHandler() {}

  /**
   * @brief Saves information about a network to a json file.
   *
   * Exports Net to json file in human readable format. File will contain
   * following data: size of input layer, number of layers, types of layers,
   * their sizes, numbers of inputs and weight matrices.
   *
   * @param network   Net object which you want to export.
   * @param file_path Path to location where you want the file to be created
   *                  followed by a file name without extension suffix.
   * @throws logic_error If network is empty.
   * @throws runtime_error If file couldn't be created.
   */
  void DumpToFile(Net& network, std::string file_path);

  /**
   * @brief Loads network data from a json file.
   *
   * Loads information about a network from json file and based on that data
   * sets size of network input layer and adds appropriate layers and their
   * weights. Types of layers handled by this funcion are: ReLuLayer,
   * SoftmaxOutputLayer and SigmoidOutputLayer.
   *
   * @param network   Net object to which you want to load architecture from
   *                  file.
   * @param file_path Path to file containing network.
   * @throws logic_error If network is not empty.
   * @throws runtime_error If file couldn't be opened.
   */
  void LoadFromFile(Net& network, std::string file_path);
};

}  // namespace neuralnet

#endif  // NEURALNET_INCLUDE_NET_JSON_IO_HANDLER_HPP_
