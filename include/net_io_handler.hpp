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

#ifndef NEURALNET_INCLUDE_NET_IO_HANDLER_HPP_
#define NEURALNET_INCLUDE_NET_IO_HANDLER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "layers/layer.hpp"

namespace neuralnet {

// Forward declaring Net in order to solve circular dependency problem.
class Net;

/**
 * @brief Abstract base class for network file import/export handler.
 */
class NetIoHandler {
 public:
  virtual ~NetIoHandler() {}
  /**
   * @brief Saves information about a network to a file.
   *
   * Implementations of this function should export information about the
   * network such as size of input layer, number of layers used, their type,
   * size and weight matrix.
   *
   * @param network   Net object which you want to export.
   * @param file_path Path to location where you want the file to be created
   *                  followed by a file name without extension suffix.
   * @throws logic_error If network is empty.
   * @throws runtime_error If file couldn't be created.
   */
  virtual void DumpToFile(Net& network, std::string file_path) = 0;

  /**
   * @brief Loads network data from a file.
   *
   * Implementation of this function should load information about the network
   * from file and assemble the network by setting input layer size and adding
   * layers and their weight matrices accordingly.
   *
   * @param network   Net object to which you want to load architecture from
   *                  file.
   * @param file_path Path to file containing network.
   * @throws logic_error If network is not empty.
   * @throws runtime_error If file couldn't be opened.
   */
  virtual void LoadFromFile(Net& network, std::string file_path) = 0;

 protected:
  /** @brief Return's a reference to vector containing network's layers. */
  std::vector<std::shared_ptr<Layer>>& GetLayers(Net& network);

  /** @brief Sets network's input layer size. */
  void SetInputLayerSize(Net& network, int new_size);
};

}  // namespace neuralnet

#endif  // NEURALNET_INCLUDE_NET_IO_HANDLER_HPP_
