#define _SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING

#include "net_json_io_handler.hpp"

#include <cstdio>
#include <iostream>
#include <fstream>

#include "gtest/gtest.h"
#include "third_party/json-3.2.0/single_include/nlohmann/json.hpp"

#include "layers/rectified_linear_unit_layer.hpp"
#include "layers/sigmoid_output_layer.hpp"
#include "net.hpp"


namespace neuralnet_src_tests {

class NetJsonIoHandlerTest : public testing::Test {
 protected:
  NetJsonIoHandlerTest() : test_network_(1, true) {}

  neuralnet::Net test_network_;
  neuralnet::NetJsonIoHandler io_handler_;
  nlohmann::json net_json_;
};

TEST_F(NetJsonIoHandlerTest, DumpsNetToJson) {
  std::shared_ptr<neuralnet::Layer> hidden_layer =
      std::make_shared<neuralnet::ReLuLayer>();
  std::shared_ptr<neuralnet::Layer> output_layer =
      std::make_shared<neuralnet::SigmoidOutputLayer>();
 
  test_network_.AddLayer(hidden_layer, 10);
  test_network_.AddLayer(output_layer, 5);
  io_handler_.DumpToFile(test_network_, "test_dump");
  std::ifstream file_stream("test_dump.json");
  file_stream >> net_json_;
  file_stream.close();
  // Removing test file, because we've already loaded it into a json object.
  std::remove("test_dump.json");
  EXPECT_EQ(test_network_.GetNumInputs(), net_json_["NetworkInfo"]["NumInputs"]);
  EXPECT_EQ(test_network_.GetNumLayers(), net_json_["NetworkInfo"]["NumLayers"]);

  EXPECT_EQ(hidden_layer->GetLayerType(), net_json_["Layer_0"]["LayerType"]);
  EXPECT_EQ(hidden_layer->GetNumInputs(), net_json_["Layer_0"]["NumInputs"]);
  EXPECT_EQ(hidden_layer->GetSize(), net_json_["Layer_0"]["NumNeurons"]);
  EXPECT_EQ(output_layer->GetLayerType(), net_json_["Layer_1"]["LayerType"]);
  EXPECT_EQ(output_layer->GetNumInputs(), net_json_["Layer_1"]["NumInputs"]);
  EXPECT_EQ(output_layer->GetSize(), net_json_["Layer_1"]["NumNeurons"]);
  
  std::vector<double> expected_weights = hidden_layer->GetWeights();
  std::vector<double> actual_weights = net_json_["Layer_0"]["Weights"];
  ASSERT_EQ(expected_weights.size(), actual_weights.size());
  for (int i = 0; i < expected_weights.size(); ++i) {
    ASSERT_EQ(expected_weights[i], actual_weights[i]);
  }

  expected_weights = output_layer->GetWeights();
  std::vector<double> weights = net_json_["Layer_1"]["Weights"];
  ASSERT_EQ(expected_weights.size(), weights.size());
  for (int i = 0; i < expected_weights.size(); ++i) {
    ASSERT_EQ(expected_weights[i], weights[i]);
  }
}

TEST_F(NetJsonIoHandlerTest, LoadsNetFromJson) {
  io_handler_.LoadFromFile(test_network_,
      "../../resources/test_data/net_json_io_test_load.json");
  std::ifstream file_stream(
      "../../resources/test_data/net_json_io_test_load.json");
  net_json_ << file_stream;
  file_stream.close();

  std::vector<std::shared_ptr<neuralnet::Layer>> layers =
      test_network_.GetLayers();
  EXPECT_EQ(net_json_["NetworkInfo"]["NumInputs"], test_network_.GetNumInputs());
  EXPECT_EQ(net_json_["NetworkInfo"]["NumLayers"], test_network_.GetNumLayers());
  EXPECT_EQ(net_json_["Layer_0"]["LayerType"], layers[0]->GetLayerType());
  EXPECT_EQ(net_json_["Layer_0"]["NumInputs"], layers[0]->GetNumInputs());
  EXPECT_EQ(net_json_["Layer_1"]["LayerType"], layers[1]->GetLayerType());
  EXPECT_EQ(net_json_["Layer_1"]["NumInputs"], layers[1]->GetNumInputs());

  ASSERT_EQ(net_json_["Layer_0"]["NumNeurons"], layers[0]->GetSize());
  std::vector<double> expected_weights_first = net_json_["Layer_0"]["Weights"];
  std::vector<double> actual_weights = layers[0]->GetWeights();
  for (int i = 0; i < expected_weights_first.size(); ++i)
    ASSERT_EQ(expected_weights_first[i], actual_weights[i]);

  ASSERT_EQ(net_json_["Layer_1"]["NumNeurons"], layers[1]->GetSize());
  std::vector<double> expected_weights_second = net_json_["Layer_1"]["Weights"];
  actual_weights = layers[1]->GetWeights();
  for (int i = 0; i < expected_weights_second.size(); ++i)
    ASSERT_EQ(expected_weights_second[i], actual_weights[i]);
}

TEST_F(NetJsonIoHandlerTest, LoadThrowsExceptionOnNonEmptyNet) {
  std::shared_ptr<neuralnet::Layer> hidden_layer =
      std::make_shared<neuralnet::ReLuLayer>();
  test_network_.AddLayer(hidden_layer, 10);

  try {
    io_handler_.LoadFromFile(test_network_,
        "../../resources/test_data/net_json_io_test_load.json");
    FAIL() << "Expected exception to be thrown (std::logic_error)";
  }
  catch (const std::logic_error& err) {
    EXPECT_EQ(err.what(),
              std::string("Load has been called on non empty network."));
    // Correct exception thrown.
  }
  catch (...) {
    FAIL() << "Expected std::logic_error.";
  }
}

TEST_F(NetJsonIoHandlerTest, DumpThrowsExceptionOnEmptyNet) {
  try {
    io_handler_.DumpToFile(test_network_, "should_fail");
    FAIL() << "Expected exception to be thrown (std::logic_error)";
  }
  catch (const std::logic_error& err) {
    EXPECT_EQ(err.what(), std::string("Dump called on empty network."));
    // Correct exception thrown.
  }
  catch (...) {
    FAIL() << "Expected std::logic_error.";
  }
}

} // namespace neuralnet_src_tests