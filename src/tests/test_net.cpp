#include "net.hpp"

#include "gtest/gtest.h"

#include "layers/layer.hpp"
#include "layers/rectified_linear_unit_layer.hpp"
#include "layers/softmax_output_layer.hpp"

namespace neuralnet_src_tests {

class NetTest : public testing::Test {
 protected:
  NetTest() : test_network_(1, false) {}
  neuralnet::Net test_network_;
};

TEST_F(NetTest, AddsLayers) {
  std::shared_ptr<neuralnet::Layer> hidden_layer =
      std::make_shared<neuralnet::ReLuLayer>();
  std::shared_ptr<neuralnet::Layer> output_layer =
      std::make_shared<neuralnet::SoftmaxOutputLayer>();
  test_network_.AddLayer(hidden_layer, 1);
  test_network_.AddLayer(output_layer, 1);
  
  std::vector<std::shared_ptr<neuralnet::Layer>> layers =
      test_network_.GetLayers();
  EXPECT_EQ(hidden_layer, layers[0]);
  EXPECT_EQ(output_layer, layers[1]);
}

TEST_F(NetTest, BackPropThrowsExceptions) {
  std::vector<double> target_output(1, 1.0);
  try {
    test_network_.BackProp(target_output, 0.9);
    FAIL() << "Expected std::logic_error to be thrown.";
  }
  catch (const std::logic_error& err) {
    EXPECT_EQ(err.what(),
              std::string("Error: attempt to BackProp on empty network."));
  }
  catch (...) {
    FAIL() << "Expected std::logic_error.";
  }

  test_network_.AddLayer(std::make_shared<neuralnet::ReLuLayer>(), 1);

  try {
    test_network_.BackProp(target_output, -0.9);
    FAIL() << "Expected std::invalid_argument to be thrown.";
  }
  catch (const std::invalid_argument& err) {
    EXPECT_EQ(err.what(), std::string("momentum coefficient should have a value\
 between 0 and 1."));
  }
  catch (...) {
    FAIL() << "Expected std::invalid_argument.";
  }

  try {
    test_network_.BackProp(target_output, 0.0);
    FAIL() << "Expected std::invalid_argument to be thrown.";
  }
  catch (const std::invalid_argument& err) {
    EXPECT_EQ(err.what(), std::string("momentum coefficient should have a value\
 between 0 and 1."));
  }
  catch (...) {
    FAIL() << "Expected std::invalid_argument.";
  }

  try {
    test_network_.BackProp(target_output, 1.0);
    FAIL() << "Expected std::invalid_argument to be thrown.";
  }
  catch (const std::invalid_argument& err) {
    EXPECT_EQ(err.what(), std::string("momentum coefficient should have a value\
 between 0 and 1."));
  }
  catch (...) {
    FAIL() << "Expected std::invalid_argument.";
  }

  try {
    test_network_.BackProp(target_output, 12.5);
    FAIL() << "Expected std::invalid_argument to be thrown.";
  }
  catch (const std::invalid_argument& err) {
    EXPECT_EQ(err.what(), std::string("momentum coefficient should have a value\
 between 0 and 1."));
  }
  catch (...) {
    FAIL() << "Expected std::invalid_argument.";
  }
}

TEST_F(NetTest, UpdateThrowsException) {
  try {
    test_network_.Update(0.1);
    FAIL() << "Expected std::logic_error to be thrown.";
  }
  catch (const std::logic_error& err) {
    EXPECT_EQ(err.what(),
              std::string("Error: attempt to update empty network"));
  }
  catch (...) {
    FAIL() << "Expected std::logic_error.";
  }

  test_network_.AddLayer(std::make_shared<neuralnet::ReLuLayer>(), 1);

  try {
    test_network_.Update(0.0);
    FAIL() << "Expected std::invalid_argument to be thrown.";
  }
  catch (const std::invalid_argument& err) {
    EXPECT_EQ(err.what(),
              std::string("learning_rate has to be a positive number."));
  }
  catch (...) {
    FAIL() << "Expected std::invalid_argument.";
  }
  
  try {
    test_network_.Update(-1.0);
    FAIL() << "Expected std::invalid_argument to be thrown.";
  }
  catch (const std::invalid_argument& err) {
    EXPECT_EQ(err.what(),
              std::string("learning_rate has to be a positive number."));
  }
  catch (...) {
    FAIL() << "Expected std::invalid_argument.";
  }

}

} // namespace neuralnet_src_tests