#include "layers/rectified_linear_unit_layer.hpp"

#include "gtest/gtest.h"

#include "weights_initialization_strategy.hpp"

namespace neuralnet_src_tests {

class MockReLuWeightsInitializer : public neuralnet::WeightsInitializationStrategy {
 public:
  double GetWeight(int, int) {
    return weights_[iterator_++ % weights_.size()];
  }
 private:
  int iterator_ = 0;
  std::vector<double> weights_ = { 0.5, 0.2, 0.7, 1.0, 1.0,
                                   0.3, 0.11, 0.5, -0.4, 1.0,
                                   0.1, -0.2, 0.4, -2.5, 1.0,
                                  -2.0, 1.0, -1.0, 1.0, -1.0 };
};

class ReLuLayerTest : public testing::Test {
 protected:
  ReLuLayerTest() {
    initializer_ = std::make_unique<MockReLuWeightsInitializer>();
    test_layer_.Initialize(4, 4, *initializer_);
  }
  neuralnet::ReLuLayer test_layer_;
  std::unique_ptr<neuralnet::WeightsInitializationStrategy> initializer_;
};

TEST_F(ReLuLayerTest, BackPropThrowsExceptionsOnInvalidArguments) {
  std::vector<double> weighted_error_top = { 1.0, 0.0, 0.0, 0.0 };
  std::vector<double> prev_layer_output(4, 1.0);

  try {
    test_layer_.BackProp(weighted_error_top, prev_layer_output, -0.6);
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
    test_layer_.BackProp(weighted_error_top, prev_layer_output, 0.0);
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
    test_layer_.BackProp(weighted_error_top, prev_layer_output, 1.0);
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
    test_layer_.BackProp(weighted_error_top, prev_layer_output, 6.2);
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

TEST_F(ReLuLayerTest, UpdateThrowsExceptionsOnInvalidArguments) {
  try {
    test_layer_.Update(0.0);
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
    test_layer_.Update(-1.0);
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

TEST_F(ReLuLayerTest, PropagatesForwardOnCpuSingletonBatch) {
  std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> test_output = test_layer_.ForwardProp(input);
  // Activations vector: [8, 1.42, -8.1, 0]
  std::vector<double> expected_output = {8.0, 1.42, 0.0, 0.0};

  ASSERT_EQ(expected_output.size(), test_output.size());

  for (int i = 0; i < expected_output.size(); ++i)
    EXPECT_DOUBLE_EQ(expected_output[i], test_output[i]);
}

TEST_F(ReLuLayerTest, PropagatesForwardOnGpuSingletonBatch) {
  test_layer_.SetGpuFlag();

  std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> test_output = test_layer_.ForwardProp(input);
  // Activations vector: [8, 1.42, -8.1, 0]
  std::vector<double> expected_output = {8, 1.42, 0., 0.};

  ASSERT_EQ(expected_output.size(), test_output.size());

  for (int i = 0; i < expected_output.size(); ++i)
    EXPECT_DOUBLE_EQ(expected_output[i], test_output[i]);
}

TEST_F(ReLuLayerTest, PropagatesForwardOnCpuBatch) {
  // Input Matrix:
  //  1.0  1.0
  //  2.0 -1.0
  //  3.0  1.0
  //  4.0 -1.0
  std::vector<double> input = {1.0, 1.0, 2.0, -1.0, 3.0, 1.0, 4.0, -1.0};
  std::vector<double> test_output = test_layer_.ForwardProp(input);
  // Activations Matrix:
  //  8.0  1.0
  //  1.42 2.09
  // -8.1  4.2
  //  0.0 -6.0
  //
  // Expected Output Matrix:
  //  8.0  1.0
  //  1.42 2.09
  //  0.0  4.2
  //  0.0  0.0
  std::vector<double> expected_output = {8.0, 1.0, 1.42, 2.09, 0.0, 4.2, 0.0,
                                         0.0};

  ASSERT_EQ(expected_output.size(), test_output.size());

  for (int i = 0; i < expected_output.size(); ++i)
    EXPECT_EQ(expected_output[i], test_output[i]);
}

TEST_F(ReLuLayerTest, PropagatesForwardOnGpuBatch) {
  test_layer_.SetGpuFlag();

  // Input Matrix:
  //  1.0  1.0
  //  2.0 -1.0
  //  3.0  1.0
  //  4.0 -1.0
  std::vector<double> input = {1.0, 1.0, 2.0, -1.0, 3.0, 1.0, 4.0, -1.0};
  std::vector<double> test_output = test_layer_.ForwardProp(input);
  // Activations Matrix:
  //  8.0  1.0
  //  1.42 2.09
  // -8.1  4.2
  //  0.0 -6.0
  //
  // Expected Output Matrix:
  //  8.0  1.0
  //  1.42 2.09
  //  0.0  4.2
  //  0.0  0.0
  std::vector<double> expected_output = {8.0, 1.0, 1.42, 2.09, 0.0, 4.2, 0.0,
                                         0.0};

  ASSERT_EQ(expected_output.size(), test_output.size());

  for (int i = 0; i < expected_output.size(); ++i)
    EXPECT_EQ(expected_output[i], test_output[i]);
}

TEST_F(ReLuLayerTest, BackPropagatesOnCpuSingletonBatch) {
  std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
  // Activations vector: [8, 1.42, -8.1, 0]
  test_layer_.ForwardProp(input);
  std::vector<double> weighted_error_top = {0.2, -1.6, 7.5, 0.0};
  std::vector<double> prev_layer_output(test_layer_.GetNumInputs() - 1, 1.0);
  std::vector<double> weighted_error = test_layer_.BackProp(weighted_error_top,
                                                            prev_layer_output,
                                                            0.9);
  // Expected Error vector: [0.2, -1.6, 0.0, 0.0]
  std::vector<double> expected_weighted_error = {-0.38, -0.136, -0.66, 0.84};

  ASSERT_EQ(expected_weighted_error.size(), weighted_error.size());

  for (int i = 0; i < expected_weighted_error.size(); ++i)
    EXPECT_DOUBLE_EQ(expected_weighted_error[i], weighted_error[i]);
}

TEST_F(ReLuLayerTest, BackPropagatesOnGpuSingletonBatch) {
  test_layer_.SetGpuFlag();

  std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
  // Activations vector: [8, 1.42, -8.1, 0]
  test_layer_.ForwardProp(input);
  std::vector<double> weighted_error_top = {0.2, -1.6, 7.5, 0.0};
  std::vector<double> prev_layer_output(test_layer_.GetNumInputs() - 1, 1.0);
  std::vector<double> weighted_error = test_layer_.BackProp(weighted_error_top,
    prev_layer_output,
    0.9);
  // Expected Error vector: [0.2, -1.6, 0.0, 0.0]
  std::vector<double> expected_weighted_error = {-0.38, -0.136, -0.66, 0.84};

  ASSERT_EQ(expected_weighted_error.size(), weighted_error.size());

  for (int i = 0; i < expected_weighted_error.size(); ++i)
    EXPECT_DOUBLE_EQ(expected_weighted_error[i], weighted_error[i]);
}

TEST_F(ReLuLayerTest, BackPropagatesOnCpuBatch) {
  // Input Matrix:
  //  1.0  1.0
  //  2.0 -1.0
  //  3.0  1.0
  //  4.0 -1.0
  //
  // Activations Matrix:
  //  8.0  1.0
  //  1.42 2.09
  // -8.1  4.2
  //  0.0 -6.0
  std::vector<double> input = {1.0, 1.0, 2.0, -1.0, 3.0, 1.0, 4.0, -1.0};
  test_layer_.ForwardProp(input);
  // Weighted Error Top Matrix:
  //  0.2  1.3
  // -1.6  0.1
  //  7.5  1.0
  //  0.0 -5.1
  std::vector<double> weighted_error_top = {0.2, 1.3, -1.6, 0.1, 7.5, 1.0, 0.0,
                                            -5.1};
  std::vector<double> prev_layer_output(test_layer_.GetNumInputs() - 1, 1.0);
  std::vector<double> weighted_error = test_layer_.BackProp(weighted_error_top,
                                                            prev_layer_output,
                                                            0.9);
  // Expected Error Matrix:
  //  0.2  1.3
  // -1.6  0.1
  //  0.0  1.0
  //  0.0  0.0
  //
  // Expected Weighted Error Matrix:
  // -0.38   0.78
  // -0.136  0.071
  // -0.66   1.36
  //  0.84  -1.24
  std::vector<double> expected_weighted_error = {-0.38, 0.78, -0.136, 0.071,
                                                 -0.66, 1.36, 0.84, -1.24};

  ASSERT_EQ(expected_weighted_error.size(), weighted_error.size());

  for (int i = 0; i < expected_weighted_error.size(); ++i)
    EXPECT_DOUBLE_EQ(expected_weighted_error[i], weighted_error[i]);
}

TEST_F(ReLuLayerTest, BackPropagatesOnGpuBatch) {
  test_layer_.SetGpuFlag();

  // Input Matrix:
  //  1.0  1.0
  //  2.0 -1.0
  //  3.0  1.0
  //  4.0 -1.0
  //
  // Activations Matrix:
  //  8.0  1.0
  //  1.42 2.09
  // -8.1  4.2
  //  0.0 -6.0
  std::vector<double> input = {1.0, 1.0, 2.0, -1.0, 3.0, 1.0, 4.0, -1.0};
  test_layer_.ForwardProp(input);
  // Weighted Error Top Matrix:
  //  0.2  1.3
  // -1.6  0.1
  //  7.5  1.0
  //  0.0 -5.1
  std::vector<double> weighted_error_top = {0.2, 1.3, -1.6, 0.1, 7.5, 1.0, 0.0,
                                            -5.1 };
  std::vector<double> prev_layer_output(test_layer_.GetNumInputs() - 1, 1.0);
  std::vector<double> weighted_error = test_layer_.BackProp(weighted_error_top,
    prev_layer_output,
    0.9);
  // Expected Error Matrix:
  //  0.2  1.3
  // -1.6  0.1
  //  0.0  1.0
  //  0.0  0.0
  //
  // Expected Weighted Error Matrix:
  // -0.38   0.78
  // -0.136  0.071
  // -0.66   1.36
  //  0.84  -1.24
  std::vector<double> expected_weighted_error = {-0.38, 0.78, -0.136, 0.071,
                                                 -0.66, 1.36, 0.84, -1.24};

  ASSERT_EQ(expected_weighted_error.size(), weighted_error.size());

  for (int i = 0; i < expected_weighted_error.size(); ++i)
    EXPECT_DOUBLE_EQ(expected_weighted_error[i], weighted_error[i]);
}

TEST_F(ReLuLayerTest, UpdatesOnCpuSingletonBatch) {
  std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
  // Activations vector: [8, 1.42, -8.1, 0]
  test_layer_.ForwardProp(input);
  std::vector<double> weighted_error_top = {0.2, -1.6, 7.5, 0.0};
  std::vector<double> prev_layer_output = {1.0, 2.0, 3.0, 4.0};

  std::vector<double> expected_error = {0.2, -1.6, 0.0, 0.0};

  double momentum = 0.9;
  double learning_rate = 0.1;

  test_layer_.BackProp(weighted_error_top, prev_layer_output, momentum);
  test_layer_.Update(learning_rate);

  MockReLuWeightsInitializer generator;
  std::vector<double> weights;
  for (int i = 0; i < test_layer_.GetNumInputs() * test_layer_.GetSize(); ++i)
    // GetWeights arguments are irrelevant with this generator.
    weights.push_back(generator.GetWeight(1, 1));

  std::vector<double> updated_weights = test_layer_.GetWeights();

  std::vector<double> expected_velocity(weights.size());

  // Velocity is initiaalized to zeros so + prev_velocity * momentum is skipped.
  for (int row = 0; row < test_layer_.GetSize(); ++row) {
    for (int column = 0; column < test_layer_.GetNumInputs() - 1; ++column) {
      expected_velocity[row * test_layer_.GetNumInputs() + column] =
          (1.0 - momentum) * prev_layer_output[column] * expected_error[row];
    }
    expected_velocity[(row + 1) * test_layer_.GetNumInputs() - 1] =
        (1.0 - momentum) * expected_error[row];
  }
  
  for (int i = 0; i < weights.size(); ++i)
    EXPECT_DOUBLE_EQ(weights[i] - learning_rate * expected_velocity[i],
                     updated_weights[i]);
}
 
TEST_F(ReLuLayerTest, UpdatesOnGpuSingletonBatch) {
  test_layer_.SetGpuFlag();

  std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
  // Activations vector: [8, 1.42, -8.1, 0]
  test_layer_.ForwardProp(input);
  std::vector<double> weighted_error_top = {0.2, -1.6, 7.5, 0.0};
  std::vector<double> prev_layer_output = {1.0, 2.0, 3.0, 4.0};

  std::vector<double> expected_error = {0.2, -1.6, 0.0, 0.0};

  double momentum = 0.9;
  double learning_rate = 0.1;

  test_layer_.BackProp(weighted_error_top, prev_layer_output, momentum);
  test_layer_.Update(learning_rate);

  MockReLuWeightsInitializer generator;
  std::vector<double> weights;
  for (int i = 0; i < test_layer_.GetNumInputs() * test_layer_.GetSize(); ++i)
    // GetWeights arguments are irrelevant with this generator.
    weights.push_back(generator.GetWeight(1, 1));

  std::vector<double> updated_weights = test_layer_.GetWeights();

  std::vector<double> expected_velocity(weights.size());

  // Velocity is initiaalized to zeros so + prev_velocity * momentum is skipped.
  for (int row = 0; row < test_layer_.GetSize(); ++row) {
    for (int column = 0; column < test_layer_.GetNumInputs() - 1; ++column) {
      expected_velocity[row * test_layer_.GetNumInputs() + column] =
          (1.0 - momentum) * prev_layer_output[column] * expected_error[row];
    }
    expected_velocity[(row + 1) * test_layer_.GetNumInputs() - 1] =
        (1.0 - momentum) * expected_error[row];
  }

  for (int i = 0; i < weights.size(); ++i)
    EXPECT_DOUBLE_EQ(weights[i] - learning_rate * expected_velocity[i],
                     updated_weights[i]);
}

TEST_F(ReLuLayerTest, UpdatesOnCpuBatch) {
  // Input Matrix:
  //  1.0  1.0
  //  2.0 -1.0
  //  3.0  1.0
  //  4.0 -1.0
  //
  // Activations Matrix:
  //  8.0  1.0
  //  1.42 2.09
  // -8.1  4.2
  //  0.0 -6.0
  std::vector<double> input = {1.0, 1.0, 2.0, -1.0, 3.0, 1.0, 4.0, -1.0};
  test_layer_.ForwardProp(input);
  // Weighted Error Top Matrix:
  //  0.2  1.3
  // -1.6  0.1
  //  7.5  1.0
  //  0.0 -5.1
  std::vector<double> weighted_error_top = {0.2, 1.3, -1.6, 0.1, 7.5, 1.0, 0.0,
                                            -5.1};
  // Prev Layer Output Matrix:
  //  1.0  5.0
  //  2.0  6.0
  //  3.0  7.0
  //  4.0  8.0
  std::vector<double> prev_layer_output = {1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0,
                                           8.0};

  double momentum = 0.9;
  double learning_rate = 0.1;

  test_layer_.BackProp(weighted_error_top, prev_layer_output, momentum);
  test_layer_.Update(learning_rate);

  MockReLuWeightsInitializer generator;
  std::vector<double> weights;
  for (int i = 0; i < test_layer_.GetNumInputs() * test_layer_.GetSize(); ++i)
    // GetWeights arguments are irrelevant with this generator.
    weights.push_back(generator.GetWeight(1, 1));

  // Expected Error Matrix:
  //  0.2  1.3
  // -1.6  0.1
  //  0.0  1.0
  //  0.0  0.0
  std::vector<double> expected_error = {0.2, 1.3, -1.6, 0.1, 0.0, 1.0, 0.0,
                                        0.0};

  std::vector<double> updated_weights = test_layer_.GetWeights();

  std::vector<double> expected_velocity(weights.size());

  // Velocity is initiaalized to zeros so + prev_velocity * momentum is skipped.
  for (int row = 0; row < test_layer_.GetSize(); ++row) {
    for (int column = 0; column < test_layer_.GetNumInputs() - 1; ++column) {
      double gradient = 0.0;
      for (int batch = 0; batch < 2; ++batch) {
        gradient += prev_layer_output[column * 2 + batch]
                    * expected_error[row * 2 + batch];
      }
      expected_velocity[row * test_layer_.GetNumInputs() + column] =
          (1.0 - momentum) * gradient;
    }
    double gradient = 0.0;
    for (int batch = 0; batch < 2; ++batch)
      gradient += expected_error[row * 2 + batch];
    expected_velocity[(row + 1) * test_layer_.GetNumInputs() - 1] =
        (1.0 - momentum) * gradient;
  }

  for (int i = 0; i < updated_weights.size(); ++i)
    EXPECT_DOUBLE_EQ(weights[i] - learning_rate * expected_velocity[i],
                     updated_weights[i]);
}

TEST_F(ReLuLayerTest, UpdatesOnGpuBatch) {
  test_layer_.SetGpuFlag();

  // Input Matrix:
  //  1.0  1.0
  //  2.0 -1.0
  //  3.0  1.0
  //  4.0 -1.0
  //
  // Activations Matrix:
  //  8.0  1.0
  //  1.42 2.09
  // -8.1  4.2
  //  0.0 -6.0
  std::vector<double> input = {1.0, 1.0, 2.0, -1.0, 3.0, 1.0, 4.0, -1.0};
  test_layer_.ForwardProp(input);
  // Weighted Error Top Matrix:
  //  0.2  1.3
  // -1.6  0.1
  //  7.5  1.0
  //  0.0 -5.1
  std::vector<double> weighted_error_top = {0.2, 1.3, -1.6, 0.1, 7.5, 1.0, 0.0,
                                            -5.1};
  // Prev Layer Output Matrix:
  //  1.0  5.0
  //  2.0  6.0
  //  3.0  7.0
  //  4.0  8.0
  std::vector<double> prev_layer_output = {1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0,
                                           8.0};

  double momentum = 0.9;
  double learning_rate = 0.1;

  test_layer_.BackProp(weighted_error_top, prev_layer_output, momentum);
  test_layer_.Update(learning_rate);

  MockReLuWeightsInitializer generator;
  std::vector<double> weights;
  for (int i = 0; i < test_layer_.GetNumInputs() * test_layer_.GetSize(); ++i)
    // GetWeights arguments are irrelevant with this generator.
    weights.push_back(generator.GetWeight(1, 1));

  // Expected Error Matrix:
  //  0.2  1.3
  // -1.6  0.1
  //  0.0  1.0
  //  0.0  0.0
  std::vector<double> expected_error = {0.2, 1.3, -1.6, 0.1, 0.0, 1.0, 0.0,
                                        0.0};

  std::vector<double> updated_weights = test_layer_.GetWeights();

  std::vector<double> expected_velocity(weights.size());

  // Velocity is initiaalized to zeros so + prev_velocity * momentum is skipped.
  for (int row = 0; row < test_layer_.GetSize(); ++row) {
    for (int column = 0; column < test_layer_.GetNumInputs() - 1; ++column) {
      double gradient = 0.0;
      for (int batch = 0; batch < 2; ++batch) {
        gradient += prev_layer_output[column * 2 + batch]
                    * expected_error[row * 2 + batch];
      }
      expected_velocity[row * test_layer_.GetNumInputs() + column] =
          (1.0 - momentum) * gradient;
    }
    double gradient = 0.0;
    for (int batch = 0; batch < 2; ++batch)
      gradient += expected_error[row * 2 + batch];
    expected_velocity[(row + 1) * test_layer_.GetNumInputs() - 1] =
        (1.0 - momentum) * gradient;
  }

  for (int i = 0; i < updated_weights.size(); ++i)
    EXPECT_DOUBLE_EQ(weights[i] - learning_rate * expected_velocity[i],
                     updated_weights[i]);
}

} // namespace neuralnet_src_tests
