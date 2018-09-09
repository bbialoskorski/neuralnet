#include "layers/softmax_output_layer.hpp"

#include "gtest/gtest.h"

namespace neuralnet_src_tests {

class MockSoftmaxWeightsInitializer : public neuralnet::WeightsInitializationStrategy {
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

class SoftmaxOutputLayerTest : public testing::Test {
 protected:
  SoftmaxOutputLayerTest() {
    initializer_ = std::make_unique<MockSoftmaxWeightsInitializer>();
    test_layer_.Initialize(4, 4, *initializer_);
  }
  neuralnet::SoftmaxOutputLayer test_layer_;
  std::unique_ptr<neuralnet::WeightsInitializationStrategy> initializer_;
};

TEST_F(SoftmaxOutputLayerTest, BackPropThrowsExceptionsOnInvalidArguments) {
  std::vector<double> target_output = {1.0, 0.0, 0.0, 0.0};
  std::vector<double> prev_layer_output(4, 1.0);

  try {
    test_layer_.BackProp(target_output, prev_layer_output, -0.6);
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
    test_layer_.BackProp(target_output, prev_layer_output, 0.0);
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
    test_layer_.BackProp(target_output, prev_layer_output, 1.0);
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
    test_layer_.BackProp(target_output, prev_layer_output, 6.2);
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

TEST_F(SoftmaxOutputLayerTest, UpdateThrowsExceptionsOnInvalidArguments) {
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

TEST_F(SoftmaxOutputLayerTest, PropagatesForwardOnCpuSingletonBatch) {
  std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> test_output = test_layer_.ForwardProp(input);
  // Activations vector: [8, 1.42, -8.1, 0]
  // Activations vector - max(activations): [0.0, -6.58, -16.1, -8]
  double exp_sum = std::exp(0.0) + std::exp(-6.58) + std::exp(-16.1)
                   + std::exp(-8);
  std::vector<double> expected_output = {std::exp(0.0) / exp_sum,
                                         std::exp(-6.58) / exp_sum,
                                         std::exp(-16.1) / exp_sum,
                                         std::exp(-8) / exp_sum};

  ASSERT_EQ(test_layer_.GetSize(), test_output.size());

  for (int i = 0; i < test_output.size(); ++i)
    EXPECT_DOUBLE_EQ(expected_output[i], test_output[i]);
}

TEST_F(SoftmaxOutputLayerTest, PropagatesForwardOnGpuSingletonBatch) {
  test_layer_.SetGpuFlag();

  std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> test_output = test_layer_.ForwardProp(input);
  // Activations vector: [8, 1.42, -8.1, 0]
  // Activations vector - max(activations): [0.0, -6.58, -16.1, -8]
  double exp_sum = std::exp(0.0) + std::exp(-6.58) + std::exp(-16.1)
                   + std::exp(-8);
  std::vector<double> expected_output = {std::exp(0.0) / exp_sum,
                                         std::exp(-6.58) / exp_sum,
                                         std::exp(-16.1) / exp_sum,
                                         std::exp(-8) / exp_sum};

  ASSERT_EQ(test_layer_.GetSize(), test_output.size());

  for (int i = 0; i < test_output.size(); ++i)
    EXPECT_DOUBLE_EQ(expected_output[i], test_output[i]);
}

TEST_F(SoftmaxOutputLayerTest, PropagatesForwardOnCpuBatch) {
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
  // Activations Matrix - max of a_column:
  //  0.0   -3.2
  // -6.58  -2.11
  // -16.1   0.0
  // -8.0   -10.2

  // Sums of exponents of columns.
  double first_col_exp_sum = std::exp(0.0) + std::exp(-6.58) + std::exp(-16.1)
                             + std::exp(-8);
  double second_col_exp_sum = std::exp(-3.2) + std::exp(-2.11) + std::exp(0.0)
                              + std::exp(-10.2);
  std::vector<double> expected_output = {std::exp(0.0) / first_col_exp_sum,
                                         std::exp(-3.2) / second_col_exp_sum,
                                         std::exp(-6.58) / first_col_exp_sum,
                                         std::exp(-2.11) / second_col_exp_sum,
                                         std::exp(-16.1) / first_col_exp_sum,
                                         std::exp(0.0) / second_col_exp_sum,
                                         std::exp(-8.0) / first_col_exp_sum,
                                         std::exp(-10.2) / second_col_exp_sum};

  ASSERT_EQ(test_layer_.GetSize() * 2, test_output.size());

  for (int i = 0; i < test_output.size(); ++i)
    EXPECT_DOUBLE_EQ(expected_output[i], test_output[i]);
}

TEST_F(SoftmaxOutputLayerTest, PropagatesForwardOnGpuBatch) {
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
  // Activations Matrix - max of a_column:
  //  0.0   -3.2
  // -6.58  -2.11
  // -16.1   0.0
  // -8.0   -10.2

  // Sums of exponents of columns.
  double first_col_exp_sum = std::exp(0.0) + std::exp(-6.58) + std::exp(-16.1)
                             + std::exp(-8);
  double second_col_exp_sum = std::exp(-3.2) + std::exp(-2.11) + std::exp(0.0)
                              + std::exp(-10.2);
  std::vector<double> expected_output = {std::exp(0.0) / first_col_exp_sum,
                                         std::exp(-3.2) / second_col_exp_sum,
                                         std::exp(-6.58) / first_col_exp_sum,
                                         std::exp(-2.11) / second_col_exp_sum,
                                         std::exp(-16.1) / first_col_exp_sum,
                                         std::exp(0.0) / second_col_exp_sum,
                                         std::exp(-8.0) / first_col_exp_sum,
                                         std::exp(-10.2) / second_col_exp_sum};

  ASSERT_EQ(test_layer_.GetSize() * 2, test_output.size());

  for (int i = 0; i < test_output.size(); ++i)
    EXPECT_DOUBLE_EQ(expected_output[i], test_output[i]);
}

TEST_F(SoftmaxOutputLayerTest, BackPropagatesOnCpuSingletonBatch) {
  std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
  test_layer_.ForwardProp(input);
  std::vector<double> target_output = {1.0, 0.0, 0.0, 0.0};
  std::vector<double> prev_layer_output(test_layer_.GetNumInputs() - 1, 1.0);
  // Activations vector: [8, 1.42, -8.1, 0]
  // Activations vector - max(activations): [0.0, -6.58, -16.1, -8]
  double exp_sum = std::exp(0.0) + std::exp(-6.58) + std::exp(-16.1)
                   + std::exp(-8);
  // We substract -1.0 from first element because it corresponds to one hot
  // target output.
  std::vector<double> expected_error = {std::exp(0.0) / exp_sum - 1.0,
                                        std::exp(-6.58) / exp_sum,
                                        std::exp(-16.1) / exp_sum,
                                        std::exp(-8) / exp_sum};
  std::vector<double> expected_weighted_error;
  MockSoftmaxWeightsInitializer generator;
  std::vector<double> weights;
  for (int i = 0; i < test_layer_.GetNumInputs() * test_layer_.GetSize(); ++i)
    // GetWeights arguments are irrelevant with this generator.
    weights.push_back(generator.GetWeight(1, 1));
  // Multiplying transposed weights by expected_error vector.
  for (int column = 0; column < test_layer_.GetNumInputs() - 1; ++column) {
    double product = 0.0;
    for (int row = 0; row < test_layer_.GetSize(); ++row) {
      product += weights[row * test_layer_.GetNumInputs() + column]
                 * expected_error[row];
    }
    expected_weighted_error.push_back(product);
  }

  std::vector<double> weighted_error = test_layer_.BackProp(target_output,
                                                            prev_layer_output,
                                                            0.9);

  ASSERT_EQ(expected_weighted_error.size(), weighted_error.size());

  for (int i = 0; i < weighted_error.size(); ++i)
    EXPECT_DOUBLE_EQ(expected_weighted_error[i], weighted_error[i]);
}

TEST_F(SoftmaxOutputLayerTest, BackPropagatesOnGpuSingletonBatch) {
  test_layer_.SetGpuFlag();

  std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
  test_layer_.ForwardProp(input);
  std::vector<double> target_output = {1.0, 0.0, 0.0, 0.0};
  std::vector<double> prev_layer_output(test_layer_.GetNumInputs() - 1, 1.0);
  // Activations vector: [8, 1.42, -8.1, 0]
  // Activations vector - max(activations): [0.0, -6.58, -16.1, -8]
  double exp_sum = std::exp(0.0) + std::exp(-6.58) + std::exp(-16.1)
                   + std::exp(-8);
  std::vector<double> expected_error = {std::exp(0.0) / exp_sum - 1.0,
                                        std::exp(-6.58) / exp_sum,
                                        std::exp(-16.1) / exp_sum,
                                        std::exp(-8) / exp_sum};
  std::vector<double> expected_weighted_error;
  MockSoftmaxWeightsInitializer generator;
  std::vector<double> weights;
  for (int i = 0; i < test_layer_.GetNumInputs() * test_layer_.GetSize(); ++i)
    // GetWeights arguments are irrelevant with this generator.
    weights.push_back(generator.GetWeight(1, 1));
  // Multiplying transposed weights by expected_error vector.
  for (int column = 0; column < test_layer_.GetNumInputs() - 1; ++column) {
    double product = 0.0;
    for (int row = 0; row < test_layer_.GetSize(); ++row) {
      product += weights[row * test_layer_.GetNumInputs() + column]
                 * expected_error[row];
    }
    expected_weighted_error.push_back(product);
  }

  std::vector<double> weighted_error = test_layer_.BackProp(target_output,
                                                            prev_layer_output,
                                                            0.9);

  ASSERT_EQ(expected_weighted_error.size(), weighted_error.size());

  for (int i = 0; i < weighted_error.size(); ++i)
    EXPECT_DOUBLE_EQ(expected_weighted_error[i], weighted_error[i]);
}

TEST_F(SoftmaxOutputLayerTest, BackPropagatesOnCpuBatch) {
  // Input Matrix:
  //  1.0  1.0
  //  2.0 -1.0
  //  3.0  1.0
  //  4.0 -1.0
  std::vector<double> input = { 1.0, 1.0, 2.0, -1.0, 3.0, 1.0, 4.0, -1.0 };
  test_layer_.ForwardProp(input);
  // Target Output Matrix:
  //  1.0  0.0
  //  0.0  0.0
  //  0.0  1.0
  //  0.0  0.0
  std::vector<double> target_output = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  std::vector<double> prev_layer_output(test_layer_.GetNumInputs() - 1, 1.0);

  std::vector<double> weighted_error = test_layer_.BackProp(target_output,
                                                            prev_layer_output,
                                                            0.9);

  // Activations Matrix:
  //  8.0  1.0
  //  1.42 2.09
  // -8.1  4.2
  //  0.0 -6.0
  //
  // Activations Matrix - max of a_column:
  //  0.0   -3.2
  // -6.58  -2.11
  // -16.1   0.0
  // -8.0   -10.2

  // Sums of exponents of columns.
  double first_col_exp_sum = std::exp(0.0) + std::exp(-6.58)
                             + std::exp(-16.1) + std::exp(-8);
  double second_col_exp_sum = std::exp(-3.2) + std::exp(-2.11)
                              + std::exp(0.0) + std::exp(-10.2);
  // -1s correspond to one hot target outputs.
  std::vector<double> expected_error = {std::exp(0.0) / first_col_exp_sum - 1.0,
                                        std::exp(-3.2) / second_col_exp_sum,
                                        std::exp(-6.58) / first_col_exp_sum,
                                        std::exp(-2.11) / second_col_exp_sum,
                                        std::exp(-16.1) / first_col_exp_sum,
                                        std::exp(0.0) / second_col_exp_sum - 1.0,
                                        std::exp(-8.0) / first_col_exp_sum,
                                        std::exp(-10.2) / second_col_exp_sum};
  std::vector<double> expected_weighted_error;

  MockSoftmaxWeightsInitializer generator;
  std::vector<double> weights;
  for (int i = 0; i < test_layer_.GetNumInputs() * test_layer_.GetSize(); ++i)
    // GetWeights arguments are irrelevant with this generator.
    weights.push_back(generator.GetWeight(1, 1));

  // Multiplying transposed weights matrix by expected_error matrix.
  for (int a_column = 0; a_column < test_layer_.GetNumInputs() - 1; ++a_column) {
    for (int b_column = 0; b_column < 2; ++b_column) {
      double product = 0.0;
      for (int a_row = 0; a_row < test_layer_.GetSize(); ++a_row) {
        product += weights[a_row * test_layer_.GetNumInputs() + a_column]
                   * expected_error[a_row * 2 + b_column];
      }
      expected_weighted_error.push_back(product);
    }
  }

  ASSERT_EQ(expected_weighted_error.size(), weighted_error.size());

  for (int i = 0; i < weighted_error.size(); ++i)
    // Using float comparision to avoid precision errors.
    EXPECT_FLOAT_EQ((float)expected_weighted_error[i],
                    (float)weighted_error[i]);
}

TEST_F(SoftmaxOutputLayerTest, BackPropagatesOnGpuBatch) {
  test_layer_.SetGpuFlag();

  // Input Matrix:
  //  1.0  1.0
  //  2.0 -1.0
  //  3.0  1.0
  //  4.0 -1.0
  std::vector<double> input = {1.0, 1.0, 2.0, -1.0, 3.0, 1.0, 4.0, -1.0};
  test_layer_.ForwardProp(input);
  // Target Output Matrix:
  //  1.0  0.0
  //  0.0  0.0
  //  0.0  1.0
  //  0.0  0.0
  std::vector<double> target_output = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  std::vector<double> prev_layer_output(test_layer_.GetNumInputs() - 1, 1.0);

  std::vector<double> weighted_error = test_layer_.BackProp(target_output,
                                                            prev_layer_output,
                                                            0.9);

  // Activations Matrix:
  //  8.0  1.0
  //  1.42 2.09
  // -8.1  4.2
  //  0.0 -6.0
  //
  // Activations Matrix - max of a_column:
  //  0.0   -3.2
  // -6.58  -2.11
  // -16.1   0.0
  // -8.0   -10.2

  // Sums of exponents of columns.
  double first_col_exp_sum = std::exp(0.0) + std::exp(-6.58)
                             + std::exp(-16.1) + std::exp(-8);
  double second_col_exp_sum = std::exp(-3.2) + std::exp(-2.11)
                              + std::exp(0.0) + std::exp(-10.2);
  // -1s correspond to one hot target outputs.
  std::vector<double> expected_error = {std::exp(0.0) / first_col_exp_sum - 1.0,
                                        std::exp(-3.2) / second_col_exp_sum,
                                        std::exp(-6.58) / first_col_exp_sum,
                                        std::exp(-2.11) / second_col_exp_sum,
                                        std::exp(-16.1) / first_col_exp_sum,
                                        std::exp(0.0) / second_col_exp_sum - 1.0,
                                        std::exp(-8.0) / first_col_exp_sum,
                                        std::exp(-10.2) / second_col_exp_sum};
  std::vector<double> expected_weighted_error;

  MockSoftmaxWeightsInitializer generator;
  std::vector<double> weights;
  for (int i = 0; i < test_layer_.GetNumInputs() * test_layer_.GetSize(); ++i)
    // GetWeights arguments are irrelevant with this generator.
    weights.push_back(generator.GetWeight(1, 1));

  // Multiplying transposed weights matrix by expected_error matrix.
  for (int a_column = 0; a_column < test_layer_.GetNumInputs() - 1; ++a_column) {
    for (int b_column = 0; b_column < 2; ++b_column) {
      double product = 0.0;
      for (int a_row = 0; a_row < test_layer_.GetSize(); ++a_row) {
        product += weights[a_row * test_layer_.GetNumInputs() + a_column]
                   * expected_error[a_row * 2 + b_column];
      }
      expected_weighted_error.push_back(product);
    }
  }

  ASSERT_EQ(expected_weighted_error.size(), weighted_error.size());

  for (int i = 0; i < weighted_error.size(); ++i)
    // Using float comparision to avoid precision errors.
    EXPECT_FLOAT_EQ((float)expected_weighted_error[i], (float)weighted_error[i]);
}

TEST_F(SoftmaxOutputLayerTest, UpdatesWeightsOnCpuSingletonBatch) {
  std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
  test_layer_.ForwardProp(input);
  std::vector<double> target_output = {1.0, 0.0, 0.0, 0.0};
  std::vector<double> prev_layer_output = {1.0, 2.0, 3.0, 4.0};

  double momentum = 0.9;
  double learning_rate = 0.1;

  test_layer_.BackProp(target_output, prev_layer_output, momentum);
  test_layer_.Update(learning_rate);

  // Activations vector: [8, 1.42, -8.1, 0]
  // Activations vector - max(activations): [0.0, -6.58, -16.1, -8]
  double exp_sum = std::exp(0.0) + std::exp(-6.58)
                   + std::exp(-16.1) + std::exp(-8);
  // We substract -1.0 from first element because it corresponds to one hot
  // target output.
  std::vector<double> expected_error = {std::exp(0.0) / exp_sum - 1.0,
                                        std::exp(-6.58) / exp_sum,
                                        std::exp(-16.1) / exp_sum,
                                        std::exp(-8) / exp_sum};

  MockSoftmaxWeightsInitializer generator;
  std::vector<double> weights;
  for (int i = 0; i < test_layer_.GetNumInputs() * test_layer_.GetSize(); ++i)
    // GetWeights arguments are irrelevant with this generator.
    weights.push_back(generator.GetWeight(1, 1));

  std::vector<double> expected_velocity(weights.size());

  // Velocity is initiaalized to zeros so + prev_velocity * momentum is skipped.
  for (int row = 0; row < test_layer_.GetSize(); ++row) {
    for (int column = 0; column < test_layer_.GetNumInputs() - 1; ++column)
      expected_velocity[row * test_layer_.GetNumInputs() + column] =
          (1.0 - momentum) * prev_layer_output[column] * expected_error[row];
    expected_velocity[(row + 1) * test_layer_.GetNumInputs() - 1] =
        (1.0 - momentum) * expected_error[row];
  }

  std::vector<double> updated_weights = test_layer_.GetWeights();

  for (int i = 0; i < updated_weights.size(); ++i)
    EXPECT_DOUBLE_EQ(weights[i] - learning_rate * expected_velocity[i],
                     updated_weights[i]);
}

TEST_F(SoftmaxOutputLayerTest, UpdatesWeightsOnGpuSingletonBatch) {
  test_layer_.SetGpuFlag();

  std::vector<double> input = { 1.0, 2.0, 3.0, 4.0 };
  test_layer_.ForwardProp(input);
  std::vector<double> target_output = { 1.0, 0.0, 0.0, 0.0 };
  std::vector<double> prev_layer_output = { 1.0, 2.0, 3.0, 4.0 };

  double momentum = 0.9;
  double learning_rate = 0.1;

  test_layer_.BackProp(target_output, prev_layer_output, momentum);
  test_layer_.Update(learning_rate);

  // Activations vector: [8, 1.42, -8.1, 0]
  // Activations vector - max(activations): [0.0, -6.58, -16.1, -8]
  double exp_sum = std::exp(0.0) + std::exp(-6.58) + std::exp(-16.1)
                   + std::exp(-8);
  // We substract -1.0 from first element because it corresponds to one hot
  // target output.
  std::vector<double> expected_error = {std::exp(0.0) / exp_sum - 1.0,
                                        std::exp(-6.58) / exp_sum,
                                        std::exp(-16.1) / exp_sum,
                                        std::exp(-8) / exp_sum};

  MockSoftmaxWeightsInitializer generator;
  std::vector<double> weights;
  for (int i = 0; i < test_layer_.GetNumInputs() * test_layer_.GetSize(); ++i)
    // GetWeights arguments are irrelevant with this generator.
    weights.push_back(generator.GetWeight(1, 1));

  std::vector<double> expected_velocity(weights.size());

  // Velocity is initiaalized to zeros so + prev_velocity * momentum is skipped.
  for (int row = 0; row < test_layer_.GetSize(); ++row) {
    for (int column = 0; column < test_layer_.GetNumInputs() - 1; ++column)
      expected_velocity[row * test_layer_.GetNumInputs() + column] =
          (1.0 - momentum) * prev_layer_output[column] * expected_error[row];
    expected_velocity[(row + 1) * test_layer_.GetNumInputs() - 1] =
        (1.0 - momentum) * expected_error[row];
  }

  std::vector<double> updated_weights = test_layer_.GetWeights();

  for (int i = 0; i < updated_weights.size(); ++i)
    EXPECT_DOUBLE_EQ(weights[i] - learning_rate * expected_velocity[i],
                     updated_weights[i]);
}

TEST_F(SoftmaxOutputLayerTest, UpdatesWeightsOnCpuBatch) {
  // Input Matrix:
  //  1.0  1.0
  //  2.0 -1.0
  //  3.0  1.0
  //  4.0 -1.0
  std::vector<double> input = {1.0, 1.0, 2.0, -1.0, 3.0, 1.0, 4.0, -1.0};
  test_layer_.ForwardProp(input);
  // Target Output Matrix:
  //  1.0  0.0
  //  0.0  0.0
  //  0.0  1.0
  //  0.0  0.0
  std::vector<double> target_output = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  // Prev Layer Output Matrix:
  //  1.0  5.0
  //  2.0  6.0
  //  3.0  7.0
  //  4.0  8.0
  std::vector<double> prev_layer_output = {1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0,
                                           8.0};

  double momentum = 0.9;
  double learning_rate = 0.1;

  test_layer_.BackProp(target_output, prev_layer_output, momentum);
  test_layer_.Update(learning_rate);

  // Activations Matrix:
  //  8.0  1.0
  //  1.42 2.09
  // -8.1  4.2
  //  0.0 -6.0
  //
  // Activations Matrix - max of a_column:
  //  0.0   -3.2
  // -6.58  -2.11
  // -16.1   0.0
  // -8.0   -10.2

  // Sums of exponents of columns.
  double first_col_exp_sum = std::exp(0.0) + std::exp(-6.58) + std::exp(-16.1)
                             + std::exp(-8);
  double second_col_exp_sum = std::exp(-3.2) + std::exp(-2.11) + std::exp(0.0)
                              + std::exp(-10.2);
  // -1s correspond to one hot target outputs.
  std::vector<double> expected_error = {std::exp(0.0) / first_col_exp_sum - 1.0,
                                        std::exp(-3.2) / second_col_exp_sum,
                                        std::exp(-6.58) / first_col_exp_sum,
                                        std::exp(-2.11) / second_col_exp_sum,
                                        std::exp(-16.1) / first_col_exp_sum,
                                        std::exp(0.0) / second_col_exp_sum - 1.0,
                                        std::exp(-8.0) / first_col_exp_sum,
                                        std::exp(-10.2) / second_col_exp_sum};

  MockSoftmaxWeightsInitializer generator;
  std::vector<double> weights;
  for (int i = 0; i < test_layer_.GetNumInputs() * test_layer_.GetSize(); ++i)
    // GetWeights arguments are irrelevant with this generator.
    weights.push_back(generator.GetWeight(1, 1));

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

  std::vector<double> updated_weights = test_layer_.GetWeights();

  for (int i = 0; i < updated_weights.size(); ++i)
    EXPECT_DOUBLE_EQ(weights[i] - learning_rate * expected_velocity[i],
                     updated_weights[i]);
}

TEST_F(SoftmaxOutputLayerTest, UpdatesWeightsOnGpuBatch) {
  test_layer_.SetGpuFlag();

  // Input Matrix:
  //  1.0  1.0
  //  2.0 -1.0
  //  3.0  1.0
  //  4.0 -1.0
  std::vector<double> input = {1.0, 1.0, 2.0, -1.0, 3.0, 1.0, 4.0, -1.0};
  test_layer_.ForwardProp(input);
  // Target Output Matrix:
  //  1.0  0.0
  //  0.0  0.0
  //  0.0  1.0
  //  0.0  0.0
  std::vector<double> target_output = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  // Prev Layer Output Matrix:
  //  1.0  5.0
  //  2.0  6.0
  //  3.0  7.0
  //  4.0  8.0
  std::vector<double> prev_layer_output = {1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0,
                                           8.0 };

  double momentum = 0.9;
  double learning_rate = 0.1;

  test_layer_.BackProp(target_output, prev_layer_output, momentum);
  test_layer_.Update(learning_rate);

  // Activations Matrix:
  //  8.0  1.0
  //  1.42 2.09
  // -8.1  4.2
  //  0.0 -6.0
  //
  // Activations Matrix - max of a_column:
  //  0.0   -3.2
  // -6.58  -2.11
  // -16.1   0.0
  // -8.0   -10.2

  // Sums of exponents of columns.
  double first_col_exp_sum = std::exp(0.0) + std::exp(-6.58) + std::exp(-16.1)
                             + std::exp(-8);
  double second_col_exp_sum = std::exp(-3.2) + std::exp(-2.11) + std::exp(0.0)
                              + std::exp(-10.2);
  // -1s correspond to one hot target outputs.
  std::vector<double> expected_error = {std::exp(0.0) / first_col_exp_sum - 1.0,
                                        std::exp(-3.2) / second_col_exp_sum,
                                        std::exp(-6.58) / first_col_exp_sum,
                                        std::exp(-2.11) / second_col_exp_sum,
                                        std::exp(-16.1) / first_col_exp_sum,
                                        std::exp(0.0) / second_col_exp_sum - 1.0,
                                        std::exp(-8.0) / first_col_exp_sum,
                                        std::exp(-10.2) / second_col_exp_sum };

  MockSoftmaxWeightsInitializer generator;
  std::vector<double> weights;
  for (int i = 0; i < test_layer_.GetNumInputs() * test_layer_.GetSize(); ++i)
    // GetWeights arguments are irrelevant with this generator.
    weights.push_back(generator.GetWeight(1, 1));

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

  std::vector<double> updated_weights = test_layer_.GetWeights();

  for (int i = 0; i < updated_weights.size(); ++i)
    EXPECT_DOUBLE_EQ(weights[i] - learning_rate * expected_velocity[i],
                     updated_weights[i]);
}

} // namsepace neuralnet_src_tests
