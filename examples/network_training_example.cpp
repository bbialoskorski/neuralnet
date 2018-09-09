#include <algorithm>
#include <fstream>
#include <iostream>

#include "layers/rectified_linear_unit_layer.hpp"
#include "layers/softmax_output_layer.hpp"
#include "net.hpp"
#include "network_trainer.hpp"

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define NUM_SAMPLES 60000
#define MINI_BATCH_SIZE 64

void transpose(std::vector<double>& output, const std::vector<double>& input,
               int rows, int columns) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < columns; ++col) {
      output[col * rows + row] = input[row * columns + col];
    }
  }
}

std::vector<std::vector<double>> LoadTrainingInputs() {
  std::vector<std::vector<double>> data;

  std::fstream data_stream;
  data_stream.open("../resources/training_images.txt");

  int data_left = NUM_SAMPLES;

  if (data_stream.is_open()) {
    while (data_left > 0) {
      int batch_size = std::min(data_left, MINI_BATCH_SIZE);
      std::vector<double> mini_batch(batch_size * INPUT_SIZE);

      for (int i = 0; i < batch_size * INPUT_SIZE; ++i) {
        double pixel_value;
        data_stream >> pixel_value;
        mini_batch[i] = pixel_value / 255.0;
      }

      std::vector<double> transposed_mini_batch(mini_batch.size());

      transpose(transposed_mini_batch, mini_batch, batch_size, INPUT_SIZE);

      data.push_back(transposed_mini_batch);

      data_left -= batch_size;
    }
    data_stream.close();
  }
  else {
    throw std::runtime_error("Couldn't open training input file.");
  }

  return data;
}

std::vector<std::vector<double>> LoadTrainingLabels() {
  std::vector<std::vector<double>> labels;

  std::fstream data_stream;
  data_stream.open("../resources/training_labels.txt");

  int data_left = NUM_SAMPLES;

  if (data_stream.is_open()) {
    while (data_left > 0) {
      int batch_size = std::min(data_left, MINI_BATCH_SIZE);
      std::vector<double> mini_batch(batch_size * OUTPUT_SIZE, 0.0);

      for (int i = 0; i < batch_size; ++i) {
        int label;
        data_stream >> label;
        mini_batch[i * OUTPUT_SIZE + label] = 1.0;
      }

      std::vector<double> transposed_mini_batch(mini_batch.size());

      transpose(transposed_mini_batch, mini_batch, batch_size, OUTPUT_SIZE);

      labels.push_back(transposed_mini_batch);

      data_left -= batch_size;
    }
    data_stream.close();
  }
  else {
    throw std::runtime_error("Couldn't open training labels file.");
  }

  return labels;
}

int main() {
  // Input layer size is 784 because each image in MNIST data set is 28x28
  // pixels.
  neuralnet::Net network(784, true);
  std::shared_ptr<neuralnet::Layer> hidden =
      std::make_shared<neuralnet::ReLuLayer>();
  std::shared_ptr<neuralnet::Layer> output =
      std::make_shared<neuralnet::SoftmaxOutputLayer>();

  // Hidden layer will use rectifier as an activation funcion and will have
  // 800 neurons (discounting bias neuron).
  network.AddLayer(hidden, 800);
  // Output layer will use softmax activation function which returns probability
  // distribution. This layer will have 10 neurons, one for each class of
  // classification ie. digit.
  network.AddLayer(output, 10);

  // Preparing training data.
  std::vector<std::vector<double>> inputs = LoadTrainingInputs();
  std::vector<std::vector<double>> labels = LoadTrainingLabels();

  std::cout << "Training data ready!" << std::endl;

  neuralnet::NetworkTrainer net_trainer(network);
  // Training the network for 10 epochs using learning rate 0.01 and momentum
  // coefficient 0.9.
  net_trainer.Train(inputs, labels, 0.01, 0.9, 10);

  // Saving trained network to a file.
  network.Save("trained_network");
}
