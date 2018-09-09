#include <algorithm>
#include <fstream>
#include <iostream>

#include "net.hpp"
#include "network_trainer.hpp"

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define NUM_SAMPLES 10000
#define MINI_BATCH_SIZE 64

void transpose(std::vector<double>& output, const std::vector<double>& input,
               int rows, int columns) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < columns; ++col) {
      output[col * rows + row] = input[row * columns + col];
    }
  }
}

std::vector<std::vector<double>> LoadTestInputs() {
  std::vector<std::vector<double>> data;

  std::fstream data_stream;
  data_stream.open("../resources/test_images.txt");

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
    throw std::runtime_error("Couldn't open test input file.");
  }

  return data;
}

std::vector<std::vector<double>> LoadTestLabels() {
  std::vector<std::vector<double>> labels;

  std::fstream data_stream;
  data_stream.open("../resources/test_labels.txt");

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
    throw std::runtime_error("Couldn't open test labels file.");
  }

  return labels;
}


int main() {
  // Creating empty Network. Input layer size doesn't matter here (as long as
  // it's a positive integer).
  neuralnet::Net network(1, true);
  // Loading pretrained network from file.
  network.Load("trained_network.json");

  // Preparing test data.
  std::vector<std::vector<double>> inputs = LoadTestInputs();
  std::vector<std::vector<double>> labels = LoadTestLabels();

  std::cout << "Test data ready!" << std::endl;

  // We will use network trainer to test how accurate our network is.
  neuralnet::NetworkTrainer net_trainer(network);
  net_trainer.Test(inputs, labels);
}
