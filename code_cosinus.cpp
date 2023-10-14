#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>


// file output
#include <fstream>
#include <sstream>

using namespace std;
using namespace torch::indexing;

// Define the neural network model
struct NeuralNetwork : torch::nn::Module {
  torch::nn::Linear layer1{nullptr};
  torch::nn::Linear layer2{nullptr};

  NeuralNetwork() {
    layer1 = register_module("layer1", torch::nn::Linear(1, 64));
    layer2 = register_module("layer2", torch::nn::Linear(64, 1));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::tanh(layer1(x));
    x = layer2(x);
    return x;
  }
};
int main(int argc, char *argv[]) {

  // Preparing sample data
  static const int N_SAMPLES = 1000;
  torch::Tensor x_sequence = torch::linspace(0, 4 * M_PI, N_SAMPLES);

  // Reshape and save x_sequence
  x_sequence = x_sequence.reshape({N_SAMPLES, 1});
  torch::save(x_sequence, "x_sequence.pt");

  // Reshape and save y_sequence
  torch::Tensor y_sequence = torch::cos(x_sequence);
  torch::save(y_sequence, "y_sequence.pt");

  // create the neural model
  NeuralNetwork model;

  // split the training and test data
  float ratio = 0.8;
  int length = x_sequence.size(0);
  int splitSize = length * ratio;
  int rest = length - splitSize;
  torch::Tensor x_TrainPart = x_sequence.slice(0, 0, splitSize);
  torch::Tensor x_TestPart = x_sequence.slice(0, splitSize, length);
  torch::Tensor y_TrainPart = y_sequence.slice(0, 0, splitSize);
  torch::Tensor y_TestPart = y_sequence.slice(0, splitSize, length);

  torch::save(y_TestPart, "y_TestPart.pt");
  torch::save(y_TrainPart, "y_TrainPart.pt");
  torch::save(x_TestPart, "x_TestPart.pt");
  torch::save(x_TrainPart, "x_TrainPart.pt");

  // 4. plot the convergency data using the max(mse(y_prediction data)) convergence (MSE: mean squared error)
  // loss function, and optimizer
  torch::nn::MSELoss loss_fn;
  torch::optim::SGD optimizer(model.parameters(), 0.01);



  // Number of training iterations
  int num_iterations = 1000;

  // Training the model
  for (int iteration = 1; iteration <= num_iterations; ++iteration) {
    optimizer.zero_grad();

    // Forward pass - Training data
    torch::Tensor y_TrainPredict = model.forward(x_TrainPart);

    // Compute the loss
    torch::Tensor loss = loss_fn(y_TrainPredict, y_TrainPart);

    // Backward pass
    loss.backward();

    // Update the model's parameters
    optimizer.step(); 

    if (iteration % 50 == 0) {
      std::cout << "Iteration " << iteration << ", Loss: " << loss.item<float>() << std::endl;
    }
  }

// Predicting on the test data
  torch::Tensor y_prediction_Train = model.forward(x_TrainPart);
  torch::save(y_prediction_Train, "y_prediction_Train.pt");

  // Predicting on the test data
  torch::Tensor y_prediction = model.forward(x_TestPart);

  // Save predicted y in to a file y_prediction.pt
  torch::save(y_prediction, "y_prediction.pt");


  for (int iteration = 0; iteration < rest; ++iteration) {
    if (iteration % 50 == 0) {
      std::cout << "y_true " << y_TestPart[iteration].item<float>() << ", y_predict " << y_prediction[iteration].item<float>() << std::endl;
    }
  }
  
 for (int iteration = 0; iteration < rest; ++iteration) {
    if (iteration % 50 == 0) {
      std::cout << "y_Train_true " << y_TrainPart[iteration].item<float>() << ", y_predict_Train " << y_prediction_Train[iteration].item<float>() << std::endl;
    }
  }
  

  // Save in CSV format

  // Save x_sequence in CSV format
  std::ofstream x_out("x_sequence.csv");
  for (int i = 0; i < N_SAMPLES; ++i) {
      x_out << x_sequence[i].item<float>() << "\n";
  }
  x_out.close();

  // Save y_sequence in CSV format
  std::ofstream y_out("y_sequence.csv");
  for (int i = 0; i < N_SAMPLES; ++i) {
    y_out << y_sequence[i].item<float>() << "\n";
  }
  y_out.close();

// Save x_TestPart in CSV format
  std::ofstream x_Test("x_TestPart.csv");
  for (int i = 0; i < N_SAMPLES - splitSize; ++i) {
    x_Test << x_TestPart[i].item<float>() << "\n";
  }
  x_Test.close();

  // Save y_prediction in CSV format
  std::ofstream y_pred("y_prediction.csv");
  for (int i = 0; i < N_SAMPLES - splitSize; ++i) {
    y_pred << y_prediction[i].item<float>() << "\n";
  }
  y_pred.close();

// Save x_TrainPart in CSV format
  std::ofstream x_Train("x_TrainPart.csv");
  for (int i = 0; i < splitSize; ++i) {
    x_Train << x_TrainPart[i].item<float>() << "\n";
  }
  x_Train.close();

  // Save y_prediction_Train in CSV format
  std::ofstream y_pred_Train("y_prediction_Train.csv");
  for (int i = 0; i < splitSize; ++i) {
    y_pred_Train << y_prediction_Train[i].item<float>() << "\n";
  }
  y_pred.close();

  

 
  
  

  return 0;
}

// ************************************************************************* //
