# neuralnet
**neuralnet** is a c++ library for implementation of feedforward neural networks developed for educational purposes. For learning process of networks this project uses a variant of backpropagation algorithm called a mini-batch gradient descent with momentum. I've created two implementations for forward, backward propagation and updates: on GPU using CUDA and on CPU using OpenMP. I've tested network's ability to learn by training it on MNIST handwritten digits [data set](http://yann.lecun.com/exdb/mnist/) containing 60000 28x28 grayscale pictures for 10 epochs with 0.01 learning rate and 0.9 momentum coefficient. Network's topology was 784 neuron input layer, 800 neuron hidden layer with ReLu activation, and 10 neuron output layer using softmax activation function with cross entropy loss function. My network achieved **~98% accuracy** on test data set containing 10000 samples not used in training. **Achieved 19x speed up in training time when using gpu implementation vs cpu one.** Net was trained on Nvidia GTX 1050, Intel i7-3820 machine.

This project includes documentation that can be found [here](https://bbialoskorski.github.io/neuralnet/annotated.html).
### Goals for this project:
* Learn about feedforward neural networks.
* Learn about parallel computing on cpu using OpenMP api.
* Learn about parallel computing on gpu using NVIDIA CUDA toolkit.
* Develop a library for implementation of feedforward neural networks with layers as base units of computation.
* Successfully implement and train a network on real life use case data set ([MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/)).
* When developing parallel code focus on performance.
* Deepen c++ knowledge in a process.
* Develop unit tests using googletest framework.

### Available layer types:
* Rectified Linear Unit Layer
* Softmax Output Layer
* Sigmoid Output Layer

### Example:
Let's build our network. First argument to the constructor is the size of the input layer and second argument is a bool setting whether gpu implementation will be used.
```c++
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
```
Now we can load our training data. In this example we are using MNIST handwritten digits dataset containing 60000 28x28 grayscale pictures.
Sample of images from the dataset:
![](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

Training data is organized into a vector of mini-batches, each containing normalized pixel intensity values of 64 pictures laid in columns of 784 by 64 matrix stored as a vector using row major ordering.
```c++
// Preparing training data.
std::vector<std::vector<double>> inputs = LoadTrainingInputs();
std::vector<std::vector<double>> labels = LoadTrainingLabels();

std::cout << "Training data ready!" << std::endl;
```
We can train the network by creating a NetworkTrainer object and passing our network to the constructor and then calling Train function of this object.
```c++
neuralnet::NetworkTrainer net_trainer(network);
// Training the network for 10 epochs using learning rate 0.01 and momentum
// coefficient 0.9.
net_trainer.Train(inputs, labels, 0.01, 0.9, 10);
```
Training on gpu resulted in the following output:
![](https://i.imgur.com/utKvO8N.png)
whereas training on cpu resulted in this output:
![](https://i.imgur.com/rARRkNp.png)

**Gpu implementation ran almost 19 times faster!**

After training is complete we can save our network to file:
```c++
network.Save("trained_network");
```
Later if we want to use saved network we can create an empty Net object and call its Load function passing path to the file that we have our network stored in.
```c++
// Creating empty Network. Input layer size doesn't matter here (as long as
// it's a positive integer.
neuralnet::Net network(1, true);
// Loading pretrained network from file.
network.Load("trained_network.json");
```
Now lets test how accurate our network is on test data set. To do that we will create NetworkTrainer object and call it's Test function. For test data we will use 10000 images from MNIST test data set containing images not used in training. Test data is organized in the same way as training data.
```c++
// Preparing test data.
std::vector<std::vector<double>> inputs = LoadTestInputs();
std::vector<std::vector<double>> labels = LoadTestLabels();

std::cout << "Test data ready!" << std::endl;

// We will use network trainer to test how accurate our network is.
neuralnet::NetworkTrainer net_trainer(network);
net_trainer.Test(inputs, labels);
```
Which resulted in the following output for network trained using gpu:
![](https://i.imgur.com/G4Gyvwg.png)

And this output for network trained using cpu:

![](https://i.imgur.com/ftqG7jx.png)


Difference in accuracy is caused by random weights initlalization.
