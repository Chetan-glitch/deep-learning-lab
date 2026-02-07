# deep-learning-lab
Experiment 1: Fundamentals of Tensor Operations using PyTorch and NumPy
Objective

To understand the basics of tensor creation, manipulation, and operations using NumPy and PyTorch.

Work Done

Created 1D, 2D, and 3D tensors using:

NumPy arrays

PyTorch tensors

Performed basic element-wise operations:

Addition

Subtraction

Multiplication

Division

Implemented:

Dot product

Matrix multiplication

Demonstrated Indexing and Slicing:

Basic indexing

Boolean masking

Extracting subtensors

Used PyTorch tensor reshaping operations:

.view()

.reshape()

.unsqueeze()

.squeeze()

Compared PyTorch reshaping methods with NumPy .reshape()

Demonstrated Broadcasting with tensors of different shapes

Explained and implemented:

In-place operations

Out-of-place operations

Compared behavior and performance differences between NumPy and PyTorch

Experiment 2: Neural Network from Scratch using NumPy on MNIST Dataset
Objective

To build and train a neural network from scratch using NumPy and understand the internal working of neural networks without using deep learning frameworks.

Work Done

Loaded and preprocessed the MNIST dataset

Implemented a feedforward neural network from scratch

Defined:

Weight initialization

Bias initialization

Implemented forward propagation

Used activation functions (Sigmoid / ReLU)

Implemented loss function (Cross-Entropy / MSE)

Implemented backpropagation algorithm

Updated weights using Gradient Descent

Trained the model and evaluated it using:

Training loss

Accuracy

Observed learning behavior over multiple epochs

Experiment 3: Neural Network for Linearly and Non-Linearly Separable Data using NumPy
Objective

To understand the difference between linear and non-linear classification and the role of hidden layers and activation functions.

Work Done

Generated datasets using make_classification

Implemented a single-layer perceptron for:

Linearly separable dataset

Observed that:

Single-layer perceptron successfully learns linear decision boundary

Tested the same perceptron on:

Non-linearly separable dataset

Observed failure of single-layer model on non-linear data

Implemented a multi-layer neural network with:

At least one hidden layer

Non-linear activation functions (Sigmoid / Tanh / ReLU)

Demonstrated how hidden layers enable learning complex decision boundaries

Compared performance between:

Single-layer network

Multi-layer neural network

Experiment 4: CNN Implementation and Transfer Learning
Objective

To design, train, and evaluate Convolutional Neural Networks (CNNs) and compare them with a pretrained ResNet-18 model.

Datasets Used

Cats vs Dogs Dataset

CIFAR-10 Dataset

Work Done
1. Custom CNN Implementation

Designed a CNN architecture with:

Multiple convolutional layers

Pooling layers

Fully connected layers

Applied Dropout and Batch Normalization for regularization

Experimented with different configurations:

Activation Functions:

ReLU

Tanh

Leaky ReLU

Weight Initialization Techniques:

Xavier Initialization

Kaiming Initialization

Random Initialization

Optimizers:

SGD

Adam

RMSprop

2. Training and Evaluation

Trained CNN models using all combinations of:

Activation functions

Weight initialization techniques

Optimizers

Evaluated models using:

Training loss

Validation loss

Accuracy

Saved:

Best-performing CNN model for each dataset

Model weights

Uploaded:

Code

Saved model weights

Results to GitHub repository

3. Transfer Learning with ResNet-18

Fine-tuned pretrained ResNet-18 on:

Cats vs Dogs dataset

CIFAR-10 dataset

Compared performance of:

Custom CNN

ResNet-18

Observed improvement in accuracy using transfer learning
