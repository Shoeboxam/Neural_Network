## Neural Network

A mathematical explanation of the network architecture is available:
https://shoeboxam.github.io/Neural_Network/

There are two different implementations of the same network:

|       Folder       |            Purpose            |
|--------------------|-------------------------------|
| Jacobian_Chain     | Custom network via chain rule |
| Tensorflow_Wrapper | Same network in Tensorflow    |


Within each folder, there is the following structure:  

|     File    |           Purpose           |
|-------------|-----------------------------|
| Optimize    | Collection of optimizers    |
| Network.py  | Network data structure      |
| Function.py | Listing of common functions |

To use the library, follow the examples shown in the demonstration files:  

|             File            |                   Purpose                   |
|-----------------------------|---------------------------------------------|
| demo_continuous_function.py | Fit multidimensional mathematical functions |
| demo_logic_gate.py          | Reproduce logic gates of arbitrary size     |
| demo_mnist.py               | Learn to classify the MNIST dataset         |


The following features are supported in the Jacobian_Chain network:  
- Optimizers
    + Backpropagation
        * Gradient Descent
        * Momentum
        * Nesterov Accelerated Gradient
        * Adagrad
        * RMSProp
        * Adam
        * Adamax
        * Nadam
        * Quickprop
    + Evolutionary Algorithms
        * Genetic Algorithm
        * Simulated Annealing
- Regularization
    + Weight decay (L1, L2)
    + Dropout
    + Dropconnect
    + Gradient noise
- Selectable functions
    + Basis / Activation
        * Regression (softplus, tanh, etc.)
        * Classification (use softmax on final layer)
    + Cost (sum squared and cross entropy)
    + Annealing rates
    + Weight initializers
- Network Configuration
    + Set number of layers
    + Set number of nodes for each individual layer
    + Selection of basis function for each layer
- Optimizer Configuration
    + Batch size (set to one for stochastic descent)
    + Set hyperparameters for each layer, or broadcast one
        * Learning rate
        * Regularizer parameters
        * Parameters specific to convergence algorithm
    + Configurable stopping criteria
    + Toggleable graphing and debugging

Links to original papers are given in comments when applicable. Many of the features are also available in the Tensorflow wrapper (for benchmarking and correctness validation).


## Adding Environments  
Each demo requires a data access class that returns stimuli and expectations from a configured data set. Network stimuli and expectation matrices are m*n, where m = size of feature and n = number of features. 

Sample is called during training, and survey is called in error evaluation. 
Size input and size output make it easy to build a network that fits a data set.  

If defined, the environment plot can be used to render a view into the prediction space.
