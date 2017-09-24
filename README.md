## Neural Network

A mathematical explanation of the network architecture is available:  
https://shoeboxam.github.io/Neural_Network/  

A rewrite to a more flexible gate-based system is in progress:  
https://github.com/Shoeboxam/Neural_Network_Graph  

There are two different implementations of the same network:  

|   Folder   |                       Name                      |
|------------|-------------------------------------------------|
| MFP        | Multilayer Feedforward Perceptron in Numpy      |
| MFP_Simple | Multilayer Feedforward Perceptron, simplified   |
| MFP_TF     | Multilayer Feedforward Perceptron in Tensorflow |
| RBM        | Restricted Boltzmann Machine (incomplete)       |
	
Each implementation has the following structure:  

|     File    |           Purpose           |
|-------------|-----------------------------|
| Optimize    | Collection of optimizers    |
| Network.py  | Network data structure      |
| Function.py | Listing of common functions |

To use the library, follow the examples shown in the demonstration files. These files can also be run, with configurable settings, to evaluate the optimizers and your selected network architecture.  

|            File            |                   Purpose                    |
|----------------------------|----------------------------------------------|
| MFP_continuous_function.py | Fit multidimensional mathematical functions  |
| MFP_figlet_fonts.py        | Autoencoder trained to reproduce ASCII fonts |
| MFP_logic_gate.py          | Reproduce logic gates of arbitrary size      |
| MFP_mnist.py               | Learn to classify the MNIST dataset          |

The simplified MFP network does not have support for batch processing, so I had to make a modified version of MFP_continuous_function.py to work with it.  

The following features are supported in the MFP network:  
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
