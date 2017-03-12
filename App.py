import numpy as np
from Neural import Neural
from Function import *

# ~~~~Learning machine parameters~~~~
S = np.array([[1,1], [-1,1], [1,-1], [-1,-1]])   # Input data  (environment)
O = np.array([-1,1,1,-1])                        # Output data (expectation)
layers = [2, 3, 1]

# Create the net
net = Neural(layers, debug=True)

net.train(S, O)
print(net.evaluate(S))