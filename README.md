The variable names and explanations mirror their implementations in the repository. The same network is implemented twice: once without any machine learning libraries, and once with Tensorflow.

### Objective Function
Supervised training of a neural network is a calculus optimization problem. The objective is to minimize the error between the predicted value produced by the network and an expected output. Error is given by the following function composition.

$$
\ell_n \circ (\delta \circ \ddot{y} \circ \Big [\underset{i=0}{\overset{N-1}{\bigcirc}} (q_i \circ r_i(w_i, s_i)\Big] \circ q_N \circ r_N(w_{vec}, s) + \lambda R(w_{vec}))
$$

The loss function $\ell(\delta)$ is the average cost over all stimuli in the current batch/iteration.

$$
\displaystyle\ell_n = n^{-1} \sum_{i=0}^{n} (\delta_i + \lambda R(w_{vec}))
$$

The cost function $\delta(y, \ddot{y})$ is the difference between the expected value and value predicted by the network. These are represented as the loss functions in Functions.py.

The prediction $\ddot{y}(q_0)$ is simply the output of the final layer, $q_0$. Notice that the derivative is the identity matrix.

The bracketed composition expands with respect to the number of layers in the network. For example, a network with three hidden layers would have a function composition $q_0(r_0(q_1(r_1(q_2(r_2)))))$.

$$
\displaystyle  \ddot{\textrm{y}} = q_0 = \underset{i=0}{\overset{N}{\bigcirc}} (q_i \circ r_i (w_i, s_i))
$$

The decay term $\lambda R(w_{vec})$ is a constraint or regularizer that penalizes growth of weights. As the norm of the weight matrix increases, so does the error contributed by the regularization term. More on this below.

We now have a differentiable objective function that quantifies error. Notice the value of the objective function is dependent solely on the weight matrices and stimuli. Since the stimuli are considered constants, the weight matrices are the only variables that can be manipulated to minimize the objective function.

### Supervised Training
Training is the use of a convergence method to find the optimal values for the weight matrices. Convergence methods are defined as iterative weight updates. Gradient descent is the simplest convergence method.

$$
w_{t+1} = w_t - \gamma_t \  \textrm{vec}^{-1}\Big(\frac{d\ell_n}{dw_{vec}}_t + \eta \frac{d\ell_n}{dw_{vec}}_{t-1}\Big)
$$

The learning step size $\gamma_t$, or learning rate, scales the magnitude of the gradient to prevent overshooting the minima of a function. If not used, the update to the weights in the direction of the minima can be so great the error increases each iteration and the prediction diverges.

This formula for gradient descent includes an extra term for momentum $\eta$. Momentum is useful for approximating a higher order convergence method without computing the hessian/second derivative. Momentum is similar to the finite difference method in that it uses two points to approximate a derivative, where the two points are the gradients from subsequent iterations.

Notice the weight matrix is vectorized, and then reshaped. This is because the derivative of a matrix with respect to a vector is not well defined in matrix calculus. To avoid using tensors, the matrix is massaged into a vector.

### First Derivative
The derivative of the error function with respect to a vectorized weight matrix $w_{vec}$ is a vectorized gradient matrix. $(g : \mathbb{R} \times \mathbb{R}^{m\*n} \mapsto \mathbb{R}^{m\*n})$

$$
\frac{d\ell_n}{dw_{vec}} = \frac{d\ell_n}{d\delta}  \frac{d\delta}{d\ddot{y}} \frac{d\ddot{y}}{dq_0} \Big( \prod_{i=0}^{N-1} \frac{dq_i}{dr_i}\frac{dr_i}{dq_{i+1}}\Big) \frac{dq_N}{dr_N}\frac{dr_N}{dw_{vec}} + \lambda\frac{d\ell_n}{dR}\frac{dR}{dw_{vec}}
$$

This is simply repeated applications of the vector chain rule.

### Reinforcement and Bias
The reinforcement function is a linear transformation with a bias constant. There are two ways to add bias units; the repository implementation and derivatives use the latter, where $b_i$ is a matrix of constants.

$$r_i = w_i\begin{bmatrix}
    s_i\\
    1  
\end{bmatrix}\;\;\;\;\;\;\;\;\;\;\;\; r_i = w_is_i + b_i$$

The derivative for the reinforcement function with respect to the weight matrix has a special case for the input layer.

$$\displaystyle\frac{dr_i}{dw_{i+1}} = s_i\;\;\;\;\;\;\;\;\;\;\;\;\displaystyle\frac{dr_N}{dw_{vec}} = s_N^\textrm{T}\otimes \textrm{I}$$

The Kronecker tensor product corrects for weight matrix vectorization. The dimension of the identity matrix is the number of input nodes. Note that using the Kronecker has $\mathcal{O}(n^3)$, where an alternative gradient propagation implementation has $\mathcal{O}(n^2)$.

The derivative with respect to the stimulus is useful for internal gradient propagation, and the derivative with respect to the bias is useful for updating bias constants.

$$\displaystyle\frac{dr_i}{ds_{i+1}} = w_i\;\;\;\;\;\;\;\;\;\;\;\;\displaystyle\frac{dr_i}{db_i} = I$$

Notice that $s_i$ and $q_{i+1}$ become interchangeable in the hidden layers, because the output of a basis function becomes the stimulus for its parent layer.

### Function Definitions

Functions are defined in functions.py for each network implementation. Some popular examples are listed below.

| class |      name     |                        function                       |                    derivative                    |
|-------|---------------|-------------------------------------------------------|--------------------------------------------------|
| cost  | sum squared   | ($y - \ddot{y})^2$                                    | $-2(y - \ddot{y})^\textrm{T}$                    |
| cost  | cross entropy | $y$log($\ddot{y}$) + $(1-y)$log(1-$\ddot{y}$)         | $-(y - \ddot{y})^\textrm{T}$                     |
| basis | softplus      | $\mathcal{J}\_i =\tau \textrm{log}(1 + e^{r_i/\tau})$ | $\textrm{diag(}\mathcal{S}_i)$                   |
| basis | logistic      | $\mathcal{S}\_i = \tau (1+e^{-r_i/\tau})^{-1}$        | $\textrm{diag(}\mathcal{S}\_i(1-\mathcal{S}_i))$ |
| decay | L1 Lasso      | $\vert w_N\vert$                                      | $I$                                              |
| decay | L2 Ridge      | $\vert w_N\vert^2$                                    | $2\vert w_N\vert$                                |