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

The prediction $\ddot{y}(q_0)$ is simply the output of the final layer, q_0. Notice that the derivative is the identity matrix.

The bracketed composition expands with respect to the number of layers in the network. For example, a network with three hidden layers would have a function composition $q_0(r_0(q_1(r_1(q_2(r_2)))))$.

$$
\displaystyle  \ddot{\textrm{y}} = q_0 = \underset{i=0}{\overset{N}{\bigcirc}} (q_i \circ r_i (w_i, s_i))
$$

The regularization term $\lambda R(w_{vec})$ is a constraint that penalizes growth of weights. As the norm of the weight matrix increases, so does the error contributed by the regularization term. More on this below.

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

### Function Definitions

Cost
$\delta \in \Big \{$ 
\begin{tabular}{ c }
 ($y - \ddot{y})^2$ \\
 $y$log($\ddot{y}$) + $(1-y)$log(1-$\ddot{y}$) 
\end{tabular}\Big\} &
$\displaystyle\frac{d\delta}{d\ddot{y}} \in \Big \{ $
\begin{tabular}{ c }
 $-2(y - \ddot{y})^\textrm{T}$ \\
 $-(y - \ddot{y})^\textrm{T}$
\end{tabular}
 \Big \} &
\begin{tabular}{ c }
sum squared error \\
cross entropy error 
\end{tabular} \\ \\

% basis
$q_i \in \Big \{ $
\begin{tabular}{ c c }
 $\mathcal{J}_i = $ & $\tau \textrm{log}(1 + e^{r_i/\tau})$ \\
 $\mathcal{S}_i = $ & \ $\tau (1+e^{-r_i/\tau})^{-1}$
\end{tabular}
 $\Big \}$ & 
$\displaystyle\frac{dq_i}{dr_i} \in \Big \{ $
\begin{tabular}{ c c }
 $\textrm{diag(}\mathcal{S}_i)$ \\
 $\textrm{diag(}\mathcal{S}_i(1-\mathcal{S}_i))$
\end{tabular}
 $\Big \}$ &
\begin{tabular}{ l }
softplus \\
sigmoid
\end{tabular} \\ \\

% Linear multiplication
$r_i = w_i\begin{bmatrix}
    s_i\\
    1  
\end{bmatrix}$ &
$\displaystyle\frac{dr_i}{dq_{i+1}} = \begin{bmatrix}
    s_i\\
    1  
\end{bmatrix}$ &
$ \ \displaystyle\frac{dr_N}{dw_{vec}} = \begin{bmatrix}
    s_N\\
    1  
\end{bmatrix}^\textrm{T}\otimes \textrm{I}$ \\ \\

% R constraints
$R \in \Big \{ $
\begin{tabular}{ c }
 $\|w_N\|$ \\
 \ $\|w_N\|^2$
\end{tabular}
 $\Big \}$ &
$\displaystyle\frac{dR}{dw} \in \Big \{ $
\begin{tabular}{ c }
 I \\
 $2\|w_N\|$
\end{tabular}
 $\Big \}$ &
\begin{tabular}{ c }
 L1 lasso regularization \\
 L2 ridge regularization
\end{tabular}
\end{tabular}
\end{center}