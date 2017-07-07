Weights are updated iteratively with gradient descent. There is one weight set and basis function for each layer.
$$
w_{t+1} = w_t - \gamma_t \  \textrm{vec}^{-1}\Big(\frac{d\ell_n}{dw_{vec}}\Big)_t + \eta \Big(\frac{d\ell_n}{dw_{vec}}\Big)_{t-1}
$$

$$
\textrm{Error} = \ell_n \circ (\delta \circ \ddot{y} \circ \Big [\underset{i=0}{\overset{N-1}{\bigcirc}} (q_i \circ r_i(w, s)\Big] \circ q_N \circ r_N(w_{vec}, s) + \lambda R(w_{vec}))
$$
 
$$
\frac{d\ell_n}{dw_{vec}} = \frac{d\ell_n}{d\delta}  \frac{d\delta}{d\ddot{y}} \frac{d\ddot{y}}{dq_0} \Big( \prod_{i=0}^{N-1} \frac{dq_i}{dr_i}\frac{dr_i}{dq_{i+1}}\Big) \frac{dq_N}{dr_N}\frac{dr_N}{dw_{vec}} + \lambda\frac{d\ell_n}{dR}\frac{dR}{dw_{vec}}
$$


% aggregators

\begin{center}\begin{tabular}{ l l l }
$\displaystyle\ell_n = n^{-1} \sum_{i=0}^{n} (\delta_i + \lambda R)$ &
$\displaystyle  \ddot{\textrm{y}} = \underset{i=0}{\overset{N}{\bigcirc}} (q_i \circ r_i (w, s)) = q_0$
 &  \ \ aggregators\\ \\

% delta
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

## Neural Network

|      File      |                    Purpose                     |
|----------------|------------------------------------------------|
| App.py         | Program driver                                 |
| Neural.py      | Neural net implementation                      |
| Function.py    | Basis, delta and regularization functions      |
| Environment.py | Setup for continuous functions and logic gates |
