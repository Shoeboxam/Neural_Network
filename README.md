$$
 w_{t+1} = w_t - \gamma_t \  \textrm{vec}^{-1}\Big(\frac{d\ell_n}{dw_{vec}}\Big)_t + \eta \Big(\frac{d\ell_n}{dw_{vec}}\Big)_{t-1}
$$

## Neural Network

|      File      |                    Purpose                     |
|----------------|------------------------------------------------|
| App.py         | Program driver                                 |
| Neural.py      | Neural net implementation                      |
| Function.py    | Basis, delta and regularization functions      |
| Environment.py | Setup for continuous functions and logic gates |
