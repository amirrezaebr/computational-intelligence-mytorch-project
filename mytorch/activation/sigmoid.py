import numpy as np
from mytorch import Tensor, Dependency


def sigmoid(x: Tensor) -> Tensor:
    """
    TODO: implement sigmoid function 
    hint: you can do it using function you've implemented (not directly define grad func)
    """  # done
    z = (-x).exp()
    sigmoid_res = (Tensor(np.ones(x.shape)) + z) ** -1
    data = sigmoid_res * Tensor(np.ones(x.shape))
    req_grad = x.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            sigmoid_data = 1 / (1 + np.exp(-x.data))
            return grad * sigmoid_data * (1 - sigmoid_data)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
