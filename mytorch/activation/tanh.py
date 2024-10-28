import numpy as np
from mytorch import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    """
    TODO: (optional) implement tanh function
    hint: you can do it using function you've implemented (not directly define grad func)
    """# done
    numerator = x.exp() - (-x).exp()
    denominator = x.exp() + (-x).exp()
    denominator = denominator ** -1
    data = numerator * denominator
    req_grad = x.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            # Derivative of tanh: 1 - tanh(x)^2
            return grad * (1 - data ** 2)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
