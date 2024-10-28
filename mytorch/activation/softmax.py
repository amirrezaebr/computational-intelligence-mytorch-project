import numpy as np
from mytorch import Tensor, Dependency

def softmax(x: Tensor) -> Tensor:
    """
    TODO: implement softmax function
    hint: you can do it using function you've implemented (not directly define grad func)
    hint: you can't use sum because it has not axis argument so there are 2 ways:
        1. implement sum by axis
        2. using matrix mul to do it :) (recommended)
    hint: a/b = a*(b^-1)
    """# done
    one_vector = np.ones((x.shape[-1],1))
    z = x.exp()
    _sum = (z @ one_vector) ** -1
    data = z * _sum

    req_grad = x.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            s = data
            jacobian = s * (grad - np.sum(grad * s, axis=-1, keepdims=True))
            return jacobian

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
