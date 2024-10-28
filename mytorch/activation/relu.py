import numpy as np
from mytorch import Tensor, Dependency

def relu(x: Tensor) -> Tensor:
    "TODO: implement relu function" #done

    # use np.maximum
    data = np.where(x.data > 0, x.data , 0)
    req_grad = x.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad * np.where(x.data > 0, 1, 0)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
