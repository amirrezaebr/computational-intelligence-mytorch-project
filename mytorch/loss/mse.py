from mytorch import Tensor, Dependency
import numpy as np


def MeanSquaredError(preds: Tensor, actual: Tensor) -> Tensor:
    "TODO: implement Mean Squared Error loss"  # done
    error = preds - actual
    error = error ** 2
    size = Tensor(np.array([error.data.size], dtype=np.float64))
    size = size ** -1
    data = error * size

    if preds.requires_grad:
        def grad_fn(grad: np.ndarray):
            return grad * 2 * (preds.data - actual.data) / preds.data.shape[0]

        depends_on = [Dependency(preds, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=preds.requires_grad, depends_on=depends_on)
