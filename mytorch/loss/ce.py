from mytorch import Tensor, Dependency
import numpy as np
from mytorch.activation import softmax


def CategoricalCrossEntropy(preds: Tensor, label: Tensor) -> Tensor:
    "TODO: implement Categorical Cross Entropy loss"  # done

    preds_clipped = np.clip(preds.data, 1e-12, 1 - 1e-12)
    _preds = softmax(preds)
    _sum = (label * _preds).sum()
    size = Tensor(np.ndarray(preds.shape).fill(label.shape[0]))
    size = size ** -1
    data = _sum * size

    if preds.requires_grad:
        def grad_fn(grad: np.ndarray):
            return grad * (-label.data / preds_clipped) / preds.data.shape[0]

        depends_on = [Dependency(preds, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=preds.requires_grad, depends_on=depends_on)
