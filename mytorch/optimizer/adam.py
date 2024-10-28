from mytorch.optimizer import Optimizer
from typing import List
from mytorch.layer import Layer
import numpy as np

class Adam(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {id(param): np.zeros_like(param.data) for layer in layers for param in layer.parameters()}
        self.v = {id(param): np.zeros_like(param.data) for layer in layers for param in layer.parameters()}
        self.t = 0

    def step(self):
        "TODO: implement Adam algorithm" #done
        self.t += 1
        for layer in self.layers:
            for param in layer.parameters():
                if param.requires_grad:
                    m = self.m[id(param)]
                    m = self.beta1 * m + (1 - self.beta1) * param.grad.data
                    self.m[id(param)] = m

                    v = self.v[id(param)]
                    v = self.beta2 * v + (1 - self.beta2) * (param.grad.data ** 2)
                    self.v[id(param)] = v

                    m_hat = m / (1 - self.beta1 ** self.t)
                    v_hat = v / (1 - self.beta2 ** self.t)

                    param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
