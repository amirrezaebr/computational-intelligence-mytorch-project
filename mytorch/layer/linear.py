from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

class Linear(Layer):
    def __init__(self, inputs: int, outputs: int, need_bias: bool = False, mode="xavier") -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        result = x @ self.weight
        if self.need_bias:
            result += self.bias
        return result

    def initialize(self):
        try:
            self.weight = Tensor(
                data=initializer((self.inputs, self.outputs), mode=self.initialize_mode),
                requires_grad=True
            )
        except NotImplementedError:
            raise ValueError(f"Invalid initialization mode '{self.initialize_mode}'. Supported modes are: xavier, he, random_normal, zero, one.")

        if self.need_bias:
            self.bias = Tensor(
                data=initializer((1, self.outputs), mode="zero"),
                requires_grad=True
            )

    def zero_grad(self):
        if self.weight.requires_grad:
            self.weight.zero_grad()
        if self.need_bias and self.bias.requires_grad:
            self.bias.zero_grad()

    def parameters(self):
        if self.need_bias:
            return [self.weight, self.bias]
        else:
            return [self.weight]

    def __str__(self) -> str:
        return "linear - total param: {} - in: {}, out: {}".format(self.inputs * self.outputs, self.inputs, self.outputs)
