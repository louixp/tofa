from typing import Callable, Dict, Union

import torch
from torch import nn


class TofaModule:
    def __init__(self, module: nn.Module):
        self.module_class = module.__class__
        self.state_dict = {k: v.clone() for k, v in module.state_dict().items()}
        self._repr = f"TofaModule({module.__repr__()})"

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor]):
        obj = cls.__new__(cls)
        obj.state_dict = {k: v.clone() for k, v in state_dict.items()}
        obj._repr = "TofaModule(state dict only)"
        return obj

    def __repr__(self):
        return self._repr

    def _check_elementwise(self, other: "TofaModule"):
        if self.state_dict.keys() != other.state_dict.keys():
            raise ValueError("Incompatible state_dict keys")
        for key in self.state_dict.keys():
            if self.state_dict[key].shape != other.state_dict[key].shape:
                raise ValueError("Incompatible state_dict shapes")

    def _elementwise_operator(
        self, other: Union[int, float, "TofaModule"], op: Callable
    ):
        if isinstance(other, (int, float)):
            new_state_dict = {
                key: op(self.state_dict[key], other) for key in self.state_dict.keys()
            }
        elif isinstance(other, TofaModule):
            self._check_elementwise(other)
            new_state_dict = {
                key: op(self.state_dict[key], other.state_dict[key])
                for key in self.state_dict.keys()
            }

        module = TofaModule.from_state_dict(new_state_dict)
        module.module_class = self.module_class
        module._repr = self._repr
        return module

    def __add__(self, other: Union[int, float, "TofaModule"]):
        if not isinstance(other, (int, float, TofaModule)):
            raise TypeError(
                f"Unsupported operand type for +: {type(self)} and {type(other)}"
            )
        return self._elementwise_operator(other, lambda x, y: x + y)

    def __sub__(self, other: Union[int, float, "TofaModule"]):
        if not isinstance(other, (int, float, TofaModule)):
            raise TypeError(
                f"Unsupported operand type for -: {type(self)} and {type(other)}"
            )
        return self._elementwise_operator(other, lambda x, y: x - y)

    def __mul__(self, other: Union[int, float, "TofaModule"]):
        if not isinstance(other, (int, float, TofaModule)):
            raise TypeError(
                f"Unsupported operand type for *: {type(self)} and {type(other)}"
            )
        return self._elementwise_operator(other, lambda x, y: x * y)

    def __truediv__(self, other: Union[int, float, "TofaModule"]):
        if not isinstance(other, (int, float, TofaModule)):
            raise TypeError(
                f"Unsupported operand type for /: {type(self)} and {type(other)}"
            )
        return self._elementwise_operator(other, lambda x, y: x / y)

    def __pow__(self, other: Union[int, float]):
        if not isinstance(other, (int, float)):
            raise TypeError(
                f"Unsupported operand type for **: {type(self)} and {type(other)}"
            )
        return self._elementwise_operator(other, lambda x, y: x**y)

    def __matmul__(self, other: "TofaModule"):
        if not isinstance(other, TofaModule):
            raise TypeError(
                f"Unsupported operand type for @: {type(self)} and {type(other)}"
            )

        self._check_elementwise(other)
        return sum(
            self.state_dict[key].flatten() @ other.state_dict[key].flatten()
            for key in self.state_dict.keys()
        ).item()
