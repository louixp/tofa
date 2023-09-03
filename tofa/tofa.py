from typing import Callable, Dict, Union

import torch
from torch import nn
import copy


class TofaModule(nn.Sequential):
    def __init__(self, module: nn.Module):
        super().__init__(module)

    def __repr__(self):
        return f"TofaModule({self[0]})"

    @property
    def _state_dict(self):
        return self[0].state_dict()

    def _check_elementwise_dims(self, other: "TofaModule") -> None:
        if self._state_dict.keys() != other._state_dict.keys():
            raise ValueError("Incompatible state_dict keys")
        for key in self._state_dict.keys():
            if self._state_dict[key].shape != other._state_dict[key].shape:
                raise ValueError("Incompatible state_dict shapes")

    def _check_elementwise_type(
        self, other: Union[int, float, torch.Tensor, "TofaModule"], op_name: str
    ) -> None:
        if not isinstance(other, (int, float, torch.Tensor, TofaModule)):
            raise TypeError(
                f"Unsupported operand type for {op_name}: {type(self)} and {type(other)}"
            )

    def _elementwise_operator(
        self, other: Union[int, float, torch.Tensor, "TofaModule"], op: Callable
    ) -> "TofaModule":
        if isinstance(other, (int, float)):
            new_state_dict = {
                key: op(self._state_dict[key], other) for key in self._state_dict.keys()
            }
        elif isinstance(other, torch.Tensor):
            if other.size() != (1,) and other.size() != ():
                raise ValueError("Incompatible tensor shape: expected (1,)")
            new_state_dict = {
                key: op(self._state_dict[key], other) for key in self._state_dict.keys()
            }
        elif isinstance(other, TofaModule):
            self._check_elementwise_dims(other)
            new_state_dict = {
                key: op(self._state_dict[key], other._state_dict[key])
                for key in self._state_dict.keys()
            }

        module = copy.deepcopy(self[0])
        module.load_state_dict(new_state_dict)
        tofa_module = TofaModule(module)
        return tofa_module

    def __add__(
        self, other: Union[int, float, torch.Tensor, "TofaModule"]
    ) -> "TofaModule":
        self._check_elementwise_type(other, "+")
        return self._elementwise_operator(other, lambda x, y: x + y)

    def __sub__(
        self, other: Union[int, float, torch.Tensor, "TofaModule"]
    ) -> "TofaModule":
        self._check_elementwise_type(other, "-")
        return self._elementwise_operator(other, lambda x, y: x - y)

    def __mul__(
        self, other: Union[int, float, torch.Tensor, "TofaModule"]
    ) -> "TofaModule":
        self._check_elementwise_type(other, "*")
        return self._elementwise_operator(other, lambda x, y: x * y)

    def __truediv__(
        self, other: Union[int, float, torch.Tensor, "TofaModule"]
    ) -> "TofaModule":
        self._check_elementwise_type(other, "/")
        return self._elementwise_operator(other, lambda x, y: x / y)

    def __pow__(self, other: Union[int, float, torch.Tensor]) -> "TofaModule":
        self._check_elementwise_type(other, "**")
        return self._elementwise_operator(other, lambda x, y: x**y)

    def __matmul__(self, other: "TofaModule") -> torch.Tensor:
        if not isinstance(other, TofaModule):
            raise TypeError(
                f"Unsupported operand type for @: {type(self)} and {type(other)}"
            )

        self._check_elementwise_dims(other)
        return sum(
            self._state_dict[key].flatten() @ other._state_dict[key].flatten()
            for key in self._state_dict.keys()
        )

    def norm(self) -> torch.Tensor:
        return torch.sqrt(self @ self)
