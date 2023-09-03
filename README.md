# ToFA: Torch Functional (Linear) Algebra 
ToFA is a library to perform functional operations on PyTorch modules.

## Installation

```bash
git clone https://github.com/louixp/tofa
pip install -e tofa
```

## What is ToFA?

ToFA is motivated by combining PyTorch modules in the weight space. Combining models is an ambiguous operation and is not native to PyTorch. For example, the following code:

```python
from torch import nn

m1, m2 = nn.Linear(2, 3), nn.Linear(2, 3)
m1 + m2
```

throws the error:

```plaintext
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported operand type(s) for +: 'Linear' and 'Linear'
```

ToFA treats each module as a vector in the weight space (think one big vector of all the weights in the module). 

## Usage

Check out our [Google Colab](https://colab.research.google.com/drive/1SD0HqB4COTIIC8VCxHMeZ9ZEcLazgrjD#scrollTo=7aO8N3ciIRtP) for an interactive demo.

### Instantiate a `TofaModule`

A `TofaModule` is a wrapper around a PyTorch module. It provides additional 
vector space operation interfaces that PyTorch does not natively support.

```python
from tofa.tofa import TofaModule

tm1, tm2 = TofaModule(m1), TofaModule(m2)
```

### Weight space operations

#### Element-wise operations

Operators `+`, `-`, `*`, `/` are overloaded in NumPy syntax to perform 
element-wise operations on the weight space of the modules.

```python
tm1 + tm2
```

#### Dot product

```python
tm1 @ tm2
``` 

#### L2-Norm

```python
tm1.norm()
```