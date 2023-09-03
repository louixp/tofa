# ToFA: Torch Functional (Linear) Algebra 
ToFA is library to perform functional operations on PyTorch modules. It treats 
each module as a vector in the weight space (think one big vector of all the 
weights in the module). 

## Installation

```bash
git clone https://github.com/louixp/tofa
pip install -e tofa
```

## Usage

[Google Colab](https://colab.research.google.com/drive/1SD0HqB4COTIIC8VCxHMeZ9ZEcLazgrjD#scrollTo=7aO8N3ciIRtP)

### Instantiate a TofaModule

```python
from torch import nn
from tofa.tofa import TofaModule

m1, m2 = nn.Linear(2, 3), nn.Linear(2, 3)
tm1, tm2 = TofaModule(m1), TofaModule(m2)
```

### Weight space element-wise operations

Operators `+`, `-`, `*`, `/` are overloaded in NumPy syntax to perform 
element-wise operations on the weight space of the modules.

```python
tm1 + tm2
```

### Weight space dot product

```python
tm1 @ tm2
``` 

### Weight space norm

```python
tm1.norm()
```