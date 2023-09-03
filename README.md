# tofa
**To**rch **F**unctional (Linear) **A**lgebra

## Installation

```bash
git clone https://github.com/louixp/tofa
pip install -e tofa
```

## Usage

### Instantiate a TofaModule

```python
from torch import nn
from tofa import TofaModule

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