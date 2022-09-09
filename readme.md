## autograd-toy

I recently stumbled upon a [tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0&t=6049s&ab_channel=AndrejKarpathy) by A. Karpathy about autograd implementation from scratch.  
After closing the video I decided to implement it from memory in my own way to fix the notion.

**Operations supported:**
- `add`
- `mul`
- `tanh`

**Demonstration:**
```python
from variable import Variable
from ops import add, mul, tanh
from topo import topo_sort

# Graph build
a = Variable(1.0, label="a")
b = Variable(2.0, label="b")
c = Variable(4.0, label="c")
d = mul(a, b, c)
d.label = "d"
e = tanh(d)
e.label = "e"
f = add(d, e)
f.label = "f"

# Backprop
f.grad = 1.0
backprop_order = topo_sort((a, b, c, d, e, f))
for var in backprop_order:
    var.backward()

# Check
print("\n".join(f"{var}" for var in (a, b, c, d, e, f)))

```
