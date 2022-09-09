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
