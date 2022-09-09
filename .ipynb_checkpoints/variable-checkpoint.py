class Variable:
    def __init__(self, data=None, label=None):
        self.data = data
        self.label = label
        self.grad = 0.0
        self._backward = lambda _: None

    def __repr__(self):
        return f"Variable({self.label}, data = {self.data}, grad = {self.grad})"
