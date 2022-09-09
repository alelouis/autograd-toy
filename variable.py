class Variable:
    def __init__(self, data=None, label=None, prev=()):
        """Numbers with gradients"""
        self.data = data
        self.label = label
        self.grad = 0.0
        self.backward = lambda: None
        self.prev = prev

    def __repr__(self):
        """Variable representation"""
        return f"Variable({self.label}, data = {self.data}, grad = {self.grad})"
