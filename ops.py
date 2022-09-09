import math

from variable import Variable


def add(*variables: Variable) -> Variable:
    """differentiable addition for Variables"""
    out = Variable(data=math.fsum((var.data for var in variables)), prev=variables)

    def back():
        """backward logic for addition"""
        for var in variables:
            var.grad += 1.0 * out.grad

    out.backward = back
    return out


def mul(*variables: Variable) -> Variable:
    """differentiable product for Variables"""
    out = Variable(data=math.prod((var.data for var in variables)), prev=variables)

    def back():
        """backward logic for product"""
        for var in variables:
            var.grad += math.prod((var.data for var in variables)) / var.data * out.grad

    out.backward = back
    return out


def tanh(var: Variable) -> Variable:
    """differentiable tanh for Variables"""
    out = Variable(data=math.tanh(var.data), prev=(var,))

    def back():
        """backward logic for tanh"""
        var.grad += 1 - math.tanh(var.data) ** 2 * out.grad

    out.backward = back
    return out
