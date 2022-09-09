import math


def add(*variables):
    out = Variable(data=math.fsum((var.data for var in variables)))

    def back():
        for var in variables:
            var.grad += 1.0 * out.grad

    out._backward = back
    return out


def mul(*variables):
    out = Variable(data=math.prod((var.data for var in variables)))

    def back():
        for var in variables:
            var.grad += math.prod((var.data for var in variables)) / var.data * out.grad

    out._backward = back
    return out


def tanh(var):
    out = Variable(data=math.tanh(var.data))

    def back():
        var.grad += 1 - math.tanh(var.data) ** 2 * out.grad

    out._backward = back
    return out
