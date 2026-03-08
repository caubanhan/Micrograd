import math

class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    # display object
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    # addition
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out


    # multiplication
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out


    # power
    def __pow__(self, other):

        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other-1)) * out.grad

        out._backward = _backward
        return out


    # negative
    def __neg__(self):
        return self * -1


    # subtraction
    def __sub__(self, other):
        return self + (-other)


    # division
    def __truediv__(self, other):
        return self * other**-1


    # tanh activation
    def tanh(self):

        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out


    # backpropagation
    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

if __name__ == "__main__":
    x1 = Value(2.0)
    x2 = Value(0.0)

    w1 = Value(-3.0)
    w2 = Value(1.0)

    b = Value(6.881373587)

    n = x1*w1 + x2*w2 + b
    y = n.tanh()

    y.backward()

    print(x1)
    print(w1)
    print(b)
    print(y)

