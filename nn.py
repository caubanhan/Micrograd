# Value → Neuron → Layer → MLP
from random import random
from cal import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random() * 2 - 1) for _ in range(nin)]
        self.b = Value(random() * 2 - 1)

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) # zip(self.w, x) → [(w1, x1), (w2, x2), ...]
        # sum() → w1*x1 + w2*x2 + ... + b
        out = act.tanh()
        return out
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        # self.neurons → [Neuron1, Neuron2, ..., Neuron_nout]

    def __call__(self, x):
        # self.neurons → [Neuron1, Neuron2, ..., Neuron_nout] 
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
# MLP(3,[4,4,1]) → input = 3, Layer1(3,4) + Layer2(4,4) + Layer3(4,1)
class MLP:
    def __init__(self, nin, nouts):
        # nouts → [4, 4, 1]
        # sz - size of each layer = number of neurons in each layer
        # [nin] = [3], [4], [4], [1]
        # nouts - number of neurons in each layer
        self.layers = []
        sz = [nin] + nouts # [3, 4, 4, 1]
        for i in range(len(nouts)):
            # i = 0 → Layer1(3,4) → self.layers.append(Layer(3,4))
            # i = 1 → Layer2(4,4) → self.layers.append(Layer(4,4))
            # i = 2 → Layer3(4,1) → self.layers.append(Layer(4,1))
            self.layers.append(Layer(sz[i], sz[i+1]))
        
    def __call__(self, x):
        # x - input to the MLP
        for layer in self.layers:
            x = layer(x) # x - output of the current layer, input to the next layer
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
if __name__ == "__main__":
    # xs - input data, ys - target labels
    xs = [
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0],
    ]

    ys = [1.0,-1.0,-1.0,1.0] 
    model = MLP(3,[4,4,1])

    # training loop
    for step in range(100):

        # forward
        ypred = [model(x) for x in xs] # ypred = [model([2.0,3.0,-1.0]), model([3.0,-1.0,0.5]), model([0.5,1.0,1.0]), model([1.0,1.0,-1.0])]

        # loss
        loss = sum((yout - ygt)**2 for ygt,yout in zip(ys,ypred))

        # reset gradient
        for p in model.parameters():
            # why reset 
            p.grad = 0.0

        # backward
        loss.backward()

        # gradient descent
        for p in model.parameters():
            p.data += -0.05 * p.grad

        print(step, loss.data)

    # test
    for x in xs:
        # model(x).data return Value object 
        print(x, model(x).data)