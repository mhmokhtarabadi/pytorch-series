
# Activation Functions

We use activation functions to make out networks more robust in non-linear problems. Here are some:

* Step
* Sigmoid
* Tanh
* Relu
* Leaky-Relu
* Softmax

We can use these functions in two ways:

## 1. Init

```python
class Example(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Example, self).__init__()
        self.lin1 = torch.nn.Linear(input_size, hidden_size)
        self.rel = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(hidden_size, 1)
        self.sig = torch.nn.Sigmoid()
    
    def forward(self, input):
        out = self.lin1(input)
        out = self.rel(out)
        out = self.lin2(out)
        out = self.sig(out)

        return out
```

Here are some:

```python
torch.nn.Sigmoid()
torch.nn.Tanh()
torch.nn.ReLU()
torch.nn.LeakyReLU()
torch.nn.Softmax()
```

## 2. Forward

```python
class Example(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Example, self).__init__()
        self.lin1 = torch.nn.Linear(input_size, hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, input):
        out1 = torch.relu(self.lin1(input))
        out2 = torch.sigmoid(self.lin2(out1))

        return out2
```

Here are some:

```python
torch.sigmoid()
torch.tanh()
torch.relu()
torch.nn.functional.leaky_relu()
torch.softmax()
```
