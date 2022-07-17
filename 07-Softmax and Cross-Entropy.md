
# Softmax and Cross-Entropy

## 1. Softmax

Softmax normalizes the outputs of a network. here is an example of using softmax in pytorch:

```python
Y = torch.tensor([[2.0, 0.55, 1.0], [1.5, 3.0, 0.2]])

Y_softmax = torch.softmax(Y, dim=0)
print(Y_softmax)

Y_softmax = torch.softmax(Y, dim=1)
print(Y_softmax)
```

The output is:

```
tensor([[0.6225, 0.0794, 0.6900],
        [0.3775, 0.9206, 0.3100]])
tensor([[0.6240, 0.1464, 0.2296],
        [0.1738, 0.7789, 0.0474]])
```

## 2. Cross-Entropy

Cross Entropy is a loss function used for multiclass networks. Here is an example of using Cross Entropy in pytorch:

```python
Y = torch.tensor([0, 1])

Y_pred_good = torch.tensor([[2.0, 1.5, 1.0], [0.1, 1.0, 0.3]])
Y_pred_bad = torch.tensor([[0.1, 2.0, 0.5], [0.5, 0.3, 0.7]])

loss = torch.nn.CrossEntropyLoss()

loss1 = loss(Y_pred_good, Y)
loss2 = loss(Y_pred_bad, Y)

print(loss1)
print(loss2)

_, prediction1 = torch.max(Y_pred_good, dim=1)
_, prediction2 = torch.max(Y_pred_bad, dim=1)

print(prediction1)
print(prediction2)
```

The output is:

```
tensor(0.6619)
tensor(1.7643)
tensor([0, 1])
tensor([1, 2])
```

---

## 3. MultyClass models

Its important that when you are using Cross Entropy loss function, you must not include softmax in model defenition:

```python
class MultiClass(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiClass, self).__init__()
        self.lin1 = torch.nn.Linear(input_size, hidden_size)
        self.rel = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(hidden_size, num_classes)
    
    def forward(self, input):
        out = self.lin1(input)
        out = self.rel(out)
        out = self.lin2(out)

        return out

model = MultiClass(input_size=28*28, hidden_size=100, num_classes=3)
criterion = torch.nn.CrossEntropyLoss() # applies softmax
```

## 4. BinaryClass

In BinaryClass networks that BCEloss is used, you should use sigmoid in model defenition:

```python
class BinaryClass(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClass, self).__init__()
        self.lin1 = torch.nn.Linear(input_size, hidden_size)
        self.rel = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(hidden_size, 1)

    def forward(self, input):
        out = self.lin1(input)
        out = self.rel(out)
        out = self.lin2(out)

        return torch.sigmoid(out)

model = BinaryClass(input_size=28*28, hidden_size=100)
criterion = torch.nn.BCELoss()
```
