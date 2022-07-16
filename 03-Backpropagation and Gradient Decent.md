
# Backpropagation and Gradient Decent

## 1. Autograd from pytorch

Here we can see an example of backpropagation and gradient decent. In this example, the backward gradient will be calculated using *Autograd* in pytorch:

```python
X = torch.tensor([1, 2, 3, 4, 6], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 7, 12], dtype=torch.float32)

w = torch.rand(1, dtype=torch.float32, requires_grad=True)

def predictor(x, w):
    return w*x

def loss(y, y_pred):
    return ((y - y_pred)**2).mean()

learning_rate = 0.01
iter_num = 20

for iter in range(iter_num):
    Y_pred = predictor(X, w)

    Loss = loss(Y, Y_pred)

    Loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
    
    w.grad.zero_()

    if iter % 2 == 0:
        print(f"epoch {iter+1}: w = {w.item()}, loss = {Loss}")

print(f"The prediction of input 5 is {5 * w.item()}")
```

---

## 2. Loss, and Optimizer from pytorch

```python
X = torch.tensor([1, 2, 3, 4, 6, 8], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8, 11, 16], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x, w):
    return x * w

loss = torch.nn.MSELoss()

learning_rate = 0.01
iter_num = 20

optemiser = torch.optim.SGD([w], learning_rate)

for iter in range(iter_num):
    Y_pred = forward(X, w)

    Loss = loss(Y, Y_pred)

    Loss.backward()

    optemiser.step()

    optemiser.zero_grad()

    if iter % 2 == 0:
        print("epoch %i: w = %f, loss = %f" % (iter+1, w.item(), Loss))

print("model prediction: f(5) = %f" % forward(5, w))
```

---

## 3. Model, Loss, and Optimizer from pytorch

```python
X = torch.tensor([[1], [2], [3], [4], [6], [8]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [11], [16]], dtype=torch.float32)

in_feature = 1
out_feature = 1

model = torch.nn.Linear(in_feature, out_feature)

loss = torch.nn.MSELoss()

learning_rate = 0.015
iter_num = 100

optemiser = torch.optim.SGD(model.parameters(), learning_rate)

for iter in range(iter_num):
    Y_pred = model(X)

    Loss = loss(Y, Y_pred)

    Loss.backward()

    optemiser.step()

    optemiser.zero_grad()

    if iter % 10 == 0:
        [w, b] = model.parameters()
        print("epoch %i: w = %f, b = %f, loss = %f" % (iter+1, w, b, Loss))

print("model prediction: f(5) = %f" % model(torch.tensor([5], dtype=torch.float32)))
```

---

## 4. Define model with class

```python
X = torch.tensor([[1], [2], [3], [4], [6], [8]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [11], [16]], dtype=torch.float32)

in_feature = 1
out_feature = 1

class LinearRegresion(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LinearRegresion, self).__init__()
        # define leyers
        self.lin = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.lin(x)

model = LinearRegresion(in_feature, out_feature)

loss = torch.nn.MSELoss()

learning_rate = 0.015
iter_num = 100

optemiser = torch.optim.SGD(model.parameters(), learning_rate)

for iter in range(iter_num):
    Y_pred = model(X)

    Loss = loss(Y, Y_pred)

    Loss.backward()

    optemiser.step()

    optemiser.zero_grad()

    if iter % 10 == 0:
        [w, b] = model.parameters()
        print("epoch %i: w = %f, b = %f, loss = %f" % (iter+1, w, b, Loss))

print("model prediction: f(5) = %f" % model(torch.tensor([5], dtype=torch.float32)))
```
