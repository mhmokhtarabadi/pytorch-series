
# Gradiant Calculation with Autograd

## 1. calculating grads

If we want to enable gradiant calculation for a network, we must set *requires_grad* equal to True when we define input vector:

```python
x = torch.rand(3, requires_grad=True)

y = (2*x - torch.log(x)).sum()
y.backward()                            # calculating dy/dx

print(x.grad)                           # printing dy/dx
```

Now if our network has more than one outputs, we should specify a vector:

```python
x = torch.rand(3, requires_grad=True)

y = (2*x - torch.log(x)).sum()
v = torch.tensor([0.1, 1.1, 0.001])
y.backward(v)                            # calculating dy1/dx, dy2/dx, dy3/dx along v vector (I think)

print(x.grad)                           # printing dy1/dx, dy2/dx, dy3/dx (I think)
```

## 2. disabling autograd

We can disable autograd by three ways:

```python
x.requires_grad_(False)
# or
y = x.detach()
# or
with torch.no_grad():
    ...
```

## 3. stop accumulating grads

Whenever we call *x.grad()* the gradians will be added to previous values. To prevent this we must set gradians value to zero at the end:

```python
weights = torch.rand(3, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)
```

the output is:

```
tensor([3., 3., 3.])
tensor([6., 6., 6.])
tensor([9., 9., 9.])
```

But in this situation:

```python
weights = torch.rand(3, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()
```

the output is:

```
tensor([3., 3., 3.])
tensor([3., 3., 3.])
tensor([3., 3., 3.])
```
