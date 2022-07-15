
# Tensors

As we use arrays in numpy, we use tensors in pytorch.

## 1. define tensor

hHre are some ways to define tensors:

```python
x = torch.empty(2, 3)   # creates an empty tensor
x = torch.rand(2, 3)    # creates a tensor with random values between 0-1
x = torch.ones(2, 3)    # creates tensor with one values
x = torch.zeros(2, 3)   # creates tensor with zero values
x = torch.tensor([[5.2, 5.3], [1.65, 32.16]])   # creates tensor from an array
```

## 2. data types

We can define tensors in different data types:

```python
x = torch.ones(2, 3, dtype=torch.float16)   # creates tensor with float16 dtype
x = torch.ones(2, 3, dtype=torch.float32)   # creates tensor with float16 dtype
x = torch.ones(2, 3, dtype=torch.int)       # creates tensor with float16 dtype
x = torch.ones(2, 3, dtype=torch.double)    # creates tensor with float16 dtype

x.dtype                                     # to see tensor dtype
x.size()                                    # to see the size of tensor
```

## 3. operators

Some operators:

```python
x = torch.rand(2, 3)
y = torch.rand(2, 3)

z = torch.add(x, y)     # z = x + y
z = torch.sub(x, y)     # z = x - y
z = torch.mul(x, y)     # z = x * y
z = torch.div(x, y)     # z = x / y

y.add_(x)               # y = y + x
y.sub_(x)               # y = y - x
y.mul_(x)               # y = y * x
y.div_(x)               # y = y / x
```

## 4. reshape

We can reshape tensors in pytorch:

```python
x = torch.rand(4, 8)

y = x.view(1, 32)       # to reshape a tensor
y = x.view(2, -1)       # number of colomns will be calculated

y = x[:, 1]             # y is the second colomn of x
y = x[2, 1]             # y is a tensor contains third row and second colomn of x
y = x[2, 1].item()      # y is value of ...
```

## 5. tensor to array and array to tensor

We can convert numpy array to pytorch tensors and vice versa:

```python
a = numpy.array([[5.2, 5.3], [1.65, 32.16]])    # a numpy array

b = torch.from_numpy(a)                         # convert numpy array to pytorch tensor

b = torch.rand(2, 3)                            # a pytorch tensor

a = b.numpy()                                   # convert a pytorch tensor to numpy array

# *** if we modify the source value the destination one will also change, unless they are in different devices
```

## 6. devices

We can define tensors in both *cpu* or *gpu* devices:

```python
y = torch.rand(2, 3, device="cuda")             # to define tensor in GPU

device = torch.device("cuda")
y = torch.rand(2, 3, device=device)             # to define tensor in GPU

y = torch.rand(2, 3)
y = y.to("cuda")                                # to send tensor to GPU
```
