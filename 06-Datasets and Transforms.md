
# Datasets and Transforms

## 1. Data loader

Here we can see how to load data from a *.txt* file or from *torchvision* datasets. To gather data from *.txt* file we have:

```python
import torch
import numpy

# ------------- dataloader class -------
class WineDataset(torch.utils.data.Dataset):

    def __init__(self):
        data = numpy.loadtxt("./pytorchTutorial/data/wine/wine.csv", dtype=numpy.float32, delimiter=',', skiprows=1)
        self.x = torch.from_numpy(data[:, 1:])
        self.y = torch.from_numpy(data[:, [0]])
        self.n_samples = data.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

dataset = WineDataset()

train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# ---------- using data -----------
num_epochs = 2
total_samples = len(dataset)
n_iteration = numpy.ceil(total_samples / 4).astype(numpy.int16)
for epoch in range(num_epochs):
    for i, (inputs, label) in enumerate(train_loader):
        
        if i % 5 == 0:
            print(f"epoch {epoch+1}/{num_epochs}: step {i+1}/{n_iteration} | input shape = {inputs.shape}, label shape = {label.shape}")
```

Or we can load data from *torchvision*:

```python
import torch
import torchvision

# ------------ loading data ----------------
mnist_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True)

mnist_dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=4, shuffle=True)

# -------------- using data --------------
dataiter = iter(mnist_dataloader)
data = dataiter.next()
inputs, targets = data
print(inputs.shape, targets.shape)
```

---

## 2. Transforms

In class defined above, we transformed data to tensor in class, we can do this transformation and any other outside of class:

```python
import torch
import numpy
import torchvision

class WineDataset(torch.utils.data.Dataset):

    def __init__(self, transform=False):
        data = numpy.loadtxt("./pytorchTutorial/data/wine/wine.csv", dtype=numpy.float32, delimiter=',', skiprows=1)
        self.x = data[:, 1:]
        self.y = data[:, [0]]
        self.n_samples = data.shape[0]

        self.transform = transform
    
    def __getitem__(self, index):
        samples =  self.x[index], self.y[index]

        if self.transform:
            samples = self.transform(samples)
        
        return samples
    
    def __len__(self):
        return self.n_samples
```

As you noticed, the output data are no longer *tensor* and they are *numpy array*. So let's define a *ToTensor* transformation:

```python
class ToTensor:
    def __call__(self, samples):
        inputs, targets = samples
        return torch.from_numpy(inputs), torch.from_numpy(targets)
```

Now here we can assign this transformation to our data:

```python
dataset = WineDataset(transform=ToTensor())
```

But let say we have two transforms, we combine them as below:

```python
class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, samples):
        inputs, targets = samples
        return inputs*self.factor, targets

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
```
