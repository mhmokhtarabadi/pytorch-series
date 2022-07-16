
# Logistic Regression

Here are an example for Logistic Regression using pytorch on dataset provided by sklearn:

```python
import torch
import numpy
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --------------- data loading ---------
data = datasets.load_breast_cancer()

X, Y = data.data, data.target

X_trian, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=10)

sc = StandardScaler()
X_trian = sc.fit_transform(X_trian)
X_test = sc.transform(X_test)

X_trian = torch.from_numpy(X_trian.astype(numpy.float32))
X_test = torch.from_numpy(X_test.astype(numpy.float32))
Y_trian = torch.from_numpy(Y_train.astype(numpy.float32))
Y_test = torch.from_numpy(Y_test.astype(numpy.float32))

Y_trian = Y_trian.view(-1, 1)
Y_test = Y_test.view(-1, 1)

# ----------- training -------------
class LogisticRegression(torch.nn.Module):

    def __init__(self, input_feature_num):
        super(LogisticRegression, self).__init__()
        self.lin = torch.nn.Linear(input_feature_num, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.lin(x))

model = LogisticRegression(30)

loss = torch.nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), 0.01)

iter_num = 1000
for iter in range(iter_num):
    Y_pred = model(X_trian)

    Loss = loss(Y_pred, Y_trian)

    Loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if iter % (iter_num/10) == 0:
        w, b = model.parameters()
        print("epoch %i: loss = %f" % (iter, Loss))

with torch.no_grad():
    Y_pred = model(X_test)
    Y_pred = Y_pred.round()
    acc = Y_pred.eq(Y_test).sum() / Y_test.shape[0]
    print(acc)
```
