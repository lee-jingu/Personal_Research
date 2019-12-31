import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

x_value = [i for i in range(11)]
x_value = np.array(x_value, dtype=np.float32)
print(x_value)

y_value = []
for i in x_value:
    result = 2*i + 1
    y_value.append(result)
y_value = np.array(y_value, dtype=np.float32)
print(y_value)

#train data set으로 변환

x_train = x_value.reshape(-1, 1)
y_train = y_value.reshape(-1 ,1)

print(x_train.shape)
print(y_train.shape)

class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)

#단계
#model 정의 <- nn.Module class 상속하여서
#model 생성
#loss 정의
#optimizer 생성
#training

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 100

for epoch in range(epochs):
    epoch += 1

    inputs = torch.from_numpy(x_train).requires_grad_()
    labels = torch.from_numpy(y_train)

    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
print(predicted)


torch.save(model.state_dict(), 'LinearRegression.pkl')
model.load_state_dict(torch.load('LinearRegression.pkl'))
