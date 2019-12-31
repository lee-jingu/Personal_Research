import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

#loading datasets

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor()
                            )

test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor()
                            )


#DataLoader

batch_size = 100

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True
                                            )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False
                                            )


#model 정의

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        out = self.fc1(x)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = self.relu3(out)

        out = self.fc4(out)

        return out

#인스턴스 생성

input_dim = 28*28
hidden_dim = 100
output_dim = 10

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

if torch.cuda.is_available():
    model.cuda()

#loss와 optimizer 정의

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


#training the MODEL

iter = 0

for epoch in range(5):

    for i, (images, labels) in enumerate(train_loader):

        if torch.cuda.is_available():
            images = Variable(images.view(-1, 28*28).cuda())
            labels = Variable(labels.cuda())

        else:
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

        #clear gradient

        optimizer.zero_grad()

        #model에 입력 넣기

        outputs = model(images)

        #loss 정의
        loss = criterion(outputs, labels)
        #gradient 정의
        loss.backward()
        #optimize
        optimizer.step()

        iter += 1

        if iter % 500 == 0:

            total = 0
            correct = 0

            for images, labels in test_loader:

                images = Variable(images.view(-1, 28*28).cuda())
                outputs = model(images)
                _, predictions = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct += (predictions.cpu() == labels.cpu()).sum()

            accuracy = 100 * correct / total

            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
