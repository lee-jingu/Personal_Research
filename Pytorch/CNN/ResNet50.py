import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dset


'''Hyper Parameter'''
lr = 0.0001
epoch = 1000
batch_size = 5
base_dim = 64


'''Loading the data'''
transform = transforms.Compose([
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean & std

training_data = dset.CIFAR10(root='./data',train=True,download=False,transform=transform)
test_data = dset.CIFAR10(root='./data',train=False,download=False,transform=transform)
'''Dataloader Setup'''
print(len(training_data))

training_set = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_set = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)


legnth=len(training_set)
print(legnth)
'''Building the ResNet50'''

class BottleNeck(nn.Module):  # size is unchanged

    def __init__(self, input_dim, mid_dim, output_dim, stride=1):
        super(BottleNeck, self).__init__()
        self.stride = stride
        self.layer = nn.Sequential(
                            nn.Conv2d(input_dim, mid_dim, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(mid_dim),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(mid_dim,mid_dim, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(mid_dim),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(mid_dim, output_dim, kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(output_dim)
        )
        self.downsample = nn.Sequential(
                    nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(output_dim)
        )

    def forward(self, x):

        downsample = self.downsample(x)
        out = self.layer(x)
        out = out + downsample
        return out

class BottleNeck_no_down(nn.Module):  # size is unchanged

    def __init__(self, input_dim, mid_dim, output_dim):
        super(BottleNeck_no_down, self).__init__()
        self.layer = nn.Sequential(
                            nn.Conv2d(input_dim, mid_dim, kernel_size=1, stride=1),
                            nn.BatchNorm2d(mid_dim),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(mid_dim,mid_dim, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(mid_dim),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(mid_dim, output_dim, kernel_size=1, stride=1),
                            nn.BatchNorm2d(output_dim)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out + x
        return out


class ResNet50(nn.Module):

    def __init__(self, base_dim, num_classes):
        super(ResNet50, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, base_dim, kernel_size=7, stride=2, padding=3),
                        nn.ReLU(),
                        nn.MaxPool2d(3, 2, 1)
        )
        self.layer2 = nn.Sequential(
                        BottleNeck(base_dim, base_dim, base_dim * 4),
                        BottleNeck_no_down(4 * base_dim, base_dim, 4 * base_dim),
                        BottleNeck(4 * base_dim, base_dim, 4 * base_dim, stride=2)
        )
        self.layer3 = nn.Sequential(
                        BottleNeck(4 * base_dim, 2 * base_dim, 8 * base_dim),
                        BottleNeck_no_down(8 * base_dim, 2 * base_dim , 8 * base_dim),
                        BottleNeck_no_down(8 * base_dim, 2 * base_dim , 8 * base_dim),
                        BottleNeck(8 * base_dim, 2 * base_dim , 8 * base_dim, stride=2)
        )
        self.layer4 = nn.Sequential(
                        BottleNeck(8 * base_dim, 4 * base_dim, 16 * base_dim),
                        BottleNeck_no_down(16 * base_dim, 4 * base_dim , 16 * base_dim),
                        BottleNeck_no_down(16 * base_dim, 4 * base_dim , 16 * base_dim),
                        BottleNeck_no_down(16 * base_dim, 4 * base_dim , 16 * base_dim),
                        BottleNeck_no_down(16 * base_dim, 4 * base_dim , 16 * base_dim),
                        BottleNeck(16 * base_dim, 4 * base_dim , 16 * base_dim, stride=2)
        )
        self.layer5 = nn.Sequential(
                        BottleNeck(16 * base_dim, 8 * base_dim, 32 * base_dim),
                        BottleNeck_no_down(32 * base_dim, 8 *base_dim, 32 * base_dim),
                        BottleNeck(32 * base_dim, 8 * base_dim, 32 * base_dim)
        )
        self.avgpool = nn.AvgPool2d(7, 1)
        self.fc = nn.Linear(base_dim*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = out.view(batch_size, -1)
        out = self.fc(out)

        return out


model = ResNet50(base_dim=64, num_classes=257).cuda()

#print(model)

'''Define Loss and Optimizer'''

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

'''Training'''
sum= 0

for i in range(epoch):

    for images, labels in training_set:

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        predictions = model(images)

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()


    print(loss)
    print("{} times trained".format(i))
    print("{} is average loss".format(avg_loss))
    if i % 10 == 0:
        print(loss)
