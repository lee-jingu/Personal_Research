import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

'''Hyperparameter Setup'''
batch_size = 100
lr = 0.0001
epoch = 100
base_dim = 64

'''Loading the data'''
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean & std

training_data = dset.CIFAR10(root='./data',train=True,download=False,transform=transform)
test_data = dset.CIFAR10(root='./data',train=False,download=False,transform=transform)
'''Dataloader Setup'''

training_set = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_set = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

def conv_2_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

def conv_3_block(input_dim, output_dim):
    model = nn.Sequential(
        nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    return model

'''VGG Setup'''
class VGG16(nn.Module):

    def __init__(self, base_dim, output_dim):
        super(VGG16, self).__init__()
        self.feature = nn.Sequential(
                                    conv_2_block(3, base_dim),
                                    conv_2_block(base_dim, 2*base_dim),
                                    conv_3_block(2*base_dim, 4*base_dim),
                                    conv_3_block(4*base_dim, 8*base_dim),
                                    conv_3_block(8*base_dim, 8*base_dim),   #output size : 1
                                    )
        self.fc = nn.Linear(8*base_dim * 1 * 1, output_dim)

    def forward(self, x):
        out = self.feature(x)
        out = out.view(-1, 8*base_dim*1*1) #(batch_size, 512)
        out = self.fc(out)
        return out



'''Training'''

model = VGG16(base_dim=base_dim, output_dim=10).cuda()
'''
for i in model.named_children():
    print(i)
'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

for i in range(epoch):
    for img,label in training_set:
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
    
    print("Batch Trained")
    if i % 10 ==0:
        print(loss)
