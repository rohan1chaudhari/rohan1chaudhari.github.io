---
categories:
  - Deep Learning
tags:
  - python
  - pytorch
  - image classification
  - MNIST
---


## Importing packages and installing Pytorch


```python
!pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
!pip install torchvision
```

    Collecting torch==0.3.0.post4 from http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
      Downloading http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl (592.3MB)
    [K    98% |███████████████████████████████▋| 584.4MB 44.7MB/s eta 0:00:01[K    100% |████████████████████████████████| 592.3MB 40.7MB/s 
    [?25hRequirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from torch==0.3.0.post4)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.6/dist-packages (from torch==0.3.0.post4)
    Installing collected packages: torch
    Successfully installed torch-0.3.0.post4
    Collecting torchvision
      Downloading torchvision-0.2.0-py2.py3-none-any.whl (48kB)
    [K    100% |████████████████████████████████| 51kB 1.8MB/s 
    [?25hRequirement already satisfied: torch in /usr/local/lib/python3.6/dist-packages (from torchvision)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from torchvision)
    Collecting pillow>=4.1.1 (from torchvision)
      Downloading Pillow-5.0.0-cp36-cp36m-manylinux1_x86_64.whl (5.9MB)
    [K    100% |████████████████████████████████| 5.9MB 234kB/s 
    [?25hRequirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from torchvision)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.6/dist-packages (from torch->torchvision)
    Installing collected packages: pillow, torchvision
      Found existing installation: Pillow 4.0.0
        Uninstalling Pillow-4.0.0:
          Successfully uninstalled Pillow-4.0.0
    Successfully installed pillow-5.0.0 torchvision-0.2.0
    


```python
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
```

## Importing Data and setting up Hyper-Parameters


```python
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data/',
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Processing...
    Done!
    

## CNN Model


```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        
cnn = CNN()
cnn.cuda() # for GPU
```




    CNN(
      (layer1): Sequential(
        (0): Conv2d (1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True)
        (2): ReLU()
        (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
      )
      (layer2): Sequential(
        (0): Conv2d (16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        (2): ReLU()
        (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
      )
      (fc): Linear(in_features=1568, out_features=10)
    )



## Using Adam Optimizer and Cross Entropy as loss


```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
```

## Training the model


```python
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

```

    Epoch [1/5], Iter [100/600] Loss: 0.1519
    Epoch [1/5], Iter [200/600] Loss: 0.1679
    Epoch [1/5], Iter [300/600] Loss: 0.1017
    Epoch [1/5], Iter [400/600] Loss: 0.0830
    Epoch [1/5], Iter [500/600] Loss: 0.0329
    Epoch [1/5], Iter [600/600] Loss: 0.1092
    Epoch [2/5], Iter [100/600] Loss: 0.0593
    Epoch [2/5], Iter [200/600] Loss: 0.0184
    Epoch [2/5], Iter [300/600] Loss: 0.0410
    Epoch [2/5], Iter [400/600] Loss: 0.0403
    Epoch [2/5], Iter [500/600] Loss: 0.0151
    Epoch [2/5], Iter [600/600] Loss: 0.0587
    Epoch [3/5], Iter [100/600] Loss: 0.0178
    Epoch [3/5], Iter [200/600] Loss: 0.0134
    Epoch [3/5], Iter [300/600] Loss: 0.0063
    Epoch [3/5], Iter [400/600] Loss: 0.0545
    Epoch [3/5], Iter [500/600] Loss: 0.0036
    Epoch [3/5], Iter [600/600] Loss: 0.0886
    Epoch [4/5], Iter [100/600] Loss: 0.0052
    Epoch [4/5], Iter [200/600] Loss: 0.0606
    Epoch [4/5], Iter [300/600] Loss: 0.0305
    Epoch [4/5], Iter [400/600] Loss: 0.0761
    Epoch [4/5], Iter [500/600] Loss: 0.0304
    Epoch [4/5], Iter [600/600] Loss: 0.0149
    Epoch [5/5], Iter [100/600] Loss: 0.0026
    Epoch [5/5], Iter [200/600] Loss: 0.0475
    Epoch [5/5], Iter [300/600] Loss: 0.0267
    Epoch [5/5], Iter [400/600] Loss: 0.0273
    Epoch [5/5], Iter [500/600] Loss: 0.0286
    Epoch [5/5], Iter [600/600] Loss: 0.0064
    

## Testing the model


```python
cnn.eval()   
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    
print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))
```

    Test Accuracy of the model on the 10000 test images: 98 %
    
