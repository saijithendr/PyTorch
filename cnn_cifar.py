import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# setting up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 60000, 32x32 color images in 10 classes, with 6000 images per class

batch_size = 100
learning_rate = 0.001
num_epochs = 50

transform = transforms.Compose([ transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

## visualizing the data 
eg = iter(train_loader)

samples, labels = eg.__next__()
print(samples.shape, labels.shape)

for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0])

#plt.show()
#
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)     # 3-color, output-6, kernal-5
        self.pool = nn.MaxPool2d(2, 2)      # pooling---> kernal-2 maxpool-2
        self.conv2 = nn.Conv2d(6, 16, 5)    # Conv---> 6-input(must be same as output of prev conv), output-16, kernal-5
        self.fcn1 = nn.Linear(16*5*5, 120)  # fully NN1---> input-16*5*5, hidden-120    32-5+
        self.fcn2 = nn.Linear(120, 90)      # fully NN2---> hidden_input-120, hidden_output-80 
        self.fcn3 = nn.Linear(90, 10)       # fully NN3---> input-80, output-10 (num target classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        x = self.fcn3(x)
        return x
    
model = ConvNet().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.to(device)
        labels = labels.to(device)
        
        # forward pass               
        output = model(images)
        loss = criterion(output, labels)
        
        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')


print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
    
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')