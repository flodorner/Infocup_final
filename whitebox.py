from config import *
import torch
from torch import nn, optim
import torch.nn.functional as f
from torch.utils.data import TensorDataset
import numpy as np


def create_whitebox(device):
    model = ResNet(num_blocks=(1, 1, 1, 1), prefilter=7, output_dim=43, norm=False).to(device)
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(WHITEBOX_DIRECTORY, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(WHITEBOX_DIRECTORY))
    return model


class ResNet(nn.Module):

    def __init__(self, num_blocks=(2, 2, 2, 2), output_dim=10, nc=3, prefilter=3, norm=True):
        super().__init__()
        self.norm = norm
        self.in_planes = 64
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=prefilter, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, output_dim)

    def _make_layer(self, BasicBlock, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlock(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        if self.norm:
            x = x / 256 - torch.mean(x / 256, [1, 2, 3]).view(-1, 1, 1, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = f.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.linear(x))
        return x


class BasicBlock(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


def train(model, device, loader, loss_function=nn.MSELoss(),
          optimizer=None):
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0)
    total_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        accuracy = torch.mean(torch.eq(torch.argmax(out, dim=1), torch.argmax(target, dim=1)).double()).item()
        print("trainig loss: %s accuracy: %s" % (total_loss / (batch_idx + 1), accuracy))

    return None


def evaluate(model, device, loader, loss_function=nn.MSELoss(), name="validation"):
    model.eval()
    (data, target) = iter(loader).next()
    data, target = data.to(device), target.to(device)
    out = model(data)
    loss = loss_function(out, target).item()
    accuracy = torch.mean(torch.eq(torch.argmax(out, dim=1), torch.argmax(target, dim=1)).double()).item()
    print("%s loss: %s accuracy: %s" % (name, loss, accuracy))

    return None


def simple_train_loop(model, device, train_loader, val_loader, save_directory, num_epochs=50):
    for epoch in range(1, num_epochs + 1):
        print('Now in epoch %d' % epoch)
        train(model, device, train_loader)
        evaluate(model, device, val_loader, name="validation")
        if epoch % int(num_epochs / 10) == 0:
            torch.save(model.state_dict(), save_directory)

    return None


def create_loader(images, labels):
    images = images.reshape(images.shape[0], images.shape[3], images.shape[1], images.shape[2])
    labels = labels.astype(np.float32)
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)
    dataset = TensorDataset(images, labels)
    return dataset
