import torch
from torch import nn
import torch.nn.functional as f


def create_distilled(device):
    model = ResNet(num_blocks=(1, 1, 1, 1), prefilter=7, output_dim=43, norm=False).to(device)
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load("Models/ResNet.pt", map_location="cpu"))
    else:
        model.load_state_dict(torch.load("Models/ResNet.pt"))
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
