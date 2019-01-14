import torch
from torch import nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()
        self.activation = torch.relu
        planes = in_planes*stride
            
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
                ,nn.BatchNorm2d(planes))

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out
    
    
class ResNet(nn.Module):
    
    def __init__(self, num_Blocks=(2,2,2,2), output_dim=10, nc=3):
        super().__init__()
        self.activation = torch.relu

        self.in_planes = 64
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, output_dim)

    def _make_layer(self, BasicBlock, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlock(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.linear(x))
    
class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x
    
class ResBlockIN(nn.Module):
    
    def __init__(self, nc):
        super().__init__()
        self.conv1 = nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.in1 = nn.InstanceNorm2d(nc)
        self.conv2 = nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.in2 = nn.InstanceNorm2d(nc)
    
    def forward(self, X):
        x = torch.relu(self.in1(self.conv1(X)))
        x = self.in2(self.conv2(x))
        x = torch.relu(x + X)
        return x


class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()
        self.activation = nn.LeakyReLU()
        planes = in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1, dropout=False):
        super().__init__()
        self.activation = nn.LeakyReLU()
        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.rg2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )
        if dropout:
            self.rg1 = nn.Dropout()
            self.rg2 = nn.Dropout()

    def forward(self, x):
        out = self.activation(self.bn2(self.conv2(x)))
        out = self.rg1(self.conv1(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ImageToLatent(nn.Module):

    def __init__(self, num_Blocks=(2,2,2,2), z_dim=10, nc=3, dropout=False):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.rg1 = nn.BatchNorm2d(64)
        if dropout:
            self.rg1 = nn.Dropout()
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.activation(self.rg1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class LatentToImage(nn.Module):

    def __init__(self, num_Blocks=(2,2,2,2), z_dim=10, nc=3):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))  # TODO: What about Batch Norm? After layer1?
        x = x.view(x.size(0), 3, 64, 64)
        return x


class ConditionalVAE(nn.Module):

    def __init__(self, bottleneck_size, len_label_vector):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.encoder = ImageToLatent(z_dim=bottleneck_size * 2)
        self.decoder = LatentToImage(z_dim=bottleneck_size + len_label_vector)

    def forward(self, x, y):
        x = self.encoder(x)
        mu = x[:, :self.bottleneck_size]
        logvar = x[:, self.bottleneck_size:]
        sample = self.reparameterize(mu, logvar)
        latent_variables = torch.cat((sample, y), dim=1)
        reconstruction = self.decoder(latent_variables)
        return reconstruction, mu, logvar
            
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 