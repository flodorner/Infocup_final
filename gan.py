import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from config import GAN_SPECS, FACES_DIRECTORY, GAN_DIRECTORY
from whitebox import create_whitebox
if GAN_SPECS['use_faces_dataset']:
    from data import Faces

if GAN_SPECS['cuda']:
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
        print('Warning: Specified use of CUDA, but CUDA is not available. Proceeding on CPU')



class PretrainedGenerator:

    def __init__(self):
        self.model = self.create_generator()

    def perturb_image(self, image):
        X, _ = self.model(image)
        return X

    @staticmethod
    def create_generator():
        state_dict = torch.load(GAN_DIRECTORY)
        model = G(3).to(DEVICE)
        model.load_state_dict(state_dict)
        return model

class BasicBlock(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()
        self.activation = F.LeakyReLU
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

class D(nn.Module):

    def __init__(self, num_Blocks=(1,1,1,1), z_dim=1, nc=3):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
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
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.linear(x))
        return x

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
        self.activation = F.leaky_relu

        super().__init__()
        self.conv1 = nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.in1 = nn.InstanceNorm2d(nc)
        self.conv2 = nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.in2 = nn.InstanceNorm2d(nc)

    def forward(self, X):
        x = self.activation(self.in1(self.conv1(X)))
        x = self.in2(self.conv2(x))
        x = self.activation(x + X)
        return x


class ImageToImage(nn.Module):

    def __init__(self, nc=3):
        super().__init__()

        self.activation = F.leaky_relu

        self.conv1 = nn.Conv2d(nc, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.in1 = nn.InstanceNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=True)
        self.in2 = nn.InstanceNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.in3 = nn.InstanceNorm2d(32)
        self.residual_blocks = nn.ModuleList([ResBlockIN(32) for _ in range(2)])
        self.conv4 = ResizeConv2d(32, 16, kernel_size=3, scale_factor=2)
        self.in4 = nn.InstanceNorm2d(32)
        self.conv5 = ResizeConv2d(16, 8, kernel_size=3, scale_factor=2)
        self.in5 = nn.InstanceNorm2d(32)
        self.conv6 = ResizeConv2d(8, nc, kernel_size=3, scale_factor=1)
        self.in6 = nn.InstanceNorm2d(32)

    def forward(self, x):
        x = self.activation(self.in1(self.conv1(x)))
        x = self.activation(self.in2(self.conv2(x)))
        x = self.activation(self.in3(self.conv3(x)))
        for block in self.residual_blocks:
            x = block(x)
        x = self.activation(self.in4(self.conv4(x)))
        x = self.activation(self.in5(self.conv5(x)))
        x = torch.tanh(self.in6(self.conv6(x)))
        x = torch.mul(x, 0.3)
        return x


class G(nn.Module):

    def __init__(self, nc):
        super().__init__()
        self.perturbation = ImageToImage(nc=nc)

    def forward(self, x):
        eps = self.perturbation(x)
        X = torch.clamp(x + eps, 0, 1)
        return X, eps

adversarial_loss = nn.MSELoss()

def perturbation_loss(perturbation):
    loss = torch.mean(torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1))
    return loss

def carlini_wagner_loss(pred_c, target):
    target_onehot = torch.eye(43)[target].to(DEVICE)
    not_target_onehot = (target_onehot - 1) * (-1)
    max_f_i, _ = torch.max(pred_c*not_target_onehot, dim=1)
    difference = max_f_i - pred_c[:,target]
    zeros = torch.zeros((GAN_SPECS['batch_size'], 1)).to(DEVICE)
    loss_adv = torch.max(difference, zeros)
    return loss_adv.sum()

if GAN_SPECS['use_faces_dataset']:
    data_set = Faces(FACES_DIRECTORY)
    train_size = int(0.8 * len(data_set))
    test_size = len(data_set) - train_size
    training_set, test_set = random_split(data_set, [train_size, test_size])
    trainloader = DataLoader(training_set, batch_size=GAN_SPECS['batch_size'], shuffle=True, drop_last=True)
else:
    print('Warning: If not using faces dataset, you must specify your own dataset!')

def train(num_epochs, target):

    generator = G(3).to(DEVICE)
    optim_generator = optim.Adam(generator.parameters(), lr=1e-4)

    discriminator = D().to(DEVICE)
    optim_discriminator = optim.SGD(discriminator.parameters(), lr=1e-4)

    classifier = create_whitebox(DEVICE)

    for epoch in tqdm(range(num_epochs)):
        generator.train()
        discriminator.train()
        classifier.eval()

        for i, (X, y) in enumerate(trainloader):

            batch_size = len(y)

            # 1) Train discriminator
            discriminator.zero_grad()
            labels_zeros = torch.randn((batch_size, 1)) * 0.1
            labels_ones = torch.ones((batch_size, 1)) - torch.randn((batch_size, 1)) * 0.1

            # a) On real images
            X = X.to(DEVICE)
            pred_real = discriminator(X)
            loss_real = adversarial_loss(pred_real, labels_zeros.to(DEVICE))

            # b) On fake images
            X_fake, _ = generator(X)
            pred_fake = discriminator(X_fake.detach())
            loss_fake = adversarial_loss(pred_fake, labels_ones.to(DEVICE))

            loss_generator = loss_fake + loss_real
            loss_generator.backward()

            optim_discriminator.step()

            # 2) Train generator
            generator.zero_grad()

            # a) Classification Loss
            X_fake, pert = generator(X)
            pred_classifier = classifier(X_fake * 255)
            target = (torch.ones((batch_size,)) * target)
            loss_classifier = carlini_wagner_loss(pred_classifier, target)

            # b) Perturbation Loss
            loss_pert = perturbation_loss(pert)

            # c) Adversarial Loss
            pred_discriminator = discriminator(X_fake)
            loss_adversarial = adversarial_loss(pred_discriminator, torch.zeros((batch_size, 1)).to(DEVICE))

            loss_generator = GAN_SPECS['epsilon'] * loss_classifier + GAN_SPECS['alpha'] * loss_adversarial \
                             + GAN_SPECS['beta'] * loss_pert
            loss_generator.backward()

            optim_generator.step()

    torch.save(generator.state_dict(), GAN_SPECS['modelsaving_directory'] + 'G' + '_' + str(target) + '_' + str(epoch) + '.pt')
    torch.save(discriminator.state_dict(), GAN_SPECS['modelsaving_directory'] + 'G' + '_' + str(target) + '_' + str(epoch) + '.pt')
    torch.cuda.empty_cache()
    return generator, discriminator


