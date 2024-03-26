import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, width_expand=1.0, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.widths = [32, 64, 128, 256]
        self.expanded = [int(elem*width_expand) for elem in self.widths]
        self.conv1 = nn.Conv2d(3, self.expanded[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.expanded[0], self.expanded[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.expanded[1], self.expanded[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.expanded[2], self.expanded[3], kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(self.expanded[3] * 2 * 2, self.expanded[3]*2)  # 2*2 comes from image dimension reduction
        self.fc2 = nn.Linear(self.expanded[3]*2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, self.expanded[3] * 2 * 2)
        
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return x

def DepthwiseSeparableConv(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class DepthCNN(nn.Module):
    def __init__(self, width_expand=1.0, num_classes=10):
        super(DepthCNN, self).__init__()
        self.widths = [32, 64, 128, 256]
        self.expanded = [int(elem*width_expand) for elem in self.widths]
        self.conv1 = DepthwiseSeparableConv(3, self.expanded[0], 1)
        self.conv2 = DepthwiseSeparableConv(self.expanded[0], self.expanded[1], 1)
        self.conv3 = DepthwiseSeparableConv(self.expanded[1], self.expanded[2], 1)
        self.conv4 = DepthwiseSeparableConv(self.expanded[2], self.expanded[3], 1)

        self.fc1 = nn.Linear(self.expanded[3] * 2 * 2, self.expanded[3]*2)
        self.fc2 = nn.Linear(self.expanded[3]*2, num_classes)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.max_pool2d(self.conv3(x), 2)
        x = F.max_pool2d(self.conv4(x), 2)

        x = x.view(-1, self.expanded[3] * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BottleneckConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(BottleneckConv, self).__init__()
        # 1x1 convolution for dimensionality reduction
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        # 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # 1x1 convolution to expand dimensions
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        return out

class BottleCNN(nn.Module):
    def __init__(self, width_expand=1.0, num_classes=10):
        super(BottleCNN, self).__init__()
        self.widths = [32, 64, 128, 256]
        self.expanded = [int(elem*width_expand) for elem in self.widths]
        self.conv1 = BottleneckConv(3, int(self.expanded[0]/4), self.expanded[0])
        self.conv2 = BottleneckConv(self.expanded[0], int(self.expanded[1]/4), self.expanded[1])
        self.conv3 = BottleneckConv(self.expanded[1], int(self.expanded[2]/4), self.expanded[2])
        self.conv4 = BottleneckConv(self.expanded[2], int(self.expanded[3]/4), self.expanded[3])

        self.fc1 = nn.Linear(self.expanded[3] * 2 * 2, self.expanded[3])
        self.fc2 = nn.Linear(self.expanded[3], num_classes)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.max_pool2d(self.conv3(x), 2)
        x = F.max_pool2d(self.conv4(x), 2)
        x = x.view(-1, self.expanded[3] * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1x1 convolution for dimensionality reduction
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        # 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # 1x1 convolution to expand dimensions
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        out = self.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, width_expand=1.0, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.widths = [32, 64, 128, 256]
        self.expanded = [int(elem*width_expand) for elem in self.widths]

        self.block1 = BottleneckBlock(3, int(self.expanded[0]/4), self.expanded[0], stride=1)
        self.block2 = BottleneckBlock(self.expanded[0], int(self.expanded[1]/4), self.expanded[1], stride=2)  # Stride=2 for downsampling
        self.block3 = BottleneckBlock(self.expanded[1], int(self.expanded[2]/4), self.expanded[2], stride=2)  # Stride=2 for downsampling
        self.block4 = BottleneckBlock(self.expanded[2], int(self.expanded[3]/4), self.expanded[3], stride=2)  # Stride=2 for downsampling

        # Fully connected layers
        self.fc1 = nn.Linear(self.expanded[3] * 2, self.expanded[3])  # Adjusted for the downsampled image size
        self.fc2 = nn.Linear(self.expanded[3], num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Global Average Pooling
        x = F.avg_pool2d(x, 4)
        x = x.view(-1, self.expanded[3] * 2)  # Flatten the output for the fully connected layer

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AsymmetricConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(AsymmetricConv2d, self).__init__()
        # First asymmetric convolution: 1xN
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, padding))
        # Second asymmetric convolution: Nx1
        self.conv2 = nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(padding, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class AsymmetricCNN(nn.Module):
    def __init__(self, width_expand=1.0, num_classes=10):
        super(AsymmetricCNN, self).__init__()
        self.widths = [32, 64, 128, 256]
        self.expanded = [int(elem*width_expand) for elem in self.widths]
        
        self.conv1 = AsymmetricConv2d(3, self.expanded[0], kernel_size=3, padding=1)
        self.conv2 = AsymmetricConv2d(self.expanded[0], self.expanded[1], kernel_size=3, padding=1)
        self.conv3 = AsymmetricConv2d(self.expanded[1], self.expanded[2], kernel_size=3, padding=1)
        self.conv4 = AsymmetricConv2d(self.expanded[2], self.expanded[3], kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.expanded[3] * 2 * 2, self.expanded[3]*2)
        self.fc2 = nn.Linear(self.expanded[3]*2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, self.expanded[3] * 2 * 2)
        
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return x

class SimpleViT(nn.Module):
    def __init__(self, in_channels=3, width_expand=1.0, num_classes=10, depth=3):
        super(SimpleViT, self).__init__()        
        width = int(128*width_expand)
        self.transformer = models.VisionTransformer(image_size=32, patch_size=4, num_layers=depth, \
                                hidden_dim=width, mlp_dim=width, num_heads=8, num_classes=num_classes)
        self.transformer.conv_proj = nn.Conv2d(in_channels, width, kernel_size=(4, 4), stride=(4, 4))

    def forward(self, x):
        return self.transformer(x)

def get_model(args):

    if args['train']['model_name'] == "simple_cnn":
        return SimpleCNN(width_expand=args['train']['width_multi'], num_classes=args['data']['num_classes'])
    elif args['train']['model_name'] == "depth_cnn":
        return DepthCNN(width_expand=args['train']['width_multi'], num_classes=args['data']['num_classes'])
    elif args['train']['model_name'] == "bottle_cnn":
        return BottleCNN(width_expand=args['train']['width_multi'], num_classes=args['data']['num_classes'])
    elif args['train']['model_name'] == "resnet_cnn":
        return SimpleResNet(width_expand=args['train']['width_multi'], num_classes=args['data']['num_classes'])
    elif args['train']['model_name'] == "asymetric_cnn":
        return AsymmetricCNN(width_expand=args['train']['width_multi'], num_classes=args['data']['num_classes'])
    elif args['train']['model_name'] == "simple_vit":
        return SimpleViT(width_expand=args['train']['width_multi'], num_classes=args['data']['num_classes'])
    else:
        raise NotImplementedError(f"{args['train']['model_name']} is not implemented")

if __name__ == '__main__':
    
    widths = [0.25, 0.5, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0]
    for width in widths:
        model = SimpleViT(width_expand=width, num_classes=10)
        print(model(torch.rand(1, 3, 32, 32)).size())