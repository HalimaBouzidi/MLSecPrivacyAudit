import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class SearchableAlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=100, depth=3, width_ratio=1.0):
        super(SearchableAlexNet, self).__init__()
        self.in_channels = in_channels
        widths_orig = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512]
        widths = [int(width_ratio * widths_orig[i]) for i in range(depth)]
        self.features = self.make_layers(depth, widths)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        final_width = widths[-1] if depth <= len(widths) else widths[depth % len(widths) - 1]
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(final_width * 2 * 2, final_width * 4),  # Adjusted for smaller feature map size
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(final_width * 4, final_width * 4),
            nn.ReLU(inplace=True),
            nn.Linear(final_width * 4, num_classes),
        )

    def make_layers(self, depth, widths):
        layers = []
        in_channels = 3
        for i in range(depth):
            out_channels = widths[i % len(widths)]
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=1, stride=1)]
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SearchableResNet(nn.Module):
    def __init__(self, block=BasicBlock, in_channels=3, num_classes=100, depth=10, width_multiplier=1):
        super(SearchableResNet, self).__init__()
        self.in_planes = 64

        # Divide depth evenly across four layers, distribute any remainder
        blocks_per_layer = [depth // 4 + (1 if i < depth % 4 else 0) for i in range(4)]

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, int(64 * width_multiplier), blocks_per_layer[0], stride=1)
        self.layer2 = self._make_layer(block, int(128 * width_multiplier), blocks_per_layer[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * width_multiplier), blocks_per_layer[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * width_multiplier), blocks_per_layer[3], stride=2)
        self.linear = nn.Linear(int(512 * block.expansion * width_multiplier), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class SearchableTransformer(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, depth=8, width_multi=1.0):
        super(SearchableTransformer, self).__init__()        
        width = int(64*width_multi)
        self.transformer = models.VisionTransformer(image_size=32, patch_size=4, num_layers=depth, \
                                hidden_dim=width, mlp_dim=width, num_heads=8, num_classes=num_classes)
        self.transformer.conv_proj = nn.Conv2d(in_channels, width, kernel_size=(4, 4), stride=(4, 4))

    def forward(self, x):
        return self.transformer(x)
    
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class SearchableMobileNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=100, depth_mult=1.0, width_mult=1.0):
        super(SearchableMobileNet, self).__init__()
        self.in_channels = in_channels
        cfg = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 1, 2],
        ]

        # Adjust number of layers based on depth multiplier
        new_cfg = []
        for c, n, s in cfg:
            n = max(round(n * depth_mult), 1)  # Ensure there's at least 1 layer
            new_cfg.append([c, n, s])

        self.features = self._make_layers(new_cfg, width_mult)

        # Assuming the input size of CIFAR-100 is 32x32
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(int(1024 * width_mult), num_classes),
        )

    def _make_layers(self, cfg, width_mult):
        layers = []
        input_channel = self.in_channels
        for c, n, s in cfg:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    layers.append(conv_dw(input_channel, output_channel, s))
                else:
                    layers.append(conv_dw(input_channel, output_channel, 1))
                input_channel = output_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, int(x.size(1)))
        x = self.classifier(x)
        return x

def get_model(args):

    if args['train']['model_name'] == "searchable_transformer":
        return SearchableTransformer(in_channels=args['data']['in_channels'], num_classes=args['data']['num_classes'], \
                                 depth=args['train']['depth_multi'], width_multi=args['train']['width_multi'])
    elif args['train']['model_name'] == "searchable_alexnet":
        return SearchableAlexNet(in_channels=args['data']['in_channels'], num_classes=args['data']['num_classes'], \
                                 depth=args['train']['depth_multi'], width_ratio=args['train']['width_multi'])
    elif args['train']['model_name'] == "searchable_resnet":
        return SearchableResNet(in_channels=args['data']['in_channels'], num_classes=args['data']['num_classes'], \
                                 depth=args['train']['depth_multi'], width_multiplier=args['train']['width_multi'])
    elif args['train']['model_name'] == "searchable_mobilenet":
        return SearchableMobileNet(in_channels=args['data']['in_channels'], num_classes=args['data']['num_classes'], \
                                 depth_mult=args['train']['depth_multi'], width_mult=args['train']['width_multi'])
    else:
        raise NotImplementedError(f"{args['train']['model_name']} is not implemented")