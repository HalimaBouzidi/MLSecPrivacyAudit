import torch.nn.functional as F
from torch import nn
from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, width_multi=1.0):
        super(SimpleCNN, self).__init__()

        base_ch = int(16*width_multi)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_ch, base_ch*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_ch*2, base_ch*2*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        self.classifier = nn.Sequential(
            nn.Linear(int(48 * 48 * width_multi), base_ch*2),
            nn.ReLU(),
            nn.Linear(base_ch*2, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, width_multi=1.0):
        super(AlexNet, self).__init__()
        self.width_multi = width_multi
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, int(16*width_multi), kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(int(64*width_multi), int(192*width_multi), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(int(192*width_multi), int(384*width_multi), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(384*width_multi), int(256*width_multi), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(256*width_multi), int(256*width_multi), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(int(256*width_multi) * 2 * 2, int(4096*width_multi)),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(int(4096*width_multi), int(4096*width_multi)),
            nn.ReLU(inplace=True),
            nn.Linear(int(4096*width_multi), num_classes),
        )

    def forward(self, inputs):
        inputs = self.features(inputs)
        inputs = inputs.reshape(inputs.size(0), int(256*self.width_multi) * 2 * 2)
        outputs = self.classifier(inputs)
        return outputs

class MobileNetV2(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, width_multi=1.0):
        super(MobileNetV2, self).__init__()
        self.cfgs = [
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(16 * width_multi)
        layers = [conv_bn(in_channels, input_channel, 2)]
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_multi)
            for i in range(n):
                if i == 0:
                    layers.append(InvertedResidual(input_channel, output_channel, s, expand_ratio=t))
                else:
                    layers.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        output_channel = int(1280 * width_multi) if width_multi > 1.0 else 1280
        layers.append(conv_1x1_bn(input_channel, output_channel))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, width_multi=1.0, depth=12):
        super(TransformerModel, self).__init__()        
        width = int(64*width_multi)
        self.transformer = models.VisionTransformer(image_size=48, patch_size=4, num_layers=depth, \
                                hidden_dim=width, mlp_dim=width, num_heads=8, num_classes=num_classes)
        self.transformer.conv_proj = nn.Conv2d(in_channels, width, kernel_size=(4, 4), stride=(4, 4))

    def forward(self, x):
        return self.transformer(x)
    

def get_model(args, model_name):

    if model_name == "cnn":
        return SimpleCNN(num_classes=args.num_classes)
    elif model_name == "alexnet":
        return AlexNet(num_classes=args.num_classes)
    elif model_name == "transformer":
        return TransformerModel(num_classes=args.num_classes)
    elif model_name == "mobilenet":
        return MobileNetV2(num_classes=args.num_classes)
    else:
        raise NotImplementedError(f"{model_name} is not implemented")