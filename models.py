import torch
import torch.nn as nn



class VGG19(nn.Module):
    def __init__(self, num_classes=4):
        super(VGG19, self).__init__()

        self.features = nn.Sequential(
            self._make_block(3, 64, num_convs=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self._make_block(64, 128, num_convs=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self._make_block(128, 256, num_convs=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self._make_block(256, 512, num_convs=4),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self._make_block(512, 512, num_convs=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        #Fully Connected layers
        self.classifier = nn.Sequential(
            # After 5 max pools, a 224x224 image becomes 7x7
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_classes),
        )

    def _make_block(self, in_channels, out_channels, num_convs):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return torch.cat([x, out], 1) # instead of adding concatenate


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels) # transition layers are used 4 shrinking (via pooling) and reducing the channel count (via 1x1 conv)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return self.pool(out)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)  #
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity #Add the original input back BEFORE the final ReLU
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet18 has [2, 2, 2, 2] blocks per layer
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        # Global Average Pooling flattens the image flexibly regardless of size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DenseNet121(nn.Module):
    def __init__(self, num_classes=4, growth_rate=32):
        super(DenseNet121, self).__init__()

        # DenseNet121 has exactly this many layers in its 4 Dense Blocks
        block_config = [6, 12, 24, 16]
        num_channels = 64

        # 1. Initial Convolution Block
        self.features = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 2. Build the Dense Blocks and Transition Layers
        for i, num_layers in enumerate(block_config):
            # Build a Dense Block
            block_layers = []
            for _ in range(num_layers):
                block_layers.append(DenseLayer(num_channels, growth_rate))
                num_channels += growth_rate  # Channels grow by 32 every single layer

            self.features.add_module(f'denseblock_{i + 1}', nn.Sequential(*block_layers))

            # Add a Transition Layer after every block EXCEPT the last one
            if i != len(block_config) - 1:
                out_channels = num_channels // 2  # Compress channels by 50%
                self.features.add_module(f'transition_{i + 1}', TransitionLayer(num_channels, out_channels))
                num_channels = out_channels

        # 3. Final BatchNorm and ReLU
        self.features.add_module('norm5', nn.BatchNorm2d(num_channels))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        # 4. Global Average Pooling and Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise: groups=in_channels ensures each input channel gets its own filter
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # Pointwise: 1x1 convolution to combine the depthwise outputs
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, num_classes=4):
        super(MobileNet, self).__init__()

        # Standard initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Sequence of Depthwise Separable Convolutions
        # Format: (in_channels, out_channels, stride)
        self.features = nn.Sequential(
            DepthwiseSeparableConv(32, 64, 1),
            DepthwiseSeparableConv(64, 128, 2),
            DepthwiseSeparableConv(128, 128, 1),
            DepthwiseSeparableConv(128, 256, 2),
            DepthwiseSeparableConv(256, 256, 1),
            DepthwiseSeparableConv(256, 512, 2),
            # 5 repeating blocks of 512 channels
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            
            DepthwiseSeparableConv(512, 1024, 2),
            DepthwiseSeparableConv(1024, 1024, 1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # Branch 1: 1x1 Convolution
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # Branch 2: 1x1 Conv reduction -> 3x3 Conv
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        # Branch 3: 1x1 Conv reduction -> 5x5 Conv (using two 3x3s for efficiency)
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
            BasicConv2d(ch5x5, ch5x5, kernel_size=3, padding=1)
        )

        # Branch 4: MaxPool -> 1x1 Conv projection
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate outputs along the channel dimension (dim=1)
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class InceptionV3(nn.Module):
    def __init__(self, num_classes=4):
        super(InceptionV3, self).__init__()
        
        # Initial standard layers
        self.pre_layers = nn.Sequential(
            BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicConv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Format: in_channels, 1x1, 3x3red, 3x3, 5x5red, 5x5, pool_proj
        self.a3 = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.b3 = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.a4 = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.b4 = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate final channels: a4 outputs 512. b4 outputs 160+224+64+64 = 512.
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        x = self.a4(x)
        x = self.b4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x