import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TAG_Para"]

class ConvTAGpara(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class AttentionModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AttentionModule, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

class ContextGuidedBlock_Down(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        super().__init__()
        self.conv1x1 = ConvTAGpara(nIn, nOut, 3, 2)  # size/2, channel: nIn--->nOut
        self.F_loc = ConvTAGpara(nOut, nOut, 3, 1)  # Local feature extraction
        self.F_sur = nn.Conv2d(nOut, nOut, 3, padding=dilation_rate, dilation=dilation_rate, bias=False)  # Surrounding context

        self.bn = nn.BatchNorm2d(nOut * 2, eps=1e-3)
        self.act = nn.PReLU(nOut * 2)
        self.reduce = ConvTAGpara(nOut * 2, nOut, 1, 1)  # Reduce dimension
        self.attention = AttentionModule(nOut)  # Attention mechanism

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)  # Joint feature
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        output = self.reduce(joi_feat)  # Reduced feature
        output = self.attention(output)  # Apply attention

        return output

class Context_Guided_Network(nn.Module):
    def __init__(self, classes=19, M=3, N=21, dropout_flag=False):
        super().__init__()
        self.level1_0 = ConvTAGpara(3, 32, 3, 2)
        self.level1_1 = ConvTAGpara(32, 32, 3, 1)
        self.level1_2 = ConvTAGpara(32, 32, 3, 1)

        self.b1 = ConvTAGpara(32 + 3, 32, 1, 1)  # Adjusted from BNPReLU

        # Stage 2
        self.level2_0 = ContextGuidedBlock_Down(32 + 3, 64, dilation_rate=2, reduction=8)
        self.level2 = nn.ModuleList([ContextGuidedBlock_Down(64, 64, dilation_rate=2, reduction=8) for _ in range(M - 1)])

        # Stage 3
        self.level3_0 = ContextGuidedBlock_Down(128 + 3, 128, dilation_rate=4, reduction=16)
        self.level3 = nn.ModuleList([ContextGuidedBlock_Down(128, 128, dilation_rate=4, reduction=16) for _ in range(N - 1)])

        self.classifier = nn.Sequential(ConvTAGpara(128, classes, 1, 1))  # Adjusted from Conv

    def forward(self, x):
        # Level 1
        x1 = self.level1_0(x)
        x1 = self.level1_1(x1)
        x1 = self.level1_2(x1)

        # Pass through first block of stage 2
        x2 = self.level2_0(torch.cat([x1, x], dim=1))  # Concatenate input with output
        for layer in self.level2:
            x2 = layer(x2)

        # Level 3
        x3 = self.level3_0(torch.cat([x2, x], dim=1))  # Concatenate input with output
        for layer in self.level3:
            x3 = layer(x3)

        # Final classification
        output = self.classifier(x3)
        return output

# Now you can instantiate your model
model = Context_Guided_Network(classes=19)

# Example input
input_tensor = torch.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels, 256x256 image
output_tensor = model(input_tensor)
print(output_tensor.shape)  # Should match (1, classes, height, width)
