import torch
import torch.functional as F


class CRN(torch.nn.Module):
    def __init__(self, dim):
        super(CRN, self).__init__()

        out_conv1 = 32
        out_conv2 = 32
        out_conv3 = 20
        out_conv_accum = 1

        # self.downsample = torch.nn.AvgPool2d((3, 3), stride=(2, 2), padding=(0, 0))
        self.downsample = torch.nn.AdaptiveAvgPool2d((13, 13))

        self.conv1 = torch.nn.Conv2d(dim, out_conv1, 5, stride=1, padding=2)

        self.conv2 = torch.nn.Conv2d(dim, out_conv2, 3, stride=1, padding=1)

        self.conv3 = torch.nn.Conv2d(dim, out_conv3, 7, stride=1, padding=3)

        self.conv_accum = torch.nn.Conv2d(
            out_conv1 + out_conv2 + out_conv3, out_conv_accum, 1, stride=1
        )

        self.upsample = torch.nn.ConvTranspose2d(
            out_conv_accum, 1, 4, stride=2, padding=1, groups=dim
        )

        _init_weights(self.conv1)
        _init_weights(self.conv2)
        _init_weights(self.conv3)
        _init_weights(self.conv_accum)

    def forward(self, x):
        x = self.downsample(x)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))

        x = torch.cat((x1, x2, x3), dim=1)

        x = F.relu(self.conv_accum(x))

        x = self.upsample(x)

        return x


def _init_weights(c):
    if isinstance(c, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform(c.weight)
        c.bias.data.fill_(0)
