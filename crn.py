import torch
import torch.nn.functional as F


class CRN(torch.nn.Module):
    def __init__(self, dim):
        super(CRN, self).__init__()

        # self.downsample = torch.nn.AvgPool2d((3, 3), stride=(2, 2), padding=(0, 0))
        self.downsample = torch.nn.AdaptiveAvgPool2d(13)

        out_ch_conv1 = 32
        self.conv1 = torch.nn.Conv2d(dim, out_ch_conv1, 3, padding=1)

        out_ch_conv2 = 32
        self.conv2 = torch.nn.Conv2d(dim, out_ch_conv2, 5, padding=2)

        out_ch_conv3 = 20
        self.conv3 = torch.nn.Conv2d(dim, out_ch_conv3, 7, padding=3)

        self.conv_accum = torch.nn.Conv2d(
            out_ch_conv1 + out_ch_conv2 + out_ch_conv3, 1, 1
        )

        # self.upsample = torch.nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, groups=1)

        self.apply(_init_weights)

    def forward(self, x):
        input_h_w = x.shape[2:]
        assert len(input_h_w) == 2

        x = self.downsample(x)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x = torch.cat((x1, x2, x3), dim=1)
        x = F.relu(self.conv_accum(x))

        # Upsampling to restore input HxW
        x = F.interpolate(x, input_h_w)

        return x


def _init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        # torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
