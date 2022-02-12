import torch
import torch.nn.functional as F


class CRN(torch.nn.Module):
    def __init__(self, dim):
        super(CRN, self).__init__()

        # Downsampling
        self.h_w = (13, 13)  # fixed "internal" image size
        self.downsample = torch.nn.AdaptiveAvgPool2d(self.h_w)

        # Filters
        n_filters_1 = 32  # 3x3
        n_filters_2 = 32  # 5x5
        n_filters_3 = 20  # 7x7
        self.conv1 = torch.nn.Conv2d(dim, n_filters_1, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(dim, n_filters_2, 5, padding=2)
        self.conv3 = torch.nn.Conv2d(dim, n_filters_3, 7, padding=3)
        self.conv_accum = torch.nn.Conv2d(n_filters_1 + n_filters_2 + n_filters_3, 1, 1)

        # Only for 13x13 -> 30x40
        # self.upsample = torch.nn.ConvTranspose2d(1, 1, kernel_size=5, stride=(2, 3), padding=(0, 1), output_padding=(1, 1))

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                # torch.nn.init.xavier_uniform_(m.weight)
                # if m.bias is not None:
                #     torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        input_h_w = x.shape[2:]

        x = self.downsample(x)
        # x = F.interpolate(x, self.h_w)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x = torch.cat((x1, x2, x3), dim=1)
        x = F.relu(self.conv_accum(x))

        # Upsampling to restore input HxW
        # x = self.upsample(x)
        x = F.interpolate(x, input_h_w)

        assert x.shape[2:] == input_h_w

        return x
