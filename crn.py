import torch
import torch.nn.functional as F


class CRN(torch.nn.Module):
    def __init__(self, dim):
        super(CRN, self).__init__()

        # Fixed "internal" image HxW
        self.h_w = (13, 13)

        # self.downsample = torch.nn.AdaptiveAvgPool2d(self.h_w)

        out_ch_conv1 = 32
        self.conv1 = torch.nn.Conv2d(dim, out_ch_conv1, 3, padding=1)

        out_ch_conv2 = 32
        self.conv2 = torch.nn.Conv2d(dim, out_ch_conv2, 5, padding=2)

        out_ch_conv3 = 20
        self.conv3 = torch.nn.Conv2d(dim, out_ch_conv3, 7, padding=3)

        self.conv_accum = torch.nn.Conv2d(
            out_ch_conv1 + out_ch_conv2 + out_ch_conv3, 1, 1
        )

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
        assert len(input_h_w) == 2

        # x = self.downsample(x)
        x = F.interpolate(x, self.h_w)

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
