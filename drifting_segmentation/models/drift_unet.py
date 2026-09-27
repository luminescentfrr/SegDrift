import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        groups = min(groups, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class DriftUNet(nn.Module):
    """5-level U-Net generator compatible with the Drifting training interface."""

    def __init__(
        self,
        image_channels=3,
        noise_channels=2,
        out_channels=1,
        base_channels=64,
        residual_prior=True,
        prior_channel_index=1,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.noise_channels = noise_channels
        self.out_channels = out_channels
        self.residual_prior = residual_prior
        self.prior_channel_index = prior_channel_index
        in_channels = image_channels + noise_channels

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.enc4 = DownBlock(c3, c4)
        self.enc5 = DownBlock(c4, c5)

        self.dec4 = UpBlock(c5, c4, c4)
        self.dec3 = UpBlock(c4, c3, c3)
        self.dec2 = UpBlock(c3, c2, c2)
        self.dec1 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, out_channels, 1)

    @staticmethod
    def _prob_to_logits(prob):
        prob = prob.clamp(1e-4, 1 - 1e-4)
        return torch.log(prob / (1 - prob))

    def forward(self, image, noise_mask, cfg_scale=None, noise_labels=None):
        x = torch.cat([image, noise_mask], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d4 = self.dec4(e5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        logits = self.head(d1)

        if self.residual_prior and noise_mask.shape[1] > self.prior_channel_index:
            prior_prob = noise_mask[:, self.prior_channel_index : self.prior_channel_index + 1]
            logits = logits + self._prob_to_logits(prior_prob)
        return logits
