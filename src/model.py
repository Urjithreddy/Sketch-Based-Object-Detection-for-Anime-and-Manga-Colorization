import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=6, n_down=4, num_filters=64):
        super(PatchDiscriminator, self).__init__()
        model = [
            nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.05)  # Add dropout
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_down):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                nn.Conv2d(num_filters * nf_mult_prev, num_filters * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(num_filters * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.05)  # Add dropout
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_down, 8)
        model += [
            nn.Conv2d(num_filters * nf_mult_prev, num_filters * nf_mult, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(num_filters * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)  # Add dropout
        ]
        model += [nn.Conv2d(num_filters * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)