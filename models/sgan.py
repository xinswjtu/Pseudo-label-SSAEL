import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=None, n_class=None, img_size=None, out_channels=3):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_class, z_dim)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, y):
        x = torch.mul(self.label_emb(y), z)
        x = x.view(-1, 100, 1, 1)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, n_class=None, img_size=None, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.adv = nn.Linear(512, 1)
        self.aux = nn.Linear(512, n_class)

    def forward(self, x, *args, **kwargs):
        return_features = kwargs.pop('return_features', False)
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        if return_features:
            return x
        else:
            return self.adv(x), self.aux(x)


