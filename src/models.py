import torch
from torch import nn
from torch.nn import functional as torch_f
import albumentations as A
from albumentations import pytorch as ATorch
import numpy as np
import cv2


IMAGE_SIZE = 128


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch_f.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.a = nn.Tanh()

    def forward(self, x):
        return self.a(self.conv(x))
    

class GeneratorWGAN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(GeneratorWGAN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    

def get_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Resize(p=1.0, height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize((0.5,), (0.5,), p=1.0),
            ATorch.transforms.ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def get_noise(*, n_samples: int, z_dim: int, device: torch.device):
    return torch.randn(n_samples, z_dim, device=device)


def combine_vectors(x, y):
    combined = torch.cat([x.float(), y.float()], dim=1)
    return combined


class Model:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen = GeneratorWGAN(
            n_channels=2, 
            n_classes=2,
        ).to(self.device)
        state_dict = torch.load("models/210_generator.pt", map_location=self.device)["model_state_dict"]
        self.gen.load_state_dict(state_dict)

        self.transforms = get_transforms()

    def __call__(self, image):
        with torch.no_grad():
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
            img = self.transforms(image=img)["image"]

            sample = {"image": img}

            gen_real = sample["image"].unsqueeze(0).to(self.device)

            fake_noise = get_noise(
                n_samples=1, 
                z_dim=128,
                device=self.device,
            ).unsqueeze(1).unsqueeze(2).expand(-1, -1, IMAGE_SIZE, IMAGE_SIZE)

            noise_and_labels = combine_vectors(gen_real, fake_noise)
            fake = self.gen(noise_and_labels)
            
            real = (gen_real.cpu().numpy()[0][0] + 1) / 2
            image_tensor = (fake[0, 0, :, :].detach().cpu().numpy() + 1) / 2
            mask_unflat = fake[0, 1, :, :].cpu().numpy()
            
            merged = np.where(mask_unflat > 0.5, image_tensor, real)
            
            # cv2.imwrite(f"tmp.png", merged * 255)
            # cv2.imwrite(f"tmp_mask.png", (mask_unflat > 0.5) * 255)

            return merged