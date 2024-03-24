import torch
import torch.nn as nn
import torch.nn.functional as F

############# Stage 1 #############
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class FourierBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.freq = nn.Conv2d(in_c, in_c, 1, 1, 0)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_c, in_c, 1, 1, 0))
        
    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.freq(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.conv(mag)
        real = mag * torch.cos(pha)
        img = mag * torch.sin(pha)
        x_out = torch.complex(real, img)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x + x_out
    

class AmplitudeUNet(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.encoder0 = nn.Sequential(
            nn.Conv2d(3, in_c, 1, 1, 0),
            FourierBlock(in_c))
        self.down = Downsample(in_c)
        self.encoder1 = FourierBlock(in_c*2)

        self.latent = FourierBlock(in_c*2)

        self.decoder1 = nn.Sequential(
            FourierBlock(in_c*4),
            nn.Conv2d(in_c*4, in_c*2, 1, 1, 0),
        )
        self.up = Upsample(in_c*2)
        self.decoder0 = nn.Sequential(
            FourierBlock(in_c * 2),
            nn.Conv2d(in_c * 2, 3, 1, 1, 0),
            nn.Sigmoid()
        )

        
    def forward(self, x):
        en0 = self.encoder0(x)
        en1 = self.down(en0)
        en1 = self.encoder1(en1)
        
        lat = self.latent(en1)

        de1 = self.decoder1(torch.cat((lat, en1), dim=1))
        de0 = self.up(de1)
        de0 = self.decoder0(torch.cat((de0, en0), dim=1))

        _, _, H, W = x.shape
        image_fft = torch.fft.fft2(x, norm='backward')
        mag_image = torch.abs(image_fft)
        pha_image = torch.angle(image_fft)
        mag_image = mag_image / (de0 + 0.00000001)
        real_image_enhanced = mag_image * torch.cos(pha_image)
        imag_image_enhanced = mag_image * torch.sin(pha_image)
        img_amp_enhanced = torch.fft.ifft2(torch.complex(real_image_enhanced, imag_image_enhanced), s=(H, W),
                                    norm='backward').real
        return img_amp_enhanced
    

############# Stage 2 #############
class BaseBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_c, in_c, 1, 1, 0))
        
    def forward(self, x):
        x_out = self.conv(x)
        return x + x_out


class BaseUNet(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.encoder0 = nn.Sequential(
            nn.Conv2d(3, in_c, 1, 1, 0),
            BaseBlock(in_c))
        self.down = Downsample(in_c)
        self.encoder1 = BaseBlock(in_c*2)

        self.latent = BaseBlock(in_c*2)

        self.decoder1 = nn.Sequential(
            BaseBlock(in_c*4),
            nn.Conv2d(in_c*4, in_c*2, 1, 1, 0),
        )
        self.up = Upsample(in_c*2)
        self.decoder0 = nn.Sequential(
            BaseBlock(in_c * 2),
            nn.Conv2d(in_c * 2, 3, 1, 1, 0)
        )

        
    def forward(self, x):
        en0 = self.encoder0(x)
        en1 = self.down(en0)
        en1 = self.encoder1(en1)
        
        lat = self.latent(en1)

        de1 = self.decoder1(torch.cat((lat, en1), dim=1))
        de0 = self.up(de1)
        de0 = self.decoder0(torch.cat((de0, en0), dim=1))
        return de0


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1,
                 use_bias=True, activation=nn.PReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size,
                              padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class GuideNN(nn.Module):
    def __init__(self, bn=True):
        super(GuideNN, self).__init__()

        self.conv1 = ConvBlock(3, 16, kernel_size=3, padding=1, batch_norm=bn)
        self.conv2 = ConvBlock(16, 16, kernel_size=3, padding=1, batch_norm=bn)
        self.conv3 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.conv3(output)

        return output

class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1)  # norm to [0,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1)  # norm to [0,1] NxHxWx1
        hg, wg = hg * 2 - 1, wg * 2 - 1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)  # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, align_corners=True)
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]
        result = torch.cat([R, G, B], dim=1)

        return result

class B_transformer(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.guide = GuideNN()

        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()
        self.p = nn.PReLU()
        self.point = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        self.unet = BaseUNet(in_c)
        self.fitune = BaseUNet(in_c)

    def forward(self, x):
        # 8x down: (3840 2160) -> (480, 270)
        x_r = F.interpolate(x, (480, 270), mode='bicubic')
        coeff = self.unet(x_r).reshape(-1, 12, 36, 30, 30)
        guidance = self.guide(x)
        slice_coeffs = self.slice(coeff, guidance)
        output = self.apply_coeffs(slice_coeffs, self.p(self.point(x))) 
        output = self.fitune(output)
        return output


############# Main Model #############

class MainModel(nn.Module):
    def __init__(self, c=8):
        super().__init__()
        self.AmpNet = AmplitudeUNet(c)
        self.BilateralNet = B_transformer(c)

    def forward(self, x):
        x_coarse = self.AmpNet(x)
        x_fine = self.BilateralNet(x)
        return x_coarse, x_fine

if __name__ == "__main__":
    model  = MainModel().cuda()
    x = torch.randn(1, 3, 3840, 2160).cuda()
    # with torch.no_grad():
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    y1, y2 = model(x)
    print(y1.shape)
    print(y2.shape)