import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# FC
class FC(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.fc(x)


# Gobal feature
class Gobal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        _, C, H, W = x.shape
        y = F.interpolate(x, size=[C, C], mode='bilinear', align_corners=True)
        # b c w h -> b c h w
        y = self.act1(self.conv1(y)).permute(0, 1, 3, 2)
        # b c h w -> b w h c
        y = self.act2(self.conv2(y)).permute(0, 3, 2, 1)
        # b w h c -> b c w h
        y = self.act3(self.conv3(y)).permute(0, 3, 1, 2)
        y = F.interpolate(y, size=[H, W], mode='bilinear', align_corners=True)
        return x + y
    

class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = nn.ReLU()
        self.norm2 = nn.ReLU()

        self.gobal = Gobal(dim)
        self.conv = nn.Conv3d(2, 1, 3, 1, 1)
        self.fc = FC(dim, ffn_scale) 

    def forward(self, x):
        y = self.norm1(x)
        y_g = self.gobal(y)
        y = self.conv(torch.cat([y.unsqueeze(1), y_g.unsqueeze(1)], dim=1)).squeeze(1) + x
        y = self.fc(self.norm2(y)) + y
        return y
    


class SAFMN(nn.Module):
    def __init__(self, dim, n_blocks=16, ffn_scale=2.0):
        super().__init__()
        self.to_feat = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1),
        )
        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x


class DAMixNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = SAFMN(dim=48, n_blocks=8, ffn_scale=2.0)
        self.model2 = SAFMN(dim=48, n_blocks=8, ffn_scale=2.0)
        self.model3 = SAFMN(dim=48, n_blocks=8, ffn_scale=2.0)
        self.model4 = SAFMN(dim=48, n_blocks=8, ffn_scale=2.0)
        
        self.fconv = nn.Conv2d(48 * 4, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        
        x1 = F.interpolate(x, size=[32, 32], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x, size=[64, 64], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x, size=[128, 128], mode='bilinear', align_corners=True)    
        
        x1 = self.model1(x1)
        x2 = self.model1(x2)
        x3 = self.model1(x3)
        x4 = self.model1(x)  
        
        x1 = F.interpolate(x1, size=[x.shape[2], x.shape[3]], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=[x.shape[2], x.shape[3]], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=[x.shape[2], x.shape[3]], mode='bilinear', align_corners=True)  
        
        output = self.fconv(torch.cat([x1, x2, x3, x4], dim=1))
        return output
from thop import profile

if __name__== '__main__': 
    x = torch.randn(1, 3, 256, 256)
    model = DAMixNet()
    flops, params = profile(model, inputs=(x, ))
 
    print(f'FLOPS: {flops / 1000000.0}')
    print(f'Params: {params / 1000000.0}')
    #from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    #print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    #print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    with torch.no_grad():
        start_time = time.time()
        output = model(x)
        end_time = time.time()
    running_time = end_time - start_time
    print(output.shape)
    print(running_time)