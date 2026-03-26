import torch
from model.basic_block import *
from model.mamba2 import BiMamba2Ac2d
def conv5x5_relu(in_channels, out_channels, stride):
    return conv(in_channels, out_channels, 5, stride, 2, activation_fn=partial(nn.ReLU, inplace=True))


def deconv5x5_relu(in_channels, out_channels, stride, output_padding):
    return deconv(in_channels, out_channels, 5, stride, 2, output_padding=output_padding,
                  activation_fn=partial(nn.ReLU, inplace=True))

def resblock(in_channels):
    return BasicBlock(in_channels, out_channels=in_channels, kernel_size=5, stride=1, use_batchnorm=False,
                      activation_fn=partial(nn.ReLU, inplace=True), last_activation_fn=None)


class EBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(type(self), self).__init__()
        # 5x5conv层
        self.conv = conv5x5_relu(in_channels, out_channels, stride)
        # 3个ResBlock块
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(out_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)

    def forward(self, x):
        x = self.conv(x)
        x = self.resblock_stack(x)
        return x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, output_padding):
        super(type(self), self).__init__()
        # 3个ResBlock块
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        # 5x5deconv层
        self.deconv = deconv5x5_relu(in_channels, out_channels, stride, output_padding)

    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.deconv(x)
        return x


class OutBlock(nn.Module):
    def __init__(self, in_channels):
        super(type(self), self).__init__()
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        self.conv = conv(in_channels, 3, 5, 1, 2, activation_fn=None)

    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.conv(x)
        return x


class OSRNet(nn.Module):

    def __init__(self, upsample_fn=partial(torch.nn.functional.interpolate, mode='bilinear'), xavier_init_all=True):
        super(type(self), self).__init__()
        self.upsample_fn = upsample_fn
        self.input_padding = None
        self.input_padding_h = None # 记录上轮的图片输出

        # 输入块
        self.inblock = EBlock(3 + 3, 16, 1)     # 这里的3+3意思是原本输入图像具有3通道,从上一个输出图像具有3通道
        # 编码块(通道c倍增,高h宽w减半)
        self.eblock1 = EBlock(16, 32, 2)
        self.eblock2 = EBlock(32, 64, 2)

        # BiMamba2单层
        self.bimamba2 = BiMamba2Ac2d(128, 64, 32)

        # 解码块(通道c倍减,高h宽w翻倍)
        self.dblock1 = DBlock(64, 32, 2, 1)
        self.dblock2 = DBlock(32, 16, 2, 1)
        # 输出块
        self.outblock = OutBlock(16)

        # 初始化参数
        if xavier_init_all:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    torch.nn.init.xavier_normal_(m.weight)

    def forward_step(self, x, hidden_state):

        e32 = self.inblock(x)
        e64 = self.eblock1(e32)
        e128 = self.eblock2(e64)

        h = self.bimamba2(torch.cat([e128, hidden_state],dim=1)) # 解码块+输出块(通道128->64->32->3,h/4和w/4在两层解码块变为h和w)

        d64 = self.dblock1(h)
        d32 = self.dblock2(d64 + e64)   # 含残差块
        d3 = self.outblock(d32 + e32)   # 含残差块
        return e32, e64, e128, d64, d32, d3, h

    def forward(self, b1, b2, b3):

        if self.input_padding is None or self.input_padding.shape != b3.shape:
            self.input_padding = torch.zeros_like(b3)

        if self.input_padding_h is None or self.input_padding_h.shape != b3.shape:
            self.input_padding_h = torch.zeros((b1.shape[0], 64, b1.shape[-2]//8, b1.shape[-1]//8))

        _ ,_ ,_ ,_ ,_ , i3, h = self.forward_step(torch.cat([b3, self.input_padding], 1), self.input_padding_h)
        h = self.upsample_fn(h, scale_factor=1.5)

        _, _, _, _, _, i2, h = self.forward_step(torch.cat([b2, self.upsample_fn(i3, scale_factor=1.5)], 1), h)
        h = self.upsample_fn(h, scale_factor=4/3)

        _, _, _, _, _, i1, _ = self.forward_step(torch.cat([b1, self.upsample_fn(i2, scale_factor=4 / 3)], 1), h)

        return i1, i2, i3




