import torch.nn as nn
from torch.nn.init import xavier_normal_, kaiming_normal_
from functools import partial


def get_weight_init_fn(activation_fn):

    fn = activation_fn
    if hasattr(activation_fn, 'func'):
        fn = activation_fn.func

    if fn == nn.LeakyReLU:
        negative_slope = 0
        if hasattr(activation_fn, 'keywords'):
            if activation_fn.keywords.get('negative_slope') is not None:
                negative_slope = activation_fn.keywords['negative_slope']
        if hasattr(activation_fn, 'args'):
            if len(activation_fn.args) > 0:
                negative_slope = activation_fn.args[0]
        return partial(kaiming_normal_, a=negative_slope)
    elif fn == nn.ReLU or fn == nn.PReLU:
        return partial(kaiming_normal_, a=0)
    else:
        return xavier_normal_
    return


def linear(in_channels, out_channels, activation_fn=None, use_batchnorm=False, pre_activation=False, bias=True,
           weight_init_fn=None):
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    # 在线性层之前添加激活函数
    if pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))
        if activation_fn is not None:
            layers.append(activation_fn())

    # 添加线性层
    linear = nn.Linear(in_channels, out_channels)
    # 线性层初始化
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn(activation_fn)
    weight_init_fn(linear.weight)
    layers.append(linear)

    # 在线性层之后添加激活函数
    if not pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, activation_fn=None, use_batchnorm=False,
         pre_activation=False, bias=True, weight_init_fn=None):
    if not pre_activation and use_batchnorm:
        assert not bias
    layers = []
    # 卷积层之前
    if pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    # 卷积层
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn(activation_fn)
        weight_init_fn(conv.weight)
    layers.append(conv)
    # 卷积层之后
    if not pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    return nn.Sequential(*layers)


def deconv(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, activation_fn=None,
           use_batchnorm=False, pre_activation=False, bias=True, weight_init_fn=None):
    if not pre_activation and use_batchnorm:
        assert not bias
    layers = []
    # 转置卷积之前
    if pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    # 转置卷积
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)
    # 初始化转置卷积
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn(activation_fn)
    weight_init_fn(deconv.weight)
    layers.append(deconv)
    # 转置卷积后
    if not pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_batchnorm=False,
                 activation_fn=partial(nn.ReLU, inplace=True), last_activation_fn=partial(nn.ReLU, inplace=True),
                 pre_activation=False, scaling_factor=1.0):
        super(BasicBlock, self).__init__()
        # 第一个卷积层(缩小stride倍,通道:in_channels--->out_channels),使用激活函数(activation_fn)
        self.conv1 = conv(in_channels, out_channels, kernel_size, stride, kernel_size // 2, activation_fn,
                          use_batchnorm)
        # 第二个卷积层(channel,h,w都未改变),不使用激活函数(activation_fn)
        self.conv2 = conv(out_channels, out_channels, kernel_size, 1, kernel_size // 2, None, use_batchnorm,
                          weight_init_fn=get_weight_init_fn(last_activation_fn))
        # 如果是下采样,残差块需要使用1x1卷积层来改变通道数
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = conv(in_channels, out_channels, 1, stride, 0, None, use_batchnorm)
        # 在最后添加激活函数
        if last_activation_fn is not None:
            self.last_activation = last_activation_fn()
        else:
            self.last_activation = None
        # 残差块的影响因子
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # 计算残差块,如果是下采样需要进入1x1卷积层改变(c,h,w)
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)
        # 第一个第二个卷积层输出
        out = self.conv1(x)
        out = self.conv2(out)
        # 残差块合并
        out += residual * self.scaling_factor
        # 激活函数
        if self.last_activation is not None:
            out = self.last_activation(out)
        return out
