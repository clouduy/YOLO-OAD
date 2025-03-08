
import torch
import torch.nn as nn
from DCNv4.modules.dcnv4 import DCNv4

def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Applies a convolution, batch normalization, and activation function to an input tensor in a neural network."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))

class Conv_DCNv4(nn.Module):
    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")  # GPU设备对象
    else:
        device = torch.device("cpu")  # CPU设备对象

    """Applies a convolution, batch normalization, and activation function to an input tensor in a neural network."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.dcnv4 = DCNv4(c1)
        self.conv = self.conv = Conv(c1, c2, k=1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def autopad(self,k, p=None, d=1):
        """Calculate the padding value"""
        if p is None:
            p = (k - 1) // 2 * d
        return p

    def forward(self, x):
        x = self.conv(x)
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        N, C, H, W = x.shape
        L = H * W
        x = x.permute(0, 2, 3, 1).reshape(N, L, C)
        # 先通过 DCNv4 层
        x = self.dcnv4(x)
        # 然后通过标准卷积层
        x = x.view(N, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        # 应用批量归一化和激活函数
        x = self.act(self.bn(x))
        return x

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        N, C, H, W = x.shape
        L = H * W
        x = x.permute(0, 2, 3, 1).reshape(N, L, C)
        # 先通过 DCNv4 层
        x = self.dcnv4(x)
        x = x.view(N, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        # 直接应用卷积和激活函数，绕过批量归一化
        x = self.act(self.conv(x))
        return x

    def __call__(self, x):
        return super().__call__(x)

class Bottleneck_DCNV4(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv_DCNv4(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3_DCNV4(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_DCNV4(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

if __name__ == "__main__":
    # 创建输入数据 (batch size, channels, height, width)
    input_tensor = torch.randn(1, 64, 32, 32)
    # 创建 DCNConv 实例
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    dcn_conv = Conv_DCNv4(c1=64, c2=128, k=3, s=1, p=1, g=1)
    print(dcn_conv(input_tensor).shape)