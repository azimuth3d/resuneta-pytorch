import torch
import torch.nn as nn

class CNR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm='bnorm', relu=0.0):
        super(CNR2d, self).__init__()
        bias = False if norm == 'bnorm' else True
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)]
        if norm == 'bnorm':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == 'lnorm':
            layers.append(nn.LayerNorm([out_channels, 1, 1]))
        if relu > 0:
            layers.append(nn.LeakyReLU(relu, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Pooling2d(nn.Module):
    def __init__(self, pool=2, type='avg'):
        super(Pooling2d, self).__init__()
        if type == 'avg':
            self.pool = nn.AvgPool2d(pool)
        elif type == 'max':
            self.pool = nn.MaxPool2d(pool)
        else:
            raise ValueError(f"Unknown pooling type: {type}")
            
    def forward(self, x):
        return self.pool(x)

class UnPooling2d(nn.Module):
    def __init__(self, nch=None, pool=2, type='conv'):
        super(UnPooling2d, self).__init__()
        self.type = type
        if type == 'conv':
            if nch is None:
                raise ValueError("nch must be provided for conv unpooling")
            self.unpool = nn.ConvTranspose2d(nch, nch, kernel_size=pool, stride=pool)
        elif type in ['bilinear', 'nearest', 'trilinear']:
            self.unpool = nn.Upsample(scale_factor=pool, mode=type, align_corners=True)
        else:
            raise ValueError(f"Unknown unpooling type: {type}")
            
    def forward(self, x):
        return self.unpool(x)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        
    def forward(self, x):
        return self.conv(x)
