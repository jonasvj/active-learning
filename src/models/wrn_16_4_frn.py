import math
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.distributions import Categorical
from src.models import BaseModel, View, FilterResponseNorm2d


class WideResNetFRN(BaseModel):
    def __init__(
        self,
        n_train,
        depth=16,
        widen_factor=4,
        num_classes=10,
        dropout_rate=0.,
        sigma_b=1.,
        sigma_w=1.,
        sigma_default=1.,
        scale_sigma_w_by_dim=False,
        use_prior=False, 
        device='cuda'
    ):
        super().__init__(
            n_train=n_train,
            sigma_b=sigma_b,
            sigma_w=sigma_w,
            sigma_default=sigma_default,
            scale_sigma_w_by_dim=scale_sigma_w_by_dim,
            use_prior=use_prior, 
            device=device
        )

        self.likelihood = 'classification'
        self.sigma_noise = 1

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout_rate)
        # global average pooling and classifier
        self.frn1 = FilterResponseNorm2d(nChannels[3])
        #self.bn1 = nn.BatchNorm2d(nChannels[3])
        #self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
        self.avg_pool2d = nn.AvgPool2d(8)
        
        # All modules in order
        self.ordered_modules = nn.ModuleList([
            self.conv1,
            self.block1,
            self.block2,
            self.block3,
            self.frn1,
            #self.bn1,
            #nn.ReLU(),
            self.avg_pool2d,
            View([-1, self.nChannels]),
            self.fc
        ])
        self.to(self.device)

        # Initialize prior distributions
        self.init_prior_dist()


    def log_density(self, model_output, target):
        return Categorical(logits=model_output).log_prob(target)
    

    def optimizer(self, weight_decay=5e-4, lr=0.1, momentum=0.9, nesterov=True):
        optimizer = SGD(
            self.parameters(),
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )

        return optimizer


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.frn1 = FilterResponseNorm2d(in_planes)
        #self.bn1 = nn.BatchNorm2d(in_planes)
        #self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.frn2 = FilterResponseNorm2d(out_planes)
        #self.bn2 = nn.BatchNorm2d(out_planes)
        #self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropout_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.dropout_module = nn.Dropout(p=self.droprate)


    def forward(self, x):
        if not self.equalInOut:
            x = self.frn1(x)
            #x = self.relu1(self.bn1(x))
        else:
            out = self.frn1(x)
            #out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.frn2(self.conv1(out))
            #out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.frn2(self.conv1(x))
            #out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = self.dropout_module(out)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout_rate)


    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropout_rate))
        return nn.Sequential(*layers)


    def forward(self, x):
        return self.layer(x)

if __name__ == '__main__':
    from src.models import WideResNet
    model = WideResNetFRN(n_train=50000, device='cpu')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params) # 2,750,698

    N, C, W, H = 16, 3, 32, 32
    x = torch.randn(N,C,W,H)
    model(x)