import torch
import torch.nn as nn
import numpy as np

from model.registry import MODEL
import gol

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)


class Domain_transform(nn.Module):
    def __init__(self, planes):
        super(Domain_transform, self).__init__()
        self.planes = planes
        self.avg = torch.nn.AdaptiveAvgPool2d((1,1))
        self.linear=torch.nn.Linear(planes, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.detach().data
        x = self.avg(x).view(-1, self.planes)
        x = self.linear(x)
        x = self.relu(x)
        domain_offset = x.mean()
        return domain_offset


class AN(nn.Module):

    def __init__(self, planes):
        super(AN, self).__init__()
        self.IN = nn.InstanceNorm2d(planes, affine=False)
        self.BN = nn.BatchNorm2d(planes, affine=False)
        self.alpha = nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True)
        self.alpha_t = torch.Tensor([0.0])
        self.domain_transform = Domain_transform(planes)

    def forward(self, x):
        if gol.get_value('is_ft') and gol.get_value('use_transform'):
            self.alpha_t = self.alpha + 0.01 * self.domain_transform(x)
            t = torch.sigmoid(self.alpha_t).cuda()
        else:
            t = torch.sigmoid(self.alpha).cuda()
        out_in = self.IN(x)
        out_bn = self.BN(x)
        out = t * out_in + (1 - t) * out_bn
        return out


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, block_size=1, is_maxpool=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.adafbi1 = AN(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.adafbi2 = AN(planes)
        self.stride = stride
        self.downsample = downsample
        self.block_size = block_size
        self.is_maxpool = is_maxpool
        self.maxpool = nn.MaxPool2d(stride)
        self.num_batches_tracked = 0

    def forward(self, x):
        self.num_batches_tracked += 1
        residual = x
        out = self.conv1(x)
        out = self.adafbi1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.adafbi2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if self.is_maxpool:
            out = self.maxpool(out)
        return out


class ResNetAN(nn.Module):

    def __init__(self, block, resolution):
        super(ResNetAN, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.adafbi = AN(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, stride=2)
        self.layer2 = self._make_layer(block, 128, stride=2)
        self.layer3 = self._make_layer(block, 256, stride=2)
        self.layer4 = self._make_layer(block, 512, stride=2)
        self.resolution = resolution

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def _make_layer(self, block, planes, stride=1, block_size=1, is_maxpool=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layer = block(self.inplanes, planes, stride, downsample,
                      block_size, is_maxpool=is_maxpool)
        layers.append(layer)
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, AN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.adafbi(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

@MODEL.register
class Dara(nn.Module):

    def __init__(self, config):
        super().__init__()
        num_channel = 512
        resolution = config.resolution
        self.feature_extractor = ResNetAN(BasicBlock, resolution=resolution)
        self.resolution = self.feature_extractor.resolution
        
        # number of channels for the feature map, correspond to d in the paper
        self.d = num_channel
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.r = nn.Parameter(torch.zeros(2), requires_grad=not config.is_pretraining)

        # number of categories during pre-training
        self.num_classes = config.num_classes

        # category matrix, correspond to matrix M of section 3.6 in the paper
        self.cat_mat = nn.Parameter(torch.randn(self.num_classes, self.resolution, self.d), requires_grad=True)

    def get_feature_map(self, inp):
        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        feature_map = feature_map / np.sqrt(640)
        feature_map = feature_map.view(batch_size, self.d, -1).permute(0, 2, 1).contiguous()  # N,HW,C
        return feature_map

    def get_recon_dist(self, query, support, beta):
        # query: way*query_shot*resolution, d
        # support: way, shot*resolution , d
        # Woodbury: whether to use the Woodbury Identity as the implementation or not
        # correspond to gamma in the paper
        lam = support.size(1) / support.size(2)
        rho = beta.exp()
        st = support.permute(0, 2, 1)  # way, d, shot*resolution
        # correspond to Equation 8 in the paper
        sst = support.matmul(st)
        sst_plus_ri = sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)
        sst_plus_ri_np = sst_plus_ri.detach().cpu().numpy()
        sst_plus_ri_inv_np = np.linalg.inv(sst_plus_ri_np)
        sst_plus_ri_inv = torch.tensor(sst_plus_ri_inv_np).cuda()
        w = query.matmul(st.matmul(sst_plus_ri_inv))  # way, d, d
        Q_bar = w.matmul(support).mul(rho)  # way, way*query_shot*resolution, d
        # way*query_shot*resolution, way
        dist = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)
        return dist

    def forward(self, inp):
        feature_map = self.get_feature_map(inp)
        batch_size = feature_map.size(0)
        feature_map = feature_map.view(batch_size * self.resolution, self.d)
        beta = self.r[1]
        recon_dist = self.get_recon_dist(query=feature_map, support=self.cat_mat, beta=beta)
        logits = recon_dist.neg().view(batch_size, self.resolution, self.num_classes).mean(1)
        logits = logits * self.scale
        return logits
