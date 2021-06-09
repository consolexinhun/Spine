import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
from torchsummary import summary
from collections import OrderedDict
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class SeparableConv3d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv3d, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv3d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv3d_aspp(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, padding=0,
                 normalization='bn', num_groups=8):
        super(SeparableConv3d_aspp, self).__init__()

        self.depthwise = nn.Conv3d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                                   groups=inplanes, bias=bias)
        if normalization == 'bn':
            self.depthwise_bn = nn.BatchNorm3d(inplanes)
        elif normalization == 'gn':
            self.depthwise_bn = nn.GroupNorm(num_groups=num_groups, num_channels=inplanes)
        self.pointwise = nn.Conv3d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        if normalization == 'bn':
            self.pointwise_bn = nn.BatchNorm3d(planes)
        elif normalization == 'gn':
            self.pointwise_bn = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        #         x = fixed_padding(x, self.depthwise.kernel_size[0], rate=self.depthwise.dilation[0])
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        x = self.relu(x)
        return x

class Decoder_module(nn.Module):
    def __init__(self, inplanes, planes, rate=1, normalization='bn', num_groups=8):
        super(Decoder_module, self).__init__()
        self.atrous_convolution = SeparableConv3d_aspp(inplanes, planes, 3, stride=1, dilation=rate,padding=1,
                                                       normalization=normalization, num_groups=num_groups)

    def forward(self, x):
        x = self.atrous_convolution(x)
        return x

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate, normalization='bn', num_groups=8):
        super(ASPP_module, self).__init__()
        if rate == 1:
            raise RuntimeError()
        else:
            kernel_size = 3
            padding = rate
            self.atrous_convolution = SeparableConv3d_aspp(inplanes, planes, kernel_size, stride=1, dilation=rate,
                                                           padding=padding,
                                                           normalization=normalization, num_groups=num_groups)

    def forward(self, x):
        x = self.atrous_convolution(x)
        return x

class ASPP_module_rate0(nn.Module):
    def __init__(self, inplanes, planes, rate=1, normalization='bn', num_groups=8):
        super(ASPP_module_rate0, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
            self.atrous_convolution = nn.Conv3d(inplanes, planes, kernel_size=kernel_size,
                                                stride=1, padding=padding, dilation=rate, bias=False)
            if normalization == 'bn':
                self.bn = nn.BatchNorm3d(planes, eps=1e-5, affine=True)
            elif normalization == 'gn':
                self.bn = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.relu = nn.ReLU()
        else:
            raise RuntimeError()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

class SeparableConv3d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, padding=0,
                 normalization='bn', num_groups=8):
        super(SeparableConv3d_same, self).__init__()
        if planes % num_groups != 0:
            num_groups = int(num_groups / 2)
        if inplanes % num_groups != 0:
            num_groups = int(num_groups / 2)
        self.depthwise = nn.Conv3d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                                   groups=inplanes, bias=bias)
        if normalization == 'bn':
            self.depthwise_bn = nn.BatchNorm3d(inplanes)
        elif normalization == 'gn':
            self.depthwise_bn = nn.GroupNorm(num_groups=num_groups, num_channels=inplanes)
        self.pointwise = nn.Conv3d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        if normalization == 'bn':
            self.pointwise_bn = nn.BatchNorm3d(planes)
        elif normalization == 'gn':
            self.pointwise_bn = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

    def forward(self, x):
        x = fixed_padding(x, self.depthwise.kernel_size[0], rate=self.depthwise.dilation[0])
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        return x

class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True,
                 is_last=False, normalization='bn', num_groups=8):
        super(Block, self).__init__()
        if planes % num_groups != 0:
            num_groups = int(num_groups / 2)
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv3d(inplanes, planes, 1, stride=stride, bias=False)
            if is_last:
                self.skip = nn.Conv3d(inplanes, planes, 1, stride=1, bias=False)
            if normalization == 'bn':
                self.skipbn = nn.BatchNorm3d(planes)
            elif normalization == 'gn':
                self.skipbn = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(inplanes, planes, 3, stride=1, dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))
#             rep.append(nn.BatchNorm3d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(filters, filters, 3, stride=1, dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))
#             rep.append(nn.BatchNorm3d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(inplanes, planes, 3, stride=1, dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))
#             rep.append(nn.BatchNorm3d(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(planes, planes, 3, stride=stride,dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))

        if is_last:
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(planes, planes, 3, stride=1,dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))


        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        # print(x.size(),skip.size())
        x += skip

        return x

class Block2(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True,
                 is_last=False, normalization='bn', num_groups=8):
        super(Block2, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv3d(inplanes, planes, 1, stride=stride, bias=False)
            if normalization == 'bn':
                self.skipbn = nn.BatchNorm3d(planes)
            elif normalization == 'gn':
                self.skipbn = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(inplanes, planes, 3, stride=1, dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))
#             rep.append(nn.BatchNorm3d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(filters, filters, 3, stride=1, dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))
#             rep.append(nn.BatchNorm3d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(inplanes, planes, 3, stride=1, dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))
#             rep.append(nn.BatchNorm3d(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            self.block2_lastconv = nn.Sequential(*[self.relu,SeparableConv3d_same(planes, planes, 3, stride=stride,
                                                                                  dilation=dilation,
                                                                                  normalization=normalization,
                                                                                  num_groups=num_groups)])

        if is_last:
            rep.append(SeparableConv3d_same(planes, planes, 3, stride=1, normalization=normalization, num_groups=num_groups))


        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        low_middle = x.clone()
        x1 = x
        x1 = self.block2_lastconv(x1)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x1 += skip

        return x1,low_middle

class Xception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, inplanes=3, os=16, pretrained=False, normalization='bn', num_groups=8):
        super(Xception, self).__init__()

        if os == 16:
            entry_block3_stride = (1, 2, 2)
            middle_block_rate = 1
            exit_block_rates = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        self.conv1 = nn.Conv3d(inplanes, 16, 3, stride=(1,2,2), padding=1, bias=False)
        if normalization == 'bn':
            self.bn1 = nn.BatchNorm3d(16)
        elif normalization == 'gn':
            self.bn1 = nn.GroupNorm(num_groups=num_groups, num_channels=16)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(16, 32, 3, stride=1, padding=1, bias=False)
        # self.conv2 = nn.Conv3d(16, 32, 3, stride=(1,2,2), padding=1, bias=False)
        if normalization == 'bn':
            self.bn2 = nn.BatchNorm3d(32)
        elif normalization == 'gn':
            self.bn2 = nn.GroupNorm(num_groups=num_groups, num_channels=32)

        self.block1 = Block(32, 64, reps=2, stride=(1,2,2), start_with_relu=False, grow_first=True,
                            normalization=normalization, num_groups=num_groups)


        self.block2 = Block2(64, 128, reps=2, stride=(1,2,2), start_with_relu=True, grow_first=True,
                             normalization=normalization, num_groups=num_groups)
        self.block3 = Block(128, 364, reps=2, stride=entry_block3_stride, start_with_relu=True, grow_first=True,
                            normalization=normalization, num_groups=num_groups)

        # Middle flow
        # self.block4  = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block5  = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block6  = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block7  = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block8  = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block9  = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block10 = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block11 = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block12 = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block13 = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block14 = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block15 = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block16 = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block17 = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block18 = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block19 = Block(364, 364, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(364, 512, reps=2, stride=1, dilation=exit_block_rates[0],
                             start_with_relu=True, grow_first=False, is_last=True,
                             normalization=normalization, num_groups=num_groups)


        self.conv3 = SeparableConv3d_aspp(512, 768, 3, stride=1, dilation=exit_block_rates[1],padding=exit_block_rates[1],
                                          normalization=normalization, num_groups=num_groups)
        # self.bn3 = nn.BatchNorm3d(1536)

        self.conv4 = SeparableConv3d_aspp(768, 768, 3, stride=1, dilation=exit_block_rates[1],padding=exit_block_rates[1],
                                          normalization=normalization, num_groups=num_groups)
        # self.bn4 = nn.BatchNorm3d(1536)

        self.conv5 = SeparableConv3d_aspp(768, 1024, 3, stride=1, dilation=exit_block_rates[1],padding=exit_block_rates[1],
                                          normalization=normalization, num_groups=num_groups)
        # self.bn5 = nn.BatchNorm3d(2048)

        # Init weights
        # self.__init_weight()

        # Load pretrained model
        if pretrained:
            self.__load_xception_pretrained()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        low_level_feat_2 = self.relu(x)
        x = self.block1(low_level_feat_2)

        # x = self.conv1(x)
        # x = self.bn1(x)
        # low_level_feat_2 = self.relu(x)
        # x = self.conv2(low_level_feat_2)
        # x = self.bn2(x)
        # x = self.relu(x)
        # x = self.block1(x)

        x,low_level_feat_4 = self.block2(x)
        # print('block2',x.size())
        x = self.block3(x)
        # print('xception block3 ',x.size())

        # Middle flow
        # x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        # x = self.block7(x)
        # x = self.block8(x)
        # x = self.block9(x)
        # x = self.block10(x)
        # x = self.block11(x)
        # x = self.block12(x)
        # x = self.block13(x)
        # x = self.block14(x)
        # x = self.block15(x)
        # x = self.block16(x)
        # x = self.block17(x)
        # x = self.block18(x)
        # x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        # x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat_2, low_level_feat_4

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __load_xception_pretrained(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('block11'):
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('conv3'):
                    model_dict[k] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

class DeepLabv3_plus_skipconnection_3d_subnet1(nn.Module):
    def __init__(self, n_classes=20, os=16, _print=True, final_sigmoid=False,
                 normalization='bn', num_groups=8):
        super(DeepLabv3_plus_skipconnection_3d_subnet1, self).__init__()
        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module_rate0(1024, 128, rate=rates[0], normalization=normalization, num_groups=num_groups)
        self.aspp2 = ASPP_module(1024, 128, rate=rates[1], normalization=normalization, num_groups=num_groups)
        self.aspp3 = ASPP_module(1024, 128, rate=rates[2], normalization=normalization, num_groups=num_groups)
        self.aspp4 = ASPP_module(1024, 128, rate=rates[3], normalization=normalization, num_groups=num_groups)

        self.relu = nn.ReLU()
        if normalization == 'bn':
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                                 nn.Conv3d(1024, 128, 1, stride=1, bias=False),
                                                 nn.BatchNorm3d(128),
                                                 nn.ReLU()
                                                 )
        elif normalization == 'gn':
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                                 nn.Conv3d(1024, 128, 1, stride=1, bias=False),
                                                 nn.GroupNorm(num_groups=num_groups, num_channels=128),
                                                 nn.ReLU()
                                                 )

        self.concat_projection_conv1 = nn.Conv3d(640, 128, 1, bias=False)
        if normalization == 'bn':
            self.concat_projection_bn1 = nn.BatchNorm3d(128)
        elif normalization == 'gn':
            self.concat_projection_bn1 = nn.GroupNorm(num_groups=num_groups, num_channels=128)

        # adopt [1x1, 48] for channel reduction.
        self.feature_projection_conv1 = nn.Conv3d(128, 24, 1, bias=False)
        if normalization == 'bn':
            self.feature_projection_bn1 = nn.BatchNorm3d(24)
        elif normalization == 'gn':
            self.feature_projection_bn1 = nn.GroupNorm(num_groups=num_groups, num_channels=24)

        self.feature_projection_conv2 = nn.Conv3d(32, 64, 1, bias=False)
        if normalization == 'bn':
            self.feature_projection_bn2 = nn.BatchNorm3d(64)
        elif normalization == 'gn':
            self.feature_projection_bn2 = nn.GroupNorm(num_groups=num_groups, num_channels=64)

        self.decoder1 = nn.Sequential(Decoder_module(152, 128, normalization=normalization, num_groups=num_groups),
                                      Decoder_module(128, 128, normalization=normalization, num_groups=num_groups))

    def forward(self, x, low_level_features_2, low_level_features_4):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='trilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)

        low_level_features_2 = self.feature_projection_conv2(low_level_features_2)
        low_level_features_2 = self.feature_projection_bn2(low_level_features_2)
        low_level_features_2 = self.relu(low_level_features_2)

        low_level_features_4 = self.feature_projection_conv1(low_level_features_4)
        low_level_features_4 = self.feature_projection_bn1(low_level_features_4)
        low_level_features_4 = self.relu(low_level_features_4)

        x = F.interpolate(x, size=low_level_features_4.size()[2:], mode='trilinear', align_corners=True)

        x = torch.cat((x, low_level_features_4), dim=1)

        x = self.decoder1(x)
        x = F.interpolate(x, size=low_level_features_2.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat((x, low_level_features_2), dim=1)

        return x
        # return x, low_level_features_2


# class DeepLabv3_plus_skipconnection_3d_subnet2(nn.Module):
#     def __init__(self, n_classes, normalization, num_groups, final_sigmoid):
#         super(DeepLabv3_plus_skipconnection_3d_subnet2, self).__init__()
#         self.decoder1 = nn.Sequential(Decoder_module(152, 128, normalization=normalization, num_groups=num_groups),
#                                       Decoder_module(128, 128, normalization=normalization, num_groups=num_groups))
#
#     def forward(self, x, low_level_features_2):
#         x = self.decoder1(x)
#         x = F.interpolate(x, size=low_level_features_2.size()[2:], mode='trilinear', align_corners=True)
#         x = torch.cat((x, low_level_features_2), dim=1)
#
#         return x

class DeepLabv3_plus_skipconnection_3d_subnet2(nn.Module):
    def __init__(self, n_classes, normalization, num_groups, final_sigmoid):
        super(DeepLabv3_plus_skipconnection_3d_subnet2, self).__init__()

        # self.decoder2 = nn.Sequential(Decoder_module(192, 256, normalization=normalization, num_groups=num_groups),
        #                               Decoder_module(256, 256, normalization=normalization, num_groups=num_groups))

        self.decoder2 = nn.Sequential(Decoder_module(192, 128, normalization=normalization, num_groups=num_groups),
                                      Decoder_module(128, 128, normalization=normalization, num_groups=num_groups))

        # self.semantic = nn.Conv3d(256, n_classes, kernel_size=1, stride=1)
        self.semantic = nn.Conv3d(128, n_classes, kernel_size=1, stride=1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x, input, is_training):
        x = self.decoder2(x)
        x = self.semantic(x)
        final_conv = F.interpolate(x, size=input.size()[2:], mode='trilinear', align_corners=True)

        if not is_training:
            x = self.final_activation(final_conv)
            return x, final_conv
        else:
            return final_conv


class DeepLabv3_plus_skipconnection_3d(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=20, os=16, pretrained=False, _print=True, final_sigmoid=False,
                 normalization='bn', num_groups=8, devices=None):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus_skipconnection_3d, self).__init__()
        self.devices = devices

        # Atrous Conv
        self.xception_features = Xception(nInputChannels, os, pretrained, normalization=normalization, num_groups=num_groups).to(self.devices[0])

        if len(self.devices) == 1:
            self.sub_net1 = DeepLabv3_plus_skipconnection_3d_subnet1(n_classes=n_classes, os=os, _print=_print,
                                                                     final_sigmoid=final_sigmoid,
                                                                     normalization=normalization,
                                                                     num_groups=num_groups).to(self.devices[0])

            self.sub_net2 = DeepLabv3_plus_skipconnection_3d_subnet2(n_classes=n_classes, normalization=normalization,
                                                                     num_groups=num_groups,
                                                                     final_sigmoid=final_sigmoid).to(self.devices[0])
        else:
            self.sub_net1 = DeepLabv3_plus_skipconnection_3d_subnet1(n_classes=n_classes, os=os, _print=_print, final_sigmoid=final_sigmoid,
                     normalization=normalization, num_groups=num_groups).to(self.devices[1])

            self.sub_net2 = DeepLabv3_plus_skipconnection_3d_subnet2(n_classes=n_classes, normalization=normalization, num_groups=num_groups,
                                                                     final_sigmoid=final_sigmoid).to(self.devices[2])


    def forward(self, input):
        x, low_level_features_2, low_level_features_4 = self.xception_features(input)

        if len(self.devices) > 1:
            x = x.to(self.devices[1])
            low_level_features_2 = low_level_features_2.to(self.devices[1])
            low_level_features_4 = low_level_features_4.to(self.devices[1])
        x = self.sub_net1(x, low_level_features_2, low_level_features_4)

        if len(self.devices) > 1:
            x = x.to(self.devices[2])
        results = self.sub_net2(x, input, self.training)


        if isinstance(results, (list, tuple)):
            activation, final_conv = results
            if len(self.devices) > 1:
                activation = activation.to(self.devices[0])
                final_conv = final_conv.to(self.devices[0])
            return activation, final_conv, None
        else:
            if len(self.devices) > 1:
                final_conv = results.to(self.devices[0])
            else:
                final_conv = results
            return final_conv, None


    def freeze_bn(self):
        for m in self.xception_features.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

    def freeze_totally_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

    def freeze_aspp_bn(self):
        for m in self.aspp1.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()
        for m in self.aspp2.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()
        for m in self.aspp3.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()
        for m in self.aspp4.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

    def learnable_parameters(self):
        layer_features_BN = []
        layer_features = []
        layer_aspp = []
        layer_projection  =[]
        layer_decoder = []
        layer_other = []
        model_para = list(self.named_parameters())
        for name,para in model_para:
            if 'xception' in name:
                if 'bn' in name or 'downsample.1.weight' in name or 'downsample.1.bias' in name:
                    layer_features_BN.append(para)
                else:
                    layer_features.append(para)
                    # print (name)
            elif 'aspp' in name:
                layer_aspp.append(para)
            elif 'projection' in name:
                layer_projection.append(para)
            elif 'decode' in name:
                layer_decoder.append(para)
            elif 'global' not in name:
                layer_other.append(para)
        return layer_features_BN,layer_features,layer_aspp,layer_projection,layer_decoder,layer_other

    def get_backbone_para(self):
        layer_features = []
        other_features = []
        model_para = list(self.named_parameters())
        for name, para in model_para:
            if 'xception' in name:
                layer_features.append(para)
            else:
                other_features.append(para)

        return layer_features, other_features

    def train_fixbn(self, mode=True, freeze_bn=True, freeze_bn_affine=False):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Returns:
            Module: self
        """
        super(DeepLabv3_plus_skipconnection_3d, self).train(mode)
        if freeze_bn:
            print("Freezing Mean/Var of BatchNorm3D.")
            if freeze_bn_affine:
                print("Freezing Weight/Bias of BatchNorm3D.")
        if freeze_bn:
            for m in self.xception_features.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_state_dict_new(self, state_dict):
        own_state = self.state_dict()
        #for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.','')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print ('unexpected key "{}" in state_dict'
                       .format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                          ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                        name, own_state[name].size(), param.size()))
                continue # i add inshop_cos 2018/02/01
                # raise
                    # print 'copying %s' %name
                # if isinstance(param, own_state):
                # backwards compatibility for serialized parameters
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.xception_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    device = torch.device('cuda:3')
    image = torch.randn(2, 1, 18, 256, 512) * 255
    image = image.to(device)
    model = DeepLabv3_plus_skipconnection_3d(nInputChannels=1, n_classes=20, os=16, pretrained=False, _print=True)
    model = model.to(device)
    print(model)
    # summary(model, input_size=(3, 20, 256, 512))
    model.eval()

    with torch.no_grad():
        output = model.forward(image)

    # print(output)






