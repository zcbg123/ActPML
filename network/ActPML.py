# 这里是SAC的编辑
# 开发时间：2024/10/29 18:42
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch.nn import Module, ModuleList, Sigmoid
from torch import nn, einsum
from .GLFA import *
from .GLSE import LGCE
from torchvision import models
from .newact1 import *


class DeconvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DeconvolutionLayer, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.deconv(x)



class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DWConv, self).__init__()

        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                                 stride, padding, groups=in_channels),
                                       nn.BatchNorm2d(in_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                       nn.BatchNorm2d(out_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicConv2d_P(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_P, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicConv2d_2P(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_2P, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=4)
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicConv2d_4P(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_4P, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=8)
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




class BasicConv2d5(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=2, dilation=1):
        super(BasicConv2d5, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class PrestageHierarchicalFusion(nn.Module):
    def __init__(self, in_d=None, out_d=None):
        super(PrestageHierarchicalFusion, self).__init__()
        if in_d is None:
            in_d = [128, 256, 512, 512]
        self.in_d = in_d
        if out_d is None:
            out_d = [128, 256, 512, 512]
        self.out_d = out_d
        self.mid_d = [128, 256, 512, 512]
        # fusion1
        self.fusion1_f1 = BasicConv2d(in_d[0],out_d[0],3,1,1)
        self.fusion1_f2 = BasicConv2d(in_d[1],out_d[0],3,1,1)
        self.fusion1_f3 = BasicConv2d(in_d[2],out_d[0],3,1,1)
        self.fusion1_f4 = BasicConv2d(in_d[3],out_d[0],3,1,1)
        self.aggregation_s1 = FeatureFusionModule(self.mid_d[0] * 4, self.in_d[0], self.out_d[0])
        # fusion2
        self.fusion2_f1 = BasicConv2d_P(in_d[0],out_d[1],3,1,1)
        self.fusion2_f2 = BasicConv2d(in_d[1],out_d[1],3,1,1)
        self.fusion2_f3 = BasicConv2d(in_d[2],out_d[1],3,1,1)
        self.fusion2_f4 = BasicConv2d(in_d[3],out_d[1],3,1,1)
        self.aggregation_s2 = FeatureFusionModule(self.mid_d[1] * 4, self.in_d[1], self.out_d[1])
        # fusion3
        self.fusion3_f1 = BasicConv2d_2P(in_d[0],out_d[2],3,1,1)
        self.fusion3_f2 = BasicConv2d_P(in_d[1],out_d[2],3,1,1)
        self.fusion3_f3 = BasicConv2d(in_d[2],out_d[2],3,1,1)
        self.fusion3_f4 = BasicConv2d(in_d[3],out_d[2],3,1,1)
        self.aggregation_s3 = FeatureFusionModule(self.mid_d[2] * 4, self.in_d[2], self.out_d[2])
        # fusion4
        self.fusion4_f1 = BasicConv2d_4P(in_d[0],out_d[3],3,1,1)
        self.fusion4_f2 = BasicConv2d_2P(in_d[1],out_d[3],3,1,1)
        self.fusion4_f3 = BasicConv2d_P(in_d[2],out_d[3],3,1,1)
        self.fusion4_f4 = BasicConv2d(in_d[3],out_d[3],3,1,1)
        self.aggregation_s4 = FeatureFusionModule(self.mid_d[3] * 4, self.in_d[3], self.out_d[3])

    def forward(self, f1, f2, f3, f4):
        # fusion1
        f1_f1 = self.fusion1_f1(f1)
        f1_f2 = self.fusion1_f2(f2)
        f1_f3 = self.fusion1_f3(f3)
        f1_f4 = self.fusion1_f4(f4)
        f1_f2 = F.interpolate(f1_f2, scale_factor=(2, 2), mode='bilinear')
        f1_f3 = F.interpolate(f1_f3, scale_factor=(4, 4), mode='bilinear')
        f1_f4 = F.interpolate(f1_f4, scale_factor=(8, 8), mode='bilinear')
        F1 = self.aggregation_s1(torch.cat([f1_f1, f1_f2, f1_f3, f1_f4], dim=1), f1)

        # fusion2
        f2_f1 = self.fusion2_f1(f1)
        f2_f2 = self.fusion2_f2(f2)
        f2_f3 = self.fusion2_f3(f3)
        f2_f4 = self.fusion2_f4(f4)
        f2_f3 = F.interpolate(f2_f3, scale_factor=(2, 2), mode='bilinear')
        f2_f4 = F.interpolate(f2_f4, scale_factor=(4, 4), mode='bilinear')
        F2 = self.aggregation_s2(torch.cat([f2_f1, f2_f2, f2_f3, f2_f4], dim=1), f2)

        # fusion3
        f3_f1 = self.fusion3_f1(f1)
        f3_f2 = self.fusion3_f2(f2)
        f3_f3 = self.fusion3_f3(f3)
        f3_f4 = self.fusion3_f4(f4)
        f3_f4 = F.interpolate(f3_f4, scale_factor=(2, 2), mode='bilinear')
        F3 = self.aggregation_s3(torch.cat([f3_f1, f3_f2, f3_f3, f3_f4], dim=1), f3)

        # fusion4
        f4_f1 = self.fusion4_f1(f1)
        f4_f2 = self.fusion4_f2(f2)
        f4_f3 = self.fusion4_f3(f3)
        f4_f4 = self.fusion4_f4(f4)
        F4 = self.aggregation_s4(torch.cat([f4_f1, f4_f2, f4_f3, f4_f4], dim=1), f4)

        return F1, F2, F3, F4



class FeatureFusionModule(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            DWConv(self.fuse_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True),
            DWConv(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))


        return c_out


class ConV5(nn.Module):
    def __init__(self, in_channels):
        super(ConV5, self).__init__()
        self.conv1 = BasicConv2d(in_channels , in_channels // 2,3,1,1)
        self.conv2 = BasicConv2d(in_channels // 2 , in_channels // 4,3,1,1)
        self.conv3 = BasicConv2d(in_channels // 4 , in_channels // 8,3,1,1)
        self.conv4 = BasicConv2d(in_channels // 8 , in_channels // 8,3,1,1)
        self.conv5 = DWConv(in_channels , in_channels,3,1,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out1 = self.conv2(out)
        out2 = self.conv3(out1)
        out3 = self.conv4(out2)
        # print(out.shape,out1.shape,out2.shape,out3.shape)
        out4 = torch.cat((out,out1,out2,out3),dim=1)
        # print(out4.shape,x.shape)
        out4 = out4 + self.conv5(x)
        return self.sigmoid(out4)




class ActPML(nn.Module):
    def __init__(self,):
        super(FMENet, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        self.PHF = PrestageHierarchicalFusion()

        self.pool = BasicConv2d_P(512,512,3,1,1)
        self.conv_reduce_1 = DWConv(128*2,128,3,1,1)
        self.conv_reduce_2 = DWConv(256*2,256,3,1,1)
        self.conv_reduce_3 = DWConv(512*2,512,3,1,1)
        self.conv_reduce_4 = DWConv(512*2,512,3,1,1)


        self.act1 = ActNet(
            input_dim=128,  # 与 MLP 输入维度一致
            embed_dim=128,  # 与 MLP 最大隐藏层维度（192）对齐
            num_layers=6,  # 对应 MLP 的 3 个隐藏层
            out_dim=128,  # 输出维度与 MLP 一致
            num_freqs=128,  # 频率数 ≈ 2*embed_dim/num_layers
            output_activation=nn.Sigmoid(),  # 若做回归任务可不加激活
            op_order='ALALAL',  # 交替使用 ActLayer 和 Skip
            # 初始化参数需与 MLP 对齐
            freqs_init=lambda t: nn.init.normal_(t, mean=0, std=0.1),
            beta_init=lambda t: nn.init.kaiming_uniform_(t),
            # 其他参数保持默认
        )
        self.act2 = ActNet(
            input_dim=64,
            embed_dim=64,
            num_layers=6,
            out_dim=64,
            num_freqs=64,
            output_activation=nn.Sigmoid(),
            op_order='ALALAL',
            # 初始化参数需与 MLP 对齐
            freqs_init=lambda t: nn.init.normal_(t, mean=0, std=0.1),
            beta_init=lambda t: nn.init.kaiming_uniform_(t),
            # 其他参数保持默认
        )
        self.act3 = ActNet(
            input_dim=32,  # 与 MLP 输入维度一致
            embed_dim=32,  # 与 MLP 最大隐藏层维度（192）对齐
            num_layers=6,  # 对应 MLP 的 3 个隐藏层
            out_dim=32,  # 输出维度与 MLP 一致
            num_freqs=32,  # 频率数 ≈ 2*embed_dim/num_layers
            output_activation=nn.Sigmoid(),  # 若做回归任务可不加激活
            op_order='ALALAL',  # 交替使用 ActLayer 和 Skip
            # 初始化参数需与 MLP 对齐
            freqs_init=lambda t: nn.init.normal_(t, mean=0, std=0.1),
            beta_init=lambda t: nn.init.kaiming_uniform_(t),
            # 其他参数保持默认
        )
        self.act4 = ActNet(
            input_dim=16,  # 与 MLP 输入维度一致
            embed_dim=16,  # 与 MLP 最大隐藏层维度（192）对齐
            num_layers=6,  # 对应 MLP 的 3 个隐藏层
            out_dim=16,  # 输出维度与 MLP 一致
            num_freqs=16,  # 频率数 ≈ 2*embed_dim/num_layers
            output_activation=nn.Sigmoid(),  # 若做回归任务可不加激活
            op_order='ALALAL',  # 交替使用 ActLayer 和 Skip
            # 初始化参数需与 MLP 对齐
            freqs_init=lambda t: nn.init.normal_(t, mean=0, std=0.1),
            beta_init=lambda t: nn.init.kaiming_uniform_(t),
            # 其他参数保持默认
        )

        self.lgaa4 = LGAA4().cuda()
        self.lgaa3 = LGAA3().cuda()
        self.lgaa2 = LGAA2().cuda()
        self.lgaa1 = LGAA1().cuda()

        self.lgce4 = LGCE(512).cuda()
        self.lgce4_1 = LGCE(1024).cuda()
        self.lgce3 = LGCE(512).cuda()
        self.lgce2 = LGCE(256).cuda()
        self.lgce1 = LGCE(128).cuda()

        self.channelfusion4 = nn.Sequential(nn.Conv2d(512, 512, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.channelfusion3 = nn.Sequential(nn.Conv2d(512, 512, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.channelfusion2 = nn.Sequential(nn.Conv2d(256, 256, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.channelfusion1 = nn.Sequential(nn.Conv2d(128, 128, 1), nn.BatchNorm2d(128), nn.ReLU())

        self.decoder = nn.Sequential(BasicConv2d(1024,64,3,1,1),nn.Conv2d(64,1,3,1,1))
        self.decoder_new = nn.Sequential(BasicConv2d(2816,64,3,1,1),nn.Conv2d(64,1,3,1,1))
        self.decoder_final = nn.Sequential(BasicConv2d(128,64,3,1,1),nn.Conv2d(64,1,1))

        self.conv1 = ConV5(128)
        self.conv2 = ConV5(256)
        self.conv3 = ConV5(512)
        self.conv4 = ConV5(512)


        #相比v2 额外的模块
        self.upsample2x=nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_module4 = BasicConv2d(1024,512,3,1,1)
        self.decoder_module3 = BasicConv2d(768,256,3,1,1)
        self.decoder_module2 = BasicConv2d(384,128,3,1,1)
        self.decoder_module1 = BasicConv2d(128,128,3,1,1)

    # def forward(self, A,B=None):
    #     if B == None:
    #         B = A
    def forward(self,A,B):
        # A = A.to(device)
        # B = B.to(device)
        size = A.size()[2:]
        layer1_pre = self.inc(A)
        layer1_A = self.down1(layer1_pre)
        layer2_A = self.down2(layer1_A)
        layer3_A = self.down3(layer2_A)
        layer4_A = self.down4(layer3_A)
        layer1_pre = self.inc(B)
        layer1_B = self.down1(layer1_pre)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)
        layer1_A, layer2_A, layer3_A, layer4_A = self.PHF(layer1_A, layer2_A, layer3_A, layer4_A)
        layer1_B, layer2_B, layer3_B, layer4_B = self.PHF(layer1_B, layer2_B, layer3_B, layer4_B)

        layer1_A_act = self.act1(layer1_A)
        layer2_A_act = self.act2(layer2_A)
        layer3_A_act = self.act3(layer3_A)
        layer4_A_act = self.act4(layer4_A)
        layer1_B_act = self.act1(layer1_B)
        layer2_B_act = self.act2(layer2_B)
        layer3_B_act = self.act3(layer3_B)
        layer4_B_act = self.act4(layer4_B)


        layer1_1 = self.lgaa1(self.channelfusion1(layer1_A_act),layer1_B_act)
        layer1_2 = self.lgaa1(self.channelfusion1(layer1_B_act),layer1_A_act)
        layer_t1 = torch.cat((layer1_1,layer1_2),dim=1)


        layer2_1 = self.lgaa2(self.channelfusion2(layer2_A_act),layer2_B_act)
        layer2_2 = self.lgaa2(self.channelfusion2(layer2_B_act),layer2_A_act)
        layer_t2 = torch.cat((layer2_1,layer2_2),dim=1)


        layer3_1 = self.lgaa3(self.channelfusion3(layer3_A_act),layer3_B_act)
        layer3_2 = self.lgaa3(self.channelfusion3(layer3_B_act),layer3_A_act)
        layer_t3 = torch.cat((layer3_1,layer3_2),dim=1)


        layer4_1 = self.lgaa4(self.channelfusion4(layer4_A_act),layer4_B_act)
        layer4_2 = self.lgaa4(self.channelfusion4(layer4_B_act),layer4_A_act)
        layer_t4 = torch.cat((layer4_1,layer4_2),dim=1)


        layer_t1 = self.conv_reduce_1(layer_t1)
        layer_t2 = self.conv_reduce_2(layer_t2)
        layer_t3 = self.conv_reduce_3(layer_t3)
        layer_t4 = self.conv_reduce_4(layer_t4)

        # print(layer_t4.shape,(self.pool(layer_t3)).shape,layer_t3.shape)
        layer4_1t = torch.cat((layer_t4,self.pool(layer_t3)),dim=1)


        layer4_1t = F.interpolate(layer4_1t, layer_t1.size()[2:], mode='bilinear', align_corners=True)
        feature_fuse=layer4_1t

        feature_fuse = self.lgce4_1(feature_fuse)
        guide_map = self.decoder(feature_fuse)

        layer_t1 = self.conv1(layer_t1)
        layer_t2 = self.conv2(layer_t2)
        layer_t3 = self.conv3(layer_t3)
        layer_t4 = self.conv4(layer_t4)


        layer_t4 = self.lgce4(layer_t4)
        feature4 = torch.cat([self.upsample2x(layer_t4), layer_t3], dim=1)
        feature4 = self.decoder_module4(feature4)

        layer_t3 = self.lgce3(layer_t3) + feature4
        feature3 = torch.cat([self.upsample2x(layer_t3), layer_t2], dim=1)
        feature3 = self.decoder_module3(feature3)

        layer_t2 = self.lgce2(layer_t2) + feature3
        feature2 = torch.cat([self.upsample2x(layer_t2), layer_t1], dim=1)
        feature2 = self.decoder_module2(feature2)

        layer_t1 = self.lgce1(layer_t1) + feature2
        layer_t1 = self.decoder_module1(layer_t1)


        guide_map = F.interpolate(guide_map, size, mode='bilinear', align_corners=True)
        final_map = self.decoder_final(layer_t1)

        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        return guide_map, final_map



from thop import profile
from thop import clever_format
testmodel=ActPML().cuda()
#print(summary(testmodel, input_size=[(1,3, 256,256), (1,3, 256,256)]))
dummy_input = torch.randn((1,3, 256,256)).cuda()
flops, params = profile(testmodel, (dummy_input,dummy_input))
flops, params = clever_format([flops, params], '%.3f')
print('flops: ', flops, 'params: ', params)