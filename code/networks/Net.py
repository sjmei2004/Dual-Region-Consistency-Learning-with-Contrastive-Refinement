import torch
from torch import nn, norm
import torch.nn.functional as F
from torch.nn.modules.conv import Conv3d
from torch.distributions.uniform import Uniform


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none', isrec=False):
        super(ConvBlock, self).__init__()
        if isrec:
            normalization = 'batchnorm'
        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', isrec=False):
        super(UpsamplingDeconvBlock, self).__init__()
        if isrec:
            normalization = 'batchnorm'
        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x





class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout


        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()


    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4


        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)

        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out



class feamap_model(nn.Module):

    def __init__(self, num_classes, ndf=64, out_channel=1):
        super(feamap_model, self).__init__()
        # downsample 16
        self.conv0 = nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d((7, 7, 5))
        self.avgpool1 = nn.AvgPool3d(7, 7, 5)
        # self.avgpool = nn.AvgPool3d((5, 7, 7))
        # self.avgpool = nn.AvgPool3d((5, 16, 16))
        self.fc1 = nn.Linear(ndf * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.leaky_relu = nn.Tanh()
        self.dropout = nn.Dropout3d(0.5)
        self.Softmax = nn.Softmax()
        self.out = nn.Conv3d(ndf * 2, num_classes, kernel_size=1)

    def forward(self, map):#2,1,112,112,80
        batch_size = map.shape[0]
        map_feature = self.conv0(map)  # (2,112,112,80)->(2,64,56,56,40)
        x = self.leaky_relu(map_feature)
        x = self.dropout(x)

        x = self.conv1(x)  # (2,64,56,56,40)->(2,128,28,28,20)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        # x = self.out(x)

        x = self.conv2(x)  # (2,128,28,28,20)->(2,256,14,14,10)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)  # (2,256,14,14,10)->(2,512,7,7,5)
        x = self.leaky_relu(x)

        x = self.avgpool(x)  # (2,512,1,1,1)
        # x = self.avgpool1(x)

        x = x.view(batch_size, -1)#2,512
        x = self.fc1(x)#2,512
        x = self.fc2(x)#2,2

        return x

class feamap_model_bra(nn.Module):

    def __init__(self, num_classes, ndf=64, out_channel=1):
        super(feamap_model_bra, self).__init__()
        # downsample 16
        self.conv0 = nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d((6, 6, 6))
        self.avgpool1 = nn.AvgPool3d(7, 7, 5)
        # self.avgpool = nn.AvgPool3d((5, 7, 7))
        # self.avgpool = nn.AvgPool3d((5, 16, 16))
        self.fc1 = nn.Linear(ndf * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.leaky_relu = nn.Tanh()
        self.dropout = nn.Dropout3d(0.5)
        self.Softmax = nn.Softmax()
        self.out = nn.Conv3d(ndf * 2, num_classes, kernel_size=1)

    def forward(self, map):#2,1,96,96,96
        batch_size = map.shape[0]
        map_feature = self.conv0(map)  # (2,1,96,96,96)->(2,64,48,48,48)
        x = self.leaky_relu(map_feature)
        x = self.dropout(x)

        x = self.conv1(x)  # (2,64,48,48,48)->(2,128,24,24,24)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        # x = self.out(x)

        x = self.conv2(x)  # (2,128,24,24,24)->(2,256,12,12,12)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)  # (2,256,12,12,12)->(2,512,6,6,6)
        x = self.leaky_relu(x)

        x = self.avgpool(x)  # (2,512,1,1,1)
        # x = self.avgpool1(x)

        x = x.view(batch_size, -1)#2,512
        x = self.fc1(x)#2,512
        x = self.fc2(x)#2,2

        return x



class center_model_bra(nn.Module):

    def __init__(self, num_classes, ndf=64, out_channel=1):
        super(center_model_bra, self).__init__()
        # downsample 16
        self.conv0 = nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d((6, 6, 6))
        self.avgpool1 = nn.AvgPool3d(7, 7, 5)
        # self.avgpool = nn.AvgPool3d((5, 7, 7))
        # self.avgpool = nn.AvgPool3d((5, 16, 16))
        self.fc1 = nn.Linear(ndf * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.leaky_relu = nn.Tanh()
        self.dropout = nn.Dropout3d(0.5)
        self.Softmax = nn.Softmax()
        self.out = nn.Conv3d(ndf * 2, num_classes, kernel_size=1)

    def forward(self, map):#2,1,96,96,96
        batch_size = map.shape[0]
        map_feature = self.conv0(map)  # (2,1,96,96,96)->(2,64,48,48,48)
        x = self.leaky_relu(map_feature)
        x = self.dropout(x)

        x = self.conv1(x)  # (2,64,48,48,48)->(2,128,24,24,24)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        # x = self.out(x)

        x = self.conv2(x)  # (2,128,24,24,24)->(2,256,12,12,12)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)  # (2,256,12,12,12)->(2,512,6,6,6)
        x = self.leaky_relu(x)

        x = self.avgpool(x)  # (2,512,1,1,1)
        # x = self.avgpool1(x)

        x = x.view(batch_size, -1)#2,512
        x = self.fc1(x)#2,512
        x = self.fc2(x)#2,2

        return x


