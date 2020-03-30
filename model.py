import torch
import torch.nn as nn
import torch.nn.functional as F


#########################################################################################################


# coordconv =========================================================================================

class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5,
                                                                                 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        # print('before coordconv ',x.size())
        ret = self.addcoords(x)
        # print('after coordconv ', ret.size())
        ret = self.conv(ret)
        return ret


class CoordConv_block(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('before coordconv ',x.size())
        ret = self.addcoords(x)
        # print('after coordconv ', ret.size())
        ret = self.conv(ret)
        ret = self.sigmoid(ret)

        batch_size, channel, a, b = x.size()
        # spatial excitation
        output_tensor = torch.mul(x, ret.view(batch_size, 1, a, b))

        return output_tensor


def TF_coordconv(encodernumber, coordconv):
    TF_coordconv_list = []
    if coordconv == None:
        TF_coordconv_list = [False for i in range(encodernumber)]
    else:
        for i in range(0, encodernumber):
            if i in coordconv:
                TF_coordconv_list.append(True)
            else:
                TF_coordconv_list.append(False)

    assert len(TF_coordconv_list) == encodernumber, 'not match coordconv_list'

    return TF_coordconv_list


# coordconv =========================================================================================


# 2d model

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, momentum=0.1, coordconv=False,
                 radius=False):
        super(ConvBnRelu, self).__init__()

        if coordconv:
            # 1x1 conv
            self.conv = CoordConv(in_channels, out_channels, kernel_size=1,
                                  padding=0, stride=1, with_r=radius)
            # self.conv = CoordConv(in_channels, out_channels, kernel_size=kernel_size,
            #                     padding=1, stride=1, with_r=radius)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  padding=padding, stride=stride)

        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, momentum=0.1, coordconv=False, radius=False):
        super(StackEncoder, self).__init__()
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum)
        self.maxPool = nn.MaxPool2d(kernel_size=(4, 4), stride=4)

    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        x_trace = x
        x = self.maxPool(x)
        return x, x_trace


class StackDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, momentum=0.1, coordconv=False, radius=False):
        super(StackDecoder, self).__init__()

        # self.upSample = nn.Upsample(size=upsample_size, scale_factor=(2,2), mode='bilinear')
        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4, 4), stride=4)

        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum)

    def _crop_concat(self, upsampled, bypass):

        margin = bypass.size()[2] - upsampled.size()[2]
        c = margin // 2
        if margin % 2 == 1:
            bypass = F.pad(bypass, (-c, -c - 1, -c, -c - 1))
        else:
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):

        x = self.transpose_conv(x)
        if down_tensor != None:
            x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        x = self.convr2(x)
        return x



######################################################################################################
# model


class Unet2D(nn.Module):
    """
    ori unet : padding =1, momentum=0.1,coordconv =False
    down_crop unet : padding =0, momentum=0.1,coordconv =False
    """

    def __init__(self, in_shape, padding=1, momentum=0.1,start_channel=8):
        super(Unet2D, self).__init__()
        self.channels, self.heights, self.width = in_shape
        self.padding = padding
        self.start_channel = start_channel


        self.down1 = StackEncoder(self.channels, self.start_channel, padding, momentum=momentum)
        self.down2 = StackEncoder(self.start_channel, self.start_channel*2, padding, momentum=momentum)
        self.down3 = StackEncoder(self.start_channel*2, self.start_channel*4, padding, momentum=momentum)
        self.down4 = StackEncoder(self.start_channel*4, self.start_channel*8, padding, momentum=momentum)

        self.center = nn.Sequential(
            ConvBnRelu(self.start_channel*8, self.start_channel*16, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum),
            ConvBnRelu(self.start_channel*16, self.start_channel*16, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        )

        self.up1 = StackDecoder(in_channels=self.start_channel*16, out_channels=self.start_channel*8, padding=padding, momentum=momentum)
        self.up2 = StackDecoder(in_channels=self.start_channel*8, out_channels=self.start_channel*4, padding=padding, momentum=momentum)
        self.up3 = StackDecoder(in_channels=self.start_channel*4, out_channels=self.start_channel*2, padding=padding, momentum=momentum)
        self.up4 = StackDecoder(in_channels=self.start_channel*2, out_channels=self.start_channel, padding=padding, momentum=momentum)

        self.output_seg_map = nn.Conv2d(self.start_channel, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(self.heights, self.width), mode='nearest')

    def forward(self, x):
        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        center = self.center(x)

        x = self.up1(center, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)

        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)

        if out.shape[-1] != self.width:
            out = self.output_up_seg_map(out)

        return out,center






##########################################################################

if __name__ == '__main__':
    from torchsummary import summary

    my_net = Unet2D(in_shape=(1, 128, 128))

    summary(model=my_net.cuda(),input_size=(1,128,128))



