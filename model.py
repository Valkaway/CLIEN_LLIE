import torch
import torch.nn as nn

from config import config

class LLIE_Model(nn.Module):
    def __init__(self):
        super(LLIE_Model, self).__init__()

        self.d_net = Decom()
        self.i_net = Illum()
        self.c_net = c_net()
        self.r_net = r_net()

    def forward(self, img, ratio):
        img_R, img_L = self.d_net(img)
        after_L = self.i_net(img_L, ratio)
        color_hist, color_feature = self.c_net(img)
        img_enhance = self.r_net(img_R, after_L, color_feature)
        return [img_R, img_L, after_L], color_hist, img_enhance



class Decom(nn.Module):
    def __init__(self):
        super(Decom, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.pooling = nn.MaxPool2d((2, 2), stride=2)
        self.r_conv2 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.r_conv3 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.deconv1 = nn.ConvTranspose2d(128, 64, (3, 3), stride=2, padding=1, output_padding=1)
        self.r_conv4 = nn.Conv2d(128, 64, (3, 3), padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, (3, 3), stride=2, padding=1, output_padding=1)
        self.r_conv5 = nn.Conv2d(64, 32, (3, 3), padding=1)
        self.r_conv6 = nn.Conv2d(32, 3, (3, 3), padding=1)
        self.l_conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.l_conv3 = nn.Conv2d(64, 1, (3, 3), padding=1)

    def forward(self, input_img):
        conv1 = self.relu(self.conv1(input_img))
        r_pool1 = self.pooling(conv1)
        r_conv2 = self.relu(self.r_conv2(r_pool1))
        r_pool2 = self.pooling(r_conv2)
        r_conv3 = self.relu(self.r_conv3(r_pool2))
        r_deconv1 = self.deconv1(r_conv3)
        r_concat1 = torch.cat((r_deconv1, r_conv2), 1)
        r_conv4 = self.relu(self.r_conv4(r_concat1))
        r_deconv2 = self.deconv2(r_conv4)
        r_concat2 = torch.cat((r_deconv2, conv1), 1)
        r_conv5 = self.relu(self.r_conv5(r_concat2))
        R_out = self.sigmoid(self.r_conv6(r_conv5))
        l_conv2 = self.relu(self.l_conv2(conv1))
        l_concat = torch.cat((l_conv2, r_conv5), 1)
        L_out = self.sigmoid(self.l_conv3(l_concat))
        return R_out, L_out



class Illum(nn.Module):
    def __init__(self):
        super(Illum, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(2, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 1, (3, 3), padding=1)

    def forward(self, input_L, ratio):
        x = torch.cat([input_L, ratio], 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x



class c_net(nn.Module):
    def __init__(self):
        super(c_net, self).__init__()

        # encoder
        self.Encoder = nn.Sequential(
            nn.Sequential(nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU()),
            RB(32),
            nn.Sequential(nn.Conv2d(32, 64, (3, 3), stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(64, 64, (3, 3), stride=1, padding=1), nn.ReLU()),
            RB(64),
            nn.Sequential(nn.Conv2d(64, 128, (3, 3), stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(128, 128, (3, 3), stride=1, padding=1), nn.ReLU()),
            RB(128)
        )

        self.conv_in = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.relu = nn.ReLU()

        # color hist
        self.conv_color = nn.Conv2d(128, 256*3, (3, 3), stride=1, padding=1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, config['d_hist'])
        self.softmax = nn.Softmax(dim=2)


    def color_forward(self, x):
        x = self.relu(self.conv_color(x))
        x = self.pooling(x)
        x = torch.reshape(x, (-1, 3, 256))
        color_hist = self.softmax(self.fc(x))
        return color_hist

    def forward(self, x):
        x = self.relu(self.conv_in(x))
        x = self.Encoder(x)
        color_hist = self.color_forward(x)
        return color_hist, x



class r_net(nn.Module):
    def __init__(self):
        super(r_net, self).__init__()

        # encoder
        self.Encoder = nn.ModuleList([
            nn.Sequential(nn.Conv2d(32, 32, (3, 3), stride=1, padding=1), nn.ReLU()),
            RB(32),
            nn.Sequential(nn.Conv2d(32, 64, (3, 3), stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(64, 64, (3, 3), stride=1, padding=1), nn.ReLU()),
            RB(64),
            nn.Sequential(nn.Conv2d(64, 128, (3, 3), stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(128, 128, (3, 3), stride=1, padding=1), nn.ReLU()),
            RB(128),
            nn.Sequential(nn.Conv2d(128, 256, (3, 3), stride=2, padding=1), nn.ReLU()),
        ])

        # Middle
        # self.middle = RB(256)

        # decoder
        self.Decoder = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(256, 128, (4, 4), stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(256, 128, (3, 3), stride=1, padding=1), nn.ReLU()),
            RB(128),
            nn.Sequential(nn.ConvTranspose2d(128, 64, (4, 4), stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(128, 64, (3, 3), stride=1, padding=1), nn.ReLU()),
            RB(64),
            nn.Sequential(nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(64, 32, (3, 3), stride=1, padding=1), nn.ReLU()),
            RB(32)
        ])

        # conv
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv_in = nn.Conv2d(4, 32, (3, 3), stride=1, padding=1)
        self.conv_out = nn.Conv2d(32, 3, (3, 3), stride=1, padding=1)
        self.pce = pce()

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts

    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i // 3 + 1)
                # print(index, x.shape, shortcuts[index].shape)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x

    def forward(self, img_R, img_L, color_feature):
        x = torch.cat([img_R, img_L], 1)
        x = self.relu(self.conv_in(x))
        x, shortcuts = self.encoder(x)
        # x = self.middle(x)
        shortcuts = self.pce(color_feature, shortcuts)
        x = self.decoder(x, shortcuts)
        img = self.sigmoid(self.conv_out(x))
        return img



class pce(nn.Module):
    def __init__(self):
        super(pce, self).__init__()

        self.cma_3 = cma(128, 64)
        self.cma_2 = cma(64, 32)
        self.cma_1 = cma(32, 16)

    def forward(self, c, shortcuts):
        # change channels
        x_3_color, c_2 = self.cma_3(c, shortcuts[2])
        x_2_color, c_1 = self.cma_2(c_2, shortcuts[1])
        x_1_color, _ = self.cma_1(c_1, shortcuts[0])
        return [x_1_color, x_2_color, x_3_color]



class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.relu = nn.ReLU()
        self.layer_1 = nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1)
        self.layer_2 = nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1)

    def forward(self, x):
        y = self.relu(self.layer_1(x))
        y = self.relu(self.layer_2(y))
        return y + x



class cma(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cma, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 3), 1, 1),
                                  nn.InstanceNorm2d(out_channels),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, c, x):
        # x: gray image features
        # c: color features

        # l1 distance
        channels = c.shape[1]
        sim_mat_l1 = -torch.abs(x - c)  # <0  (b,c,h,w)
        sim_mat_l1 = torch.sum(sim_mat_l1, dim=1, keepdim=True)  # (b,1,h,w)
        sim_mat_l1 = torch.sigmoid(sim_mat_l1)  # (0, 0.5) (b,1,h,w)
        sim_mat_l1 = sim_mat_l1.repeat(1, channels, 1, 1)
        sim_mat_l1 = 2 * sim_mat_l1  # (0, 1)

        # cos distance
        sim_mat_cos = x * c  # >0 (b,c,h,w)
        sim_mat_cos = torch.sum(sim_mat_cos, dim=1, keepdim=True)  # (b,1,h,w)
        sim_mat_cos = torch.tanh(sim_mat_cos)  # (0, 1) (b,1,h,w)
        sim_mat_cos = sim_mat_cos.repeat(1, channels, 1, 1)  # (0, 1)

        # similarity matrix
        sim_mat = sim_mat_l1 * sim_mat_cos  # (0, 1)

        # color embeding
        x_color = x + c * sim_mat

        # color features upsample
        c_up = self.conv(c)

        return x_color, c_up

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.pooling = nn.MaxPool2d((2, 2), stride=2)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 1, (3, 3), padding=1)
        self.fc1 = nn.Linear(int((config['picture_size'][0]*config['picture_size'][1])/16), 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pooling(x)
        x = self.relu(self.conv2(x))
        x = self.pooling(x)
        x = self.relu(self.conv3(x))
        x = x.reshape([x.shape[0], int((config['picture_size'][0]*config['picture_size'][1])/16)])
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
