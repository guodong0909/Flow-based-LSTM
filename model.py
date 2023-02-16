
import torch
import torch.nn as nn

class SimplePnPNet(nn.Module):
    def __init__(self, nIn, batch_size):
        super(SimplePnPNet, self).__init__()

        self.batch_size = batch_size

        self.conv1 = torch.nn.Conv2d(nIn, 128, 1)
        self.conv2 = torch.nn.Conv2d(128, 128, 1)
        self.conv3 = torch.nn.Conv2d(128, 128, 1)

        self.conv4 = torch.nn.Conv1d(128, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 128, 1)
        self.conv6 = torch.nn.Conv1d(128, 128, 1)


        self.maxpool = torch.nn.MaxPool2d(kernel_size = 1, stride= 2)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_qt = nn.Linear(256, 7)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)

        x = x.view(self.batch_size, 128, 4800)
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.conv6(x)


        x = x.view(self.batch_size, 128, 300, 16)
        x = torch.max(x, dim=2, keepdim=True)[0]
        # x = torch.mean(x, dim=2, keepdim=True)

        x = x.view(self.batch_size, 2048)
        # 
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        # 
        qt = self.fc_qt(x)

        return qt


class MaskEX(nn.Module):
    def __init__(self, flow_size):
        super(MaskEX, self).__init__()

        self.flow_size = flow_size
        self.conv1 = nn.Conv2d(in_channels=2, out_channels = 128, kernel_size= 3, stride= 1, padding= 1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels = 128, kernel_size= 3, stride= 1, padding= 1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels = 128, kernel_size= 3, stride= 1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels = 1, kernel_size= 1, stride= 1, padding= 0)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = torch.abs(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))


        return x



class FCN32s(nn.Module):

    def __init__(self, flow_size, n_class):
        super().__init__()

        self.conv1_1 = nn.Conv2d(2, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 1024, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(1024, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 32, stride=32,
                                          bias=False, output_padding=0)

        

    def forward(self, x):

        h = torch.abs(x)

        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)


        h = self.score_fr(h)

        h = self.upscore(h)


        # h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return h  # size=(N, n_class, x.H/1, x.W/1)




