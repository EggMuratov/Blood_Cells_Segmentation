class UNet(nn.Module):
    class ResNet_Block(nn.Module):
        def __init__(self, input_channels, output_channels):
            super().__init__()

            self.model_convolution = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )

            self.model_ResNet = nn.Sequential(
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels)
            )

        def forward(self, x):
            y = self.model_convolution(x)
            out = self.model_ResNet(y)
            return torch.nn.functional.relu(y + out)


    class Two_Conv_Layers(nn.Module):
        def __init__(self, input_channels, output_channels):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.model(x)

    class Encoder_Block(nn.Module):
        def __init__(self, input_channels, output_channels):
            super().__init__()
            self.Max_Pool_Layer = nn.MaxPool2d(2)
            self.ResNet = UNet.ResNet_Block(input_channels, output_channels)

        def forward(self, x):
            rn = self.ResNet(x)
            y = self.Max_Pool_Layer(rn)
            return y, rn


    class Decoder_Block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
            self.block = UNet.ResNet_Block(in_channels, out_channels)

        def forward(self, x, y):
            x = self.transpose(x)
            u = torch.cat([x, y], dim=1)
            u = self.block(u)
            return u

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.enc_block1 = self.Encoder_Block(in_channels, 32)
        self.enc_block2 = self.Encoder_Block(32, 64)
        self.enc_block3 = self.Encoder_Block(64, 128)
        self.enc_block4 = self.Encoder_Block(128, 256)

        self.bottleneck = self.Two_Conv_Layers(256, 512)

        self.dec_block1 = self.Decoder_Block(512, 256)
        self.dec_block2 = self.Decoder_Block(256, 128)
        self.dec_block3 = self.Decoder_Block(128, 64)
        self.dec_block4 = self.Decoder_Block(64, 32)

        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x, y1 = self.enc_block1(x)
        x, y2 = self.enc_block2(x)
        x, y3 = self.enc_block3(x)
        x, y4 = self.enc_block4(x)

        x = self.bottleneck(x)

        x = self.dec_block1(x, y4)
        x = self.dec_block2(x, y3)
        x = self.dec_block3(x, y2)
        x = self.dec_block4(x, y1)

        return torch.sigmoid(self.out(x))
