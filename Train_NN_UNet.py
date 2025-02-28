from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs_v2
from segmentation_models_pytorch import utils
import torch.nn as nn
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_DATASET_IMAGES = '//NAS/Download/BloodCells/sartorius-cell-instance-segmentation/dataset/train/images/img_'
TRAIN_DATASET_MASKS = "//NAS/Download/BloodCells/sartorius-cell-instance-segmentation/dataset/train/masks/mask_"
TEST_DATASET_IMAGES = "//NAS/Download/BloodCells/sartorius-cell-instance-segmentation/dataset/test/images/img_"
TEST_DATASET_MASKS = "//NAS/Download/BloodCells/sartorius-cell-instance-segmentation/dataset/test/masks/mask_"

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

    class Max_Pool_Block(nn.Module):
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
        self.enc_block1 = self.Max_Pool_Block(in_channels, 64)
        self.enc_block2 = self.Max_Pool_Block(64, 128)
        self.enc_block3 = self.Max_Pool_Block(128, 256)
        self.enc_block4 = self.Max_Pool_Block(256, 512)

        self.bottleneck = self.Two_Conv_Layers(512, 1024)

        self.dec_block1 = self.Decoder_Block(1024, 512)
        self.dec_block2 = self.Decoder_Block(512, 256)
        self.dec_block3 = self.Decoder_Block(256, 128)
        self.dec_block4 = self.Decoder_Block(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

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

        return self.out(x)


'''class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = nn.functional.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2 * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score
'''
class HuBMAPDataset(data.Dataset):
    def __init__(self, img_path, mask_path, num_img, transform_img=None, transform_mask=None):
        self.path_img = img_path
        self.path_mask = mask_path
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.images = []
        self.masks = []
        self.length = num_img - 1

        for i in range(1, num_img):
            self.images.append(img_path + str(i) + ".jpg")
            self.masks.append(mask_path + str(i) + ".jpg")

    def __getitem__(self, item):
        path_img, path_mask = self.images[item], self.masks[item]
        img = Image.open(path_img).convert('RGB')
        mask = Image.open(path_mask).convert('L')

        if self.transform_img:
            img = self.transform_img(img)

        if self.transform_mask:
            mask = self.transform_mask(mask)
            mask[mask < 250] = 1
            mask[mask >= 250] = 0

        return img, mask

    def __len__(self):
        return self.length


def Training_UNet(train_dataset, test_dataset):
    loss = utils.losses.DiceLoss()

    metrics = [
        utils.metrics.Fscore(),
        utils.metrics.IoU()
    ]
    model = UNet().to(device)
    optimizer = optim.RMSprop(params=model.parameters(), lr=0.001)

    train_epochs = utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True
    )

    test_epochs = utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True
    )

    epochs = 10
    best_result = 0
    loss_logs = {"train": [], "test": []}
    metrics_logs = {"train": [], "test": []}
    for i in range(epochs):
        print('\nEPOCHS: {}'.format(i + 1))
        train_logs = train_epochs.run((train_dataset))
        train_loss, train_metric, train_metric_IOU = list(train_logs.values())
        loss_logs["train"].append(train_loss)
        metrics_logs["train"].append(train_metric_IOU)

        test_logs = test_epochs.run((test_dataset))
        test_loss, test_metric, test_metric_IOU = list(test_logs.values())
        loss_logs["test"].append(test_loss)
        metrics_logs["test"].append(test_metric_IOU)

        if best_result < test_logs['iou_score']:
            best_result = test_logs['iou_score']
            torch.save(model, 'best_Blood_Cells_model_unet.pth')


tr_img = tfs_v2.Compose([tfs_v2.ToImage(), tfs_v2.ToDtype(torch.float32, scale=True)])
tr_mask = tfs_v2.Compose([tfs_v2.ToImage(), tfs_v2.ToDtype(torch.float32)])

d_train = HuBMAPDataset(TRAIN_DATASET_IMAGES, TRAIN_DATASET_MASKS, 60802, transform_img=tr_img, transform_mask=tr_mask)
d_test = HuBMAPDataset(TEST_DATASET_IMAGES, TEST_DATASET_MASKS, 1413, transform_img=tr_img, transform_mask=tr_mask)
train_data = data.DataLoader(d_train, batch_size=4, shuffle=True)
test_data = data.DataLoader(d_test, batch_size=1, shuffle=True)


Training_UNet(train_data, test_data)
