import numpy as np
import pandas as pd
import copy
import cv2
import rasterio
import csv


INFO = "//NAS/Download/BloodCells/sartorius-cell-instance-segmentation/train.csv"
NEW_DATASET_IMAGES = "//NAS/Download/BloodCells/sartorius-cell-instance-segmentation/dataset/all_images/images/"
NEW_DATASET_MASKS = "//NAS/Download/BloodCells/sartorius-cell-instance-segmentation/dataset/all_images/masks/"
TRAIN_DATASET_IMAGES = '//NAS/Download/BloodCells/sartorius-cell-instance-segmentation/dataset/train/images/'
TRAIN_DATASET_MASKS = "//NAS/Download/BloodCells/sartorius-cell-instance-segmentation/dataset/train/masks/"
TEST_DATASET_IMAGES = "//NAS/Download/BloodCells/sartorius-cell-instance-segmentation/dataset/test/images/"
TEST_DATASET_MASKS = "//NAS/Download/BloodCells/sartorius-cell-instance-segmentation/dataset/test/masks/"
DATA = "//NAS/Download/BloodCells/sartorius-cell-instance-segmentation/train/"


def params_of_coord_mask():
    params = dict()
    with open(INFO, encoding='utf8') as blood_cell_params:
        blood_cell_reader = csv.reader(blood_cell_params, delimiter=',')
        count = 0
        for i in blood_cell_reader:
            if count != 0:
                coordinates = i[1].split()
                mask_coord = [[int(coordinates[j]), int(coordinates[j + 1])]for j in range(0, len(coordinates), 2)]
                if i[0] in params:
                    params[i[0]] += mask_coord
                else:
                    params[i[0]] = mask_coord
            count += 1
        return params, params.keys()


def creating_mask(mask_list):
    h, w = 520, 704
    mask = np.empty((h, w), dtype=int)
    filled_pixels = np.zeros(h * w, dtype=int)
    for i in mask_list:
        for j in range(i[1]):
            filled_pixels[i[0] - 1 + j] = 255
    print("CREATE FILLED PIXELS")
    for i in range(h):
        mask[i] = filled_pixels[(w * i):(w * (i + 1))]
    #path_general_mask = DATA + "GENERAL_MASK.jpg"
    #cv2.imwrite(path_general_mask, mask)
    return mask


#def read_tiff(file_name):
 #   p = DATA + file_name + ".tiff"
  #  src = rasterio.open(p)
   # img = src.read()
    #return img


def splitting_on_dataset_256X256():
    coord_masks, names_of_images = params_of_coord_mask()
    print("READ NAMES AND MASKS COORDINATES")
    count = 1
    for name in names_of_images:
        print(name)
        mask_coord = coord_masks[name]
        print("Длина", len(mask_coord))
        general_mask = creating_mask(mask_coord)
        print("CREATE GENERAL MASK")
        image_png = cv2.imread("//NAS/Download/BloodCells/sartorius-cell-instance-segmentation/train/{}.png".format(name))
        print("READ PNG")
        size = image_png.shape
        width, height = 704, 520
        print(size, general_mask.shape)
        for h in range(0, height - 256, 12):
            for w in range(0, width - 256, 16):
                img = np.zeros([256, 256, 3], dtype=int)
                mask = np.zeros([256, 256], dtype=int)
                for k_h in range(256):
                    for k_w in range(256):
                        x = w + k_w
                        y = h + k_h
                        mask[k_h][k_w] = general_mask[y][x]
                        img[k_h][k_w][0] = image_png[y][x][0]
                        img[k_h][k_w][1] = image_png[y][x][1]
                        img[k_h][k_w][2] = image_png[y][x][2]
                path_img = NEW_DATASET_IMAGES + "img_{}.jpg".format(count)
                path_m = NEW_DATASET_MASKS + "mask_{}.jpg".format(count)
                # if count_colors == 1:
                #    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cv2.imwrite(path_img, img)
                cv2.imwrite(path_m, mask)
                print(str(count) + "-ое изображение готово " + name)
                count += 1


def splitting_on_train_and_test_dataset():
    for j in range(6, 373296, 6):
        i = j // 6
        img_path = NEW_DATASET_IMAGES + "img_{}.jpg".format(i)
        mask_path = NEW_DATASET_MASKS + "mask_{}.jpg".format(i)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if i % 44 == 0:
            st_img_path = TEST_DATASET_IMAGES + "img_{}.jpg".format(i // 44)
            st_mask_path = TEST_DATASET_MASKS + "mask_{}.jpg".format(i // 44)
        else:
            st_img_path = TRAIN_DATASET_IMAGES + "img_{}.jpg".format(i - (i // 44))
            st_mask_path = TRAIN_DATASET_MASKS + "mask_{}.jpg".format(i - (i // 44))

        cv2.imwrite(st_img_path, img)
        cv2.imwrite(st_mask_path, mask)


splitting_on_dataset_256X256()
#всего изображений - 373296
splitting_on_train_and_test_dataset()
#train dataset - 60802
#test dataset - 1413
