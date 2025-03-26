import os
import shutil
import torch
import numpy as np
import cv2
import torchvision.transforms.v2 as tfs_v2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # выбор устройства
tr_img = tfs_v2.Compose([tfs_v2.ToImage(), tfs_v2.ToDtype(torch.float32, scale=True)]) #преобразование изображения для того, чтобы подать его на вход модели
path = '' #путь до изображения, для которого необходимо получить маску
model_path = '' #путь до обученной модели
path_save_predicted = '' #путь, по которому будет сохранена маска
path_buffer_img = '' #буферная папака, в которою будут сохраняться изображения 256Х256 (всё содержимое очищается после отработки)
path_buffer_mask = '' #буферная папака, в которою будут сохраняться маски 256Х256 (всё содержимое очищается после отработки)


def splitting_photo(photo_path, buffer_path):
    print('SPLITTING PHOTO')
    image = cv2.imread(photo_path)
    height, width = 520, 704
    for h in range(0, height - 255, 24):
        for w in range(0, width - 255, 32):
            img = np.zeros([256, 256, 3], dtype=int)
            for k_h in range(256):
                for k_w in range(256):
                    x = w + k_w
                    y = h + k_h
                    img[k_h][k_w][0] = image[y][x][0]
                    img[k_h][k_w][1] = image[y][x][1]
                    img[k_h][k_w][2] = image[y][x][2]
            saving_path = buffer_path + f'img_{h // 24}_{w // 32}.jpg'
            cv2.imwrite(saving_path, img)


def getting_masks(model_path, buffer_path_img, buffer_path_masks):
    print('FORMING 256X256 MASKS')
    model = torch.load(model_path, map_location=device)
    model.eval()
    for i in range(12):
        for j in range(15):
            path = buffer_path_img + f'img_{i}_{j}.jpg'
            img = cv2.imread(path)
            img_tensor = tr_img(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
            output = output.detach().to('cpu').numpy()[0][0]

            mask = np.zeros((256, 256))
            for k in range(256):
                for s in range(256):
                    if 0.999999 < output[k][s] < 1.000001:
                        mask[k][s] = 255
            cv2.imwrite(buffer_path_masks + f'mask_{i}_{j}.jpg', mask)


def forming_general_mask(path_for_saving_predicting_general_mask, buffer_masks):
    general_mask = np.zeros((520, 704))
    for i in range(12):
        for j in range(15):
            path = buffer_masks + f'mask_{i}_{j}.jpg'
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            coord_start = (i * 24, j * 32)
            for y in range(256):
                for x in range(256):
                    h, w = coord_start[0] + y, coord_start[1] + x
                    general_mask[h][w] = mask[y][x]
    kernel = np.ones((3, 3))
    cleaned_mask = cv2.morphologyEx(general_mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    final_mask *= 255
    cv2.imwrite(path_for_saving_predicting_general_mask, final_mask)
    print('SAVE MASK')


def clean_directory(path):
    shutil.rmtree(path)
    os.mkdir(path)



splitting_photo(path, path_buffer_img)
print(f"SPLITTING IS OVER")

getting_masks(model_path, path_buffer_img, path_buffer_mask)
print(f"GETTING MASKS IS OVER")

forming_general_mask(path_save_predicted, path_buffer_mask)
print(f"FORMING GENERAL MASK IS OVER")

clean_directory(path_buffer_img)
clean_directory(path_buffer_mask)
