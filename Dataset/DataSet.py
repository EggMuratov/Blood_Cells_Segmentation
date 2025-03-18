class DataSet(data.Dataset):
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
        img = cv2.imread(path_img)
        mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
        if self.transform_img:
            img = self.transform_img(img)
            img = img.to(device)

        if self.transform_mask:
            mask = self.transform_mask(mask)
            mask[mask < 200] = 0
            mask[mask > 200] = 1
            mask = mask.to(device)

        return img, mask

    def __len__(self):
        return self.length
