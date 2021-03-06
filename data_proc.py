import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import random
import pandas as pd


class LandmarkDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transform=None, img_size = 416):
        self.img_list = img_list
        self.transform = transform
        self.img_size = img_size
   
    def __getitem__(self, index):
        image = Image.open(self.img_list[index])
        label = self.getLabel(self.img_list[index])

        image = self.ResizeImage(image, (self.img_size, self.img_size))
        image = self.transform(image)

        #image = transforms.ToTensor()(image)
        
        sample = {"image":image[0:3], "label":label}
        return sample

    def __len__(self):
        return len(self.img_list)

    def getLabel(self, path):
        img_name = path.split('/')[-1]
        if img_name in label_lookup:
            label = label_lookup[img_name]
        else:
            label = -1
            
        return label

        return new_image

    def ResizeImage(self, img, size):
        width, height = img.size

        # if the original image is square
        if width == height:
            return img.resize(size)

        # determine width is larger or height is larger
        if width > height:
            pad_width = False
        else:
            pad_width = True

        if pad_width:
            padding = (height, height)
        else:
            padding = (width, width)
        
        ig = img
        new_width, new_height = padding

        scale = min(new_width / width, new_height / height)
        nw = int(width * scale)
        nh = int(height * scale)

        new_image = Image.new('RGB', padding, (255, 255, 255))  # white picture
        new_image.paste(ig, ((new_width - nw) // 2, (new_height - nh) // 2))

        return new_image.resize(size)

def get_all_img_path(root_folder, label_lookup):
    img_path = []
    length = len(label_lookup)
    counter = 1
    for img in label_lookup.keys():
        path = os.path.join(root_folder, img[0])
        path = os.path.join(path, img[1])
        path = os.path.join(path, img[2])
        path = os.path.join(path, img)
        #print(path)
        if os.path.exists(path):
            img_path.append(path)
        
        if counter % 10000 == 0:
            print('{}/{}'.format(counter, length))
        counter += 1
    return img_path

df = pd.read_csv('~/russell/google-landmark/413ProjectSync/label_lookup.csv', header = None)
label_lookup = {}
img_path = []
for row in df.index:
    label = df[1][row] 
    images = df[0][row]
    label_lookup[images] = label

    path = os.path.join("../train", images[0])
    path = os.path.join(path, images[1])
    path = os.path.join(path, images[2])
    path = os.path.join(path, images)

    # if os.path.exists(path):
    img_path.append(path)
print('# of images in lookup: {}'.format(len(label_lookup)))

#img_path = get_all_img_path("../train", label_lookup)
print('# of all images: {}'.format(len(img_path)))
print('example path: {}'.format(img_path[0]))

transform = transforms.Compose(
    [transforms.ToTensor()
    ])

train_dt = LandmarkDataset(img_path, transform, img_size = 416)

train_loader = torch.utils.data.DataLoader(train_dt, batch_size=16,
                        shuffle=True, num_workers=0)

sample = next(iter(train_loader))
sample["image"].shape, sample["label"].shape
plt.imshow(sample["image"][0].permute(1, 2, 0))